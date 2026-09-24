 
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#if defined(_OPENMP)
#include <omp.h>
#define GET_TIME() (omp_get_wtime()) // wall time
#else
#define GET_TIME() ((double)clock() / CLOCKS_PER_SEC) // cpu time  (clock() retourne le nombre de coups de clock depuis le lancement du programme et CLOCK_PER_SEC est une constante qui donne le nbre de coups de clock par sec)
#endif

struct data {
  const char *name;
  int nx, ny;
  double dx, dy, *values;
};

#define GET(data, i, j) ((data)->values[(data)->nx * (j) + (i)])  //Macros pour accéder aux valeurs dans le tableau 1D comme si c’était 2D: GET(&ez, 2, 3) → valeur en (i=2, j=3) du champ Ez (data est un pointeur de structure)
// (data)->values[...] rend l'élément (i,j) de values qui est en argument de la structure data
#define SET(data, i, j, val) ((data)->values[(data)->nx * (j) + (i)] = (val))  //SET(&ez, i, j, val) → assigne la valeur val au point (i,j)


/*****************************************************************************************************************************/


int init_data(struct data *data, const char *name, int nx, int ny, double dx,  //struct data *data signifie : “data est un pointeur vers une structure data
              double dy, double val)
{
  data->name = name;  //A la variable const char *name de la structure data, on associe la vaiable const char *name qui est elle, en argument de la fonction int init_data
  data->nx = nx;  //A la variable int nx de la structure data, on associe la variable int nx qui est elle, en argument de la fonction int init_data (dimensions de la grille selon x)
  data->ny = ny;
  data->dx = dx;  //A la variable double dx de la structure data, on associe la variable double dx qui est elle, en argument de la fonction int init_data (pas de la grille selon x)
  data->dy = dy;
  data->values = (double *)malloc(nx * ny * sizeof(double));
  if(!data->values) {
    printf("Error: Could not allocate data\n");
    return 1;
  }
  for(int i = 0; i < nx * ny; i++) data->values[i] = val;
  return 0;
}


/*****************************************************************************************************************************/


void free_data(struct data *data) { free(data->values); }


/*****************************************************************************************************************************/


int write_data_vtk(struct data *data, int step, int rank) //Fonction qui vérifie si le nom du fichier n'est pas tropl long lorsque l'on en aura écrit (strlen -> taille du string)
{
  char out[512];
  if(strlen(data->name) > 256) {
    printf("Error: data name too long for output VTK file\n");
    return 1;
  }
  sprintf(out, "%s_rank%d_%d.vti", data->name, rank, step);

  FILE *fp = fopen(out, "wb");  //ouvre le fichier en mode binaire, c'est pour Paraview en gros
  if(!fp) {
    printf("Error: Could not open output VTK file '%s'\n", out);
    return 1;
  }

  uint64_t num_points = data->nx * data->ny;  //Calcule le nombre totale de points dans la grille. uint64_t est un entier non signé sur 64 bits
  uint64_t num_bytes = num_points * sizeof(double);  //Calcule le nombre d’octets nécessaires pour stocker tous les points. sizeof(double) = nombre d’octets d’un double (généralement 8)

  fprintf(fp, "<?xml version=\"1.0\"?>\n"
              "<VTKFile"
              " type=\"ImageData\""
              " version=\"1.0\""
              " byte_order=\"LittleEndian\""
              " header_type=\"UInt64\""
              ">\n"
              "  <ImageData"
              " WholeExtent=\"0 %d 0 %d 0 %d\""
              " Spacing=\"%lf %lf %lf\""
              " Origin=\"%lf %lf %lf\""
              ">\n"
              "    <Piece Extent=\"0 %d 0 %d 0 %d\">\n"
              "      <PointData Scalars=\"scalar_data\">\n"
              "        <DataArray"
              " type=\"Float64\""
              " Name=\"%s\""
              " format=\"appended\""
              " offset=\"0\""
              ">\n"
              "        </DataArray>\n"
              "      </PointData>\n"
              "    </Piece>\n"
              "  </ImageData>\n"
              "  <AppendedData encoding=\"raw\">\n_",
          data->nx - 1, data->ny - 1, 0,
          data->dx, data->dy, 0.,
          0., 0., 0.,
          data->nx - 1, data->ny - 1, 0,
          data->name);

  fwrite(&num_bytes, sizeof(uint64_t), 1, fp);  //Ecrit le nombre d'octets dans un fichier
  fwrite(data->values, sizeof(double), num_points, fp);  //Ecrit les valeurs du champ dans un fichier

  fprintf(fp, "  </AppendedData>\n"
              "</VTKFile>\n");

  fclose(fp);

  return 0;
}


/*****************************************************************************************************************************/


int write_manifest_vtk(const char *name, double dt, int nt, int sampling_rate,
                       int numranks)
{
  char out[512];
  if(strlen(name) > 256) {
    printf("Error: name too long for Paraview manifest file\n");
    return 1;
  }
  sprintf(out, "%s.pvd", name);

  FILE *fp = fopen(out, "wb");
  if(!fp) {
    printf("Error: Could not open output VTK manifest file '%s'\n", out);
    return 1;
  }

  fprintf(fp, "<VTKFile"
              " type=\"Collection\""
              " version=\"0.1\""
              " byte_order=\"LittleEndian\">\n"
              "  <Collection>\n");

  for(int n = 0; n < nt; n++) {  //Pour chaque pas de temps échantillonné, ajoute une entrée .vti dans le fichier .pvd.
    if(sampling_rate && !(n % sampling_rate)) {
      double t = n * dt;
      for(int rank = 0; rank < numranks; rank++) {
        fprintf(fp, "    <DataSet"
                    " timestep=\"%g\""
                    " part=\"%d\""
                    " file='%s_rank%d_%d.vti'/>\n",
                t, rank, name, rank, n);
      }
    }
  }

  fprintf(fp, "  </Collection>\n"
              "</VTKFile>\n");
  fclose(fp);
  return 0;
}


/*******************************************************************************************************************/
/*************************    MAIN    ******************************************************************************/
/*******************************************************************************************************************/


int main(int argc, char **argv)
{
  MPI_Init(&argc , &argv);

  /******** Déclaration des variables "non-MPI" ********/
  double dx = 1., dy = 1., dt = 1.;
  int nx = 1, ny = 1, nt = 1, sampling_rate = 1;
  double eps = 8.854187817e-12;
  double mu = 1.2566370614359173e-06;

  dx = dy = (3.e8 / 2.4e9) / 40.; // wavelength / 40
  nx = ny = 16000;
  dt = 0.5 / (3.e8 * sqrt(1. / (dx * dx) + 1. / (dy * dy))); // cfl / 2
  nt = 500;
  sampling_rate = 0; // don't save results

  //Constantes dont on a besoin
  double chy = dt / (dy * mu);
  double chx = dt / (dx * mu);
  double cex = dt / (dx * eps);
  double cey = dt / (dy * eps);


  /******** Déclaration des variables "MPI" ********/
  int Px = 16;
  int Py = 16;

  int rank, num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks); // On met MPI_COMM_WORLD pour avoir le nombre de ranks available dans num_ranks



if(rank == 0) {
    printf("MPI 2D decomposition: %d ranks = %d x %d grid\n", num_ranks, Px, Py);
    printf("Global grid: %d x %d\n", nx, ny);
  }

  // Subdomain coordinates
  const int px = rank % Px;
  const int py = rank / Px;
  // Subdomain sizes
  int nx_loc0 = nx / Px;
  int ny_loc0 = ny / Py;
  int rx = nx % Px;  // On prend en compte le fait que les domaines sur le bord droit et le bord du dessous, ont un peu plus de points que les autres ranks car on a pas forcément que nx*ny est divisible par Px*Py.
  int ry = ny % Py;

  const int nx_loc = nx_loc0 + ((px == Px-1) ? rx : 0);
  const int ny_loc = ny_loc0 + ((py == Py-1) ? ry : 0);
  //Subdomain starts and ends
  int istart = px * nx_loc0;
  int jstart = py * ny_loc0;
  int iend = istart + nx_loc - 1;
  int jend = jstart + ny_loc - 1;



  //On vérifie que les données ont pu être allouées sur chacun des ranks 
  struct data ez, hx, hy;
  if(init_data(&ez, "ez", nx_loc + 2, ny_loc + 2, dx, dy, 0.) ||  //Pour chaque champ, on alloue sur chaque rank, ((nx_loc + 2) * (ny_loc + 2)) points (car ghost cells)
    init_data(&hx, "hx", nx_loc + 2, ny_loc + 2, dx, dy, 0.) ||  //init_data() renvoie 0 si allocation OK, et -1 si échec => if est true si un échec et on arrête le processus, si non les buffers sont bien alloués et initié
    init_data(&hy, "hy", nx_loc + 2, ny_loc + 2, dx, dy, 0.)) {
  printf("Rank %d: Error: could not allocate data\n", rank);
  MPI_Abort(MPI_COMM_WORLD, 1);
  }


  const int left_rank = (rank % Px == 0)      ?  MPI_PROC_NULL : rank - 1;  // MPI_PROC_NULL return immédiatement et ne performe aucune opération (que ce soit pour un recv ou un send)
  const int right_rank = (rank % Px == Px-1)  ?  MPI_PROC_NULL : rank + 1;
  const int up_rank = (rank < Px)             ?  MPI_PROC_NULL : rank - Px;
  const int down_rank = (rank >= Px*(Py-1))   ?  MPI_PROC_NULL : rank + Px;

  //On envoie et reçoit des lignes de taille nx_loc vers le haut et le bas, et on envoie et reçoit des colonnes de taille ny_loc vers la gauche et la droite
  // Buffers pour échanges
  double *send_ez_top = malloc(nx_loc * sizeof(double));
  double *send_ez_bottom = malloc(nx_loc * sizeof(double));
  double *send_ez_left = malloc(ny_loc * sizeof(double));
  double *send_ez_right = malloc(ny_loc * sizeof(double));
  double *recv_ez_top = malloc(nx_loc * sizeof(double));
  double *recv_ez_bottom = malloc(nx_loc * sizeof(double));
  double *recv_ez_left = malloc(ny_loc * sizeof(double));
  double *recv_ez_right = malloc(ny_loc * sizeof(double));

  double *send_hx_top = malloc(nx_loc * sizeof(double));
  double *send_hx_bottom = malloc(nx_loc * sizeof(double));
  double *recv_hx_top = malloc(nx_loc * sizeof(double));
  double *recv_hx_bottom = malloc(nx_loc * sizeof(double));

  double *send_hy_left = malloc(ny_loc * sizeof(double));
  double *send_hy_right = malloc(ny_loc * sizeof(double));
  double *recv_hy_left = malloc(ny_loc * sizeof(double));
  double *recv_hy_right = malloc(ny_loc * sizeof(double));

  //On vérifie que les allocations se soient faites avec succès
  if(!send_ez_top || !send_ez_bottom || !send_ez_left || !send_ez_right ||
     !recv_ez_top || !recv_ez_bottom || !recv_ez_left || !recv_ez_right ||
     !send_hx_top || !send_hx_bottom || !recv_hx_top || !recv_hx_bottom ||
     !send_hy_left || !send_hy_right || !recv_hy_left || !recv_hy_right) {
    fprintf(stderr, "Rank %d: failed to allocate comm buffers\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  double start = MPI_Wtime();


  for(int n = 0; n < nt; n++) //Boucle sur tous les pas de temps et affiche régulièrement une estimation du temps restant (ETA)
  {

    if(rank == 0 && n && (n % (nt / 10)) == 0) {
      double time_sofar = MPI_Wtime() - start;
      double eta = (nt - n) * time_sofar / n;
      printf("Computing time step %d/%d (ETA: %g seconds)     \r", n, nt, eta);
      fflush(stdout);
    }


    /************* update hx and hy locally *************/
    for(int j = 1; j < ny_loc + 1; j++) { //On mets à jour les points au milieu, les points sans les ghost cells. Donc de j=1 à j=ny_loc et i=1 à i=nx_loc
      for(int i = 1; i < nx_loc + 1; i++) {
        double hx_ij =
          GET(&hx, i, j) - chy * (GET(&ez, i, j + 1) - GET(&ez, i, j));
        SET(&hx, i, j, hx_ij);
      }
    }
    for(int j = 1; j < ny_loc + 1; j++) {
      for(int i = 1; i < nx_loc + 1; i++) {
        double hy_ij =
          GET(&hy, i, j) + chx * (GET(&ez, i + 1, j) - GET(&ez, i, j));
        SET(&hy, i, j, hy_ij);
      }
    }


    /************* Préparation des buffers de Hx top et bottom, ainsi que des buffers de Hy left et right *************/
    for(int i=0 ; i < nx_loc ; i++) //On envoie des rows de taille nx_loc
    {
      send_hx_top[i] = GET(&hx , i+1 , 1);
      send_hx_bottom[i] = GET(&hx , i+1 , ny_loc);
    }
    for(int j=0 ; j < ny_loc ; j++) //On envoie des columns de tailles ny_loc
    {
      send_hy_left[j] = GET(&hy , 1 , j+1);
      send_hy_right[j] = GET(&hy , nx_loc , j+1);
    }
    

    /************* Communications pour Hx (top/bottom) et Hy (left/right) *************/
    // send top to up_rank, receive bottom from down_rank
    MPI_Sendrecv(send_hx_top, nx_loc, MPI_DOUBLE, up_rank, 10,  //Avec un Sendrecv, on est sûr de ne jamais avoir de deadlock à cause de dépendance cyclique, c'est géré automatiquement, on doit juste l'appeler 2 fois
                 recv_hx_bottom, nx_loc, MPI_DOUBLE, down_rank, 10,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
     // send bottom to down_rank, receive top from up_rank
    MPI_Sendrecv(send_hx_bottom, nx_loc, MPI_DOUBLE, down_rank, 11,
                 recv_hx_top, nx_loc, MPI_DOUBLE, up_rank, 11,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
     // send left to left_rank, receive right from right_rank
    MPI_Sendrecv(send_hy_left, ny_loc, MPI_DOUBLE, left_rank, 12,
                 recv_hy_right, ny_loc, MPI_DOUBLE, right_rank, 12,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // send right to right_rank, receive left from left_rank
    MPI_Sendrecv(send_hy_right, ny_loc, MPI_DOUBLE, right_rank, 13,
                 recv_hy_left, ny_loc, MPI_DOUBLE, left_rank, 13,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);


    /************* Rangement des ghost rows et ghost columns reçues "aux lignes j=0 et j = nx_loc + 1, et aux colonnes i=0 et i = ny_loc + 1"  *************/
    for(int i = 0; i < nx_loc; i++) 
    {
      SET(&hx, i+1, 0, recv_hx_top[i]);           // top
      SET(&hx, i+1, ny_loc+1, recv_hx_bottom[i]); // bottom
    }
    for(int j = 0; j < ny_loc; j++) 
    {
      SET(&hy, 0, j+1, recv_hy_left[j]);         // left 
      SET(&hy, nx_loc+1, j+1, recv_hy_right[j]); // right               
    }
    

    /************* update de ez en utilisant les ghosts cells reçues *************/
    for(int j = 1; j < ny_loc + 1; j++) {
      for(int i = 1; i < nx_loc + 1; i++) {
        double ez_ij = GET(&ez, i, j)
          + cex * (GET(&hy, i, j) - GET(&hy, i-1, j))
          - cey * (GET(&hx, i, j) - GET(&hx, i, j-1));
        SET(&ez, i, j, ez_ij);
      }
    }


    /************* Injection de la source au centre de la grille globale *************/
    const int center_global_i = nx/2;
    const int center_global_j = ny/2;
    double t = n * dt;
    if((istart <= center_global_i && center_global_i <= iend)  &&  
       (jstart <= center_global_j && center_global_j <= jend)) //Ce if est true si on est le rank qui contient le point du centre du réseau total
    {
      //On converti les coordonnées globale en coordonnées locales, pour que le rank centrale impose la source en son centre selon ses coordonnées locales (local_i, local_j)
      int local_i = center_global_i - istart + 1; // +1 car on a une colonnes de ghost cells à gauche en plus
      int local_j = center_global_j - jstart + 1; // +1 car on a une ligne de ghost cells en haut en plus
      // simple harmonic source
      SET(&ez, local_i, local_j, sin(2. * M_PI * 2.4e9 * t));
    }


    /************* Préparation des buffers de Ez *************/
    for (int i = 0; i < nx_loc; i++) {
      send_ez_top[i]    = GET(&ez, i+1, 1);
      send_ez_bottom[i] = GET(&ez, i+1, ny_loc);
    }

    for (int j = 0; j < ny_loc; j++) {
      send_ez_left[j]  = GET(&ez, 1, j+1);
      send_ez_right[j] = GET(&ez, nx_loc, j+1);
    }

    /************* Communications pour Ez (top/bottom et left/right) *************/
    //Send top to up_rank, receive bottom from down_rank
    MPI_Sendrecv(send_ez_top, nx_loc, MPI_DOUBLE, up_rank, 20,
             recv_ez_bottom, nx_loc, MPI_DOUBLE, down_rank, 20,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //Send bottom to down_rank, receive top from top_rank
    MPI_Sendrecv(send_ez_bottom, nx_loc, MPI_DOUBLE, down_rank, 21,
                recv_ez_top, nx_loc, MPI_DOUBLE, up_rank, 21,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //Send left to left_rank, receive right from right_rank
    MPI_Sendrecv(send_ez_left, ny_loc, MPI_DOUBLE, left_rank, 22,
                recv_ez_right, ny_loc, MPI_DOUBLE, right_rank, 22,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //Send rigt to right_rank, receive left from left_rank
    MPI_Sendrecv(send_ez_right, ny_loc, MPI_DOUBLE, right_rank, 23,
                recv_ez_left, ny_loc, MPI_DOUBLE, left_rank, 23,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);


    /************* Rangement des ghost rows et ghost columns reçues "aux lignes j=0 et j = nx_loc + 1, et aux colonnes i=0 et i = ny_loc + 1" pour le champ Ez *************/
    for (int i = 0; i < nx_loc; i++) {
        SET(&ez, i+1, 0,        recv_ez_top[i]);
        SET(&ez, i+1, ny_loc+1, recv_ez_bottom[i]);
    }

    for (int j = 0; j < ny_loc; j++) {
        SET(&ez, 0, j+1, recv_ez_left[j]);
        SET(&ez, nx_loc+1, j+1, recv_ez_right[j]);
    }








    /************* Ecriture des data dans un fichier .vtk *************/
    if(sampling_rate && !(n % sampling_rate)) {  //Sauvegarde le champ Ez en .vti tous les sampling_rate pas de temps.
      write_data_vtk(&ez, n, rank);
      // write_data_vtk(&hx, n, 0);
      // write_data_vtk(&hy, n, 0);
    }
  }// fin boucle for(n)



  // write VTK manifest, linking to individual step data files
  //if(rank == 0){
  //write_manifest_vtk("ez", dt, nt, sampling_rate, num_ranks);  //Crée un fichier .pvd pour regrouper tous les .vti.
  // write_manifest_vtk("hx", dt, nt, sampling_rate, 1);
  // write_manifest_vtk("hy", dt, nt, sampling_rate, 1);
  //}


  double end = MPI_Wtime();
  double local_time = end - start;
  double max_time;
  MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); //Combine toutes les data de tous les ranks ensemble dans un communicateur en un seul résultat en utilisant l'opération de réduction MPI_MAX
                                                                                 // On envoie 1 élément au root (rank 0)
  if(rank == 0) {
    printf("Total time: %g s\n", max_time); //Le temps requis pour faire les cacluls est celui du rank qui a pris le plus de temps
    printf("Performance: %g MUpdates/s\n",
           1e-6 * (double)nx * (double)ny * (double)nt / max_time); //Affiche le temps total d’exécution et la performance en millions de mises à jour par seconde. (nx*ny*nt = nbre total d'updates)
  }



//On free les data des 3 champs
  free_data(&ez);
  free_data(&hx);
  free_data(&hy);
  //On free les data des buffers de communication pour le champ Hx
  free(send_hx_top); free(recv_hx_top); 
  free(send_hx_bottom); free(recv_hx_bottom);
  //On free les data des buffers de communication pour le champ Hy
  free(send_hy_left); free(recv_hy_left); 
  free(send_hy_right); free(recv_hy_right);
  //On free les data des buffers de communication pour le champ Ez
  free(send_ez_top); free(recv_ez_top);              
  free(send_ez_bottom); free(recv_ez_bottom);
  free(send_ez_left); free(recv_ez_left);
  free(send_ez_right); free(recv_ez_right);


  MPI_Finalize();
  return 0;
}
