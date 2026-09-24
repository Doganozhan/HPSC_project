#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>


#define BLOCK_X 32
#define BLOCK_Y 8
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


#define CHECK_CUDA(call) do {                                 \
  cudaError_t err = (call);                                   \
  if(err != cudaSuccess) {                                    \
    fprintf(stderr, "CUDA check failed at %s:%d: %s\n",                 \
            __FILE__, __LINE__, cudaGetErrorString(err));     \
    exit(EXIT_FAILURE);                                       \
  }                                                           \
} while(0)  // Comme ça, quand on appellera la macro plus tard, on devra mettre un ";", comme si c'était une fonction, comme avec un do...while() il faut un ";". En gros le fait que la macro soit sur plusieurs lignes
// peut poser problème avec les structures de branching, le fait de mettre un do...while(0) fait que la macro est vue comme une seule et unique instruction.  
// Et ici on se dit: tient pourquoi utiliser une structure de branching dans une macro qu'on va utiliser à foison par la suite alors que si 2 threads prennent différents chemins dans le branching, ça ralentira 
// les performances ? Tout simplement par ce qu'il n'est pas supposé y avoir d'erreur, donc tous les threads sont supposés prendre le même chemin.


struct data_host {
  const char *name;
  int nx, ny;
  float dx, dy, *values;
};

/****************************************************************************************************************************/


int init_host_data(struct data_host *data, const char *name, int nx, int ny, float dx, float dy, float val)
{
  data->name = name;  //A la variable const char *name de la structure data, on associe la vaiable const char *name qui est elle, en argument de la fonction int init_data
  data->nx = nx;  //A la variable int nx de la structure data, on associe la variable int nx qui est elle, en argument de la fonction int init_data (dimensions de la grille selon x)
  data->ny = ny;
  data->dx = dx;  //A la variable double dx de la structure data, on associe la variable double dx qui est elle, en argument de la fonction int init_data (pas de la grille selon x)
  data->dy = dy;
  data->values = (float *)malloc(nx * ny * sizeof(float));
  if(!data->values) {
    printf("Error: Could not allocate data\n");
    return 1;
  }
  for(int i = 0; i < nx * ny; i++) data->values[i] = val;
  return 0;
}


/****************************************************************************************************************************/


__host__ __device__ float get(const float* field, int nx, int i, int j)
{
  return field[j*nx + i];
}


/****************************************************************************************************************************/


__host__ __device__ void set(float* field, int nx, int i, int j, float val)
{
  field[j*nx + i] = val;
}


/****************************************************************************************************************************/


/* Version de Write data VTK pour les type float, c'est le seul truc de changé (avant c'était en double)*/
int write_data_vtk_float(struct data_host *data, int step, int rank)
{
  char out[512];
  if(strlen(data->name) > 256) {
    printf("Error: data name too long for output VTK file\n");
    return 1;
  }
  sprintf(out, "%s_rank%d_%d.vti", data->name, rank, step);

  FILE *fp = fopen(out, "wb");
  if(!fp) {
    printf("Error: Could not open output VTK file '%s'\n", out);
    return 1;
  }

  uint64_t num_points = (uint64_t)data->nx * (uint64_t)data->ny;
  uint64_t num_bytes = num_points * sizeof(float);

  fprintf(fp, "<?xml version=\"1.0\"?>\n"
              "<VTKFile"
              " type=\"ImageData\""
              " version=\"1.0\""
              " byte_order=\"LittleEndian\""
              " header_type=\"UInt64\""
              ">\n"
              "  <ImageData"
              " WholeExtent=\"0 %d 0 %d 0 %d\""
              " Spacing=\"%g %g %g\""
              " Origin=\"%g %g %g\""
              ">\n"
              "    <Piece Extent=\"0 %d 0 %d 0 %d\">\n"
              "      <PointData Scalars=\"scalar_data\">\n"
              "        <DataArray"
              " type=\"Float32\""
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
          data->dx, data->dy, 0.0f,
          0.0f, 0.0f, 0.0f,
          data->nx - 1, data->ny - 1, 0,
          data->name);

  fwrite(&num_bytes, sizeof(uint64_t), 1, fp);
  fwrite(data->values, sizeof(float), (size_t)num_points, fp);

  fprintf(fp, "  </AppendedData>\n"
              "</VTKFile>\n");

  fclose(fp);

  return 0;
}


/****************************************************************************************************************************/


/* Cettez version de write manifest vtk est exactement la même que pour les codes CPU */
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

  for(int n = 0; n < nt; n++) {
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



/****************************************************************************************************************************/
/***************************************************** KERNELS CUDA *********************************************************/
/****************************************************************************************************************************/


// update hx: global_i = 0, ... , nx-1 et global_j = 0, ... , ny-2  
__global__ void kernel_update_hx_hy(float *hx, float *hy, const float *ez, int nx, int ny, const float chy, const float chx)
{
  
  extern __shared__ float shared_ez[]; //La size sera déclarée au lancement du kernel en argument <<< >>>. Elle sera égale à blockDim.x * (blockDim.y + 1), car il nous faut ez[i,j+1] pour calculer hx[i, j + 1/2], donc
  // Il nous faut une ghost row "venant du haut"

  const int global_i = blockIdx.x * blockDim.x + threadIdx.x; // [0, nx-1]
  const int global_j = blockIdx.y * blockDim.y + threadIdx.y; // [0, ny-1]

  const int shared_i = threadIdx.x; // [0,15]
  const int shared_j = threadIdx.y; // [0,15]

  //On ne garde que les threads qui sont dans le range de Hx, donc i = 0, ..., nx-1 et j=0, ..., ny-2
  if(global_i >= nx || global_j >= ny) //Pour quand on alloue "trop de threads", car on les alloue par block. Les deux conditions sont avec un "=", regarder l'instruction suivante pour comprendre: "out of
    return;                            //range access", pour global_j == ny.


  /* Remplissage du tableau shared pour les threads à l'intérieur du domaine (de shared_j == 1 jusque shared_j == 16) */
  shared_ez[shared_j * (blockDim.x + 1) + shared_i] = ez[global_j * nx + global_i];  // Car le tableau shared_ez possède blockDim.x colonnes et ez, lui en possède nx. Cette instruction ne remplit que les 16 première lignes
                                                                               // du tableau shared_ez, il faut encore remplir la 17 ème ligne i.e la ghost row du haut


  /* Remplissage du tableau shared pour les threads tout au dessus du domaine */
  if( (threadIdx.y == (blockDim.y - 1)  ||  global_j == ny-2)  &&  global_j < ny-1) // "Si je suis un thread du dessus du domaine...". La deuxième condition n'est utile que pour les derniers blocs, ceux tout au dessus
    shared_ez[(shared_j + 1) * (blockDim.x + 1) + shared_i] = ez[(global_j + 1) * nx + global_i];

  if( (threadIdx.x == (blockDim.x - 1)  ||  global_i == nx-2)  &&  global_i < nx-1)  // "Si je suis un thread à droite du domaine..."
    shared_ez[shared_j * (blockDim.x + 1) + (shared_i + 1)] = ez[global_j * nx + (global_i + 1)];



    __syncthreads();


  // hx[i,j] = hx[i,j] - chy * (ez[i, j+1] - ez[i, j]);
  //float hx_ij = get(hx, nx, global_i, global_j) - chy * (get(shared_ez, blockDim.x, shared_i, shared_j + 1) - get(shared_ez, blockDim.x, shared_i, shared_j));
  if(global_j < ny-1)
  {
    float hx_ij = hx[global_j * nx + global_i]
              - chy * (
                shared_ez[(shared_j + 1) * (blockDim.x + 1) + shared_i] 
              - shared_ez[shared_j * (blockDim.x + 1) + shared_i]
              );

    //set(hx, nx, global_i, global_j, hx_ij);
    hx[global_j * nx + global_i] = hx_ij;
  }
  


  // hy[i,j] = hy[i,j] + chx * (ez[i+1, j] - ez[i, j]);
  //float hy_ij = get(hy, nx, i, j) + chx * (get(ez, nx, i+1, j) - get(ez, nx, i, j));
  if(global_i < nx-1)
  {
    float hy_ij = hy[global_j * (nx-1) + global_i] 
              + chx * (
                shared_ez[shared_j * (blockDim.x + 1) + (shared_i + 1)]
              - shared_ez[shared_j * (blockDim.x + 1) + shared_i]
              );

    //set(hy, nx, i, j, hy_ij);
    hy[global_j * (nx-1) + global_i] = hy_ij;
  }
}


/****************************************************************************************************************************/


// update ez: i = 1, ... , nx-2 et j = 1, ... , ny-2
__global__ void kernel_update_ez(float *ez, const float *hx, const float *hy, int nx, int ny, float cex, float cey)
{

  __shared__ float shared_hx[BLOCK_X * (BLOCK_Y + 1)]; // On n'a pas besoin de ghost row en haut car j = j + 1/2 numériquement !
  __shared__ float shared_hy[(BLOCK_X + 1) * BLOCK_Y]; // On n'a pas besoin de ghost row en haut car i = i + 1/2 numériquement !

  const int g_i = blockIdx.x * blockDim.x + threadIdx.x;
  const int g_j = blockIdx.y * blockDim.y + threadIdx.y;

  const int s_i = threadIdx.x;
  const int s_j = threadIdx.y;

  //Domaine de Ez
  if(g_i == 0  ||  g_j == 0  ||  g_i >= nx-1  ||  g_j >= ny-1) // Car on ne met pas à jour Ez sur les bords, il est posé == 0 par les conditions de bords de Dirichlet. "Si je suis un thread sur les bords de la grille..."
    return;
  

  /********** Intérieur du domaine **********/
  shared_hx[(s_j + 1) * blockDim.x + s_i] = hx[g_j * nx + g_i]; // On ne remplit pas la première ligne de shared_hx, elle sera pour la ghost row du bas
  shared_hy[s_j * (blockDim.x + 1) + (s_i + 1)] = hy[g_j * (nx-1) + g_i]; // On ne remplit pas la première colonne de shared_hy, elle sera pour la ghost column de gauche


  /********** Ghost rows bas pour Hx **********/
  if(threadIdx.y == 0) // bas
    shared_hx[0 * blockDim.x + s_i] = hx[(g_j - 1) * nx + g_i];


  /********** Ghost columns gauche pour Hy **********/
  if(threadIdx.x == 0) // gauche
    shared_hy[s_j * (blockDim.x + 1) + 0] = hy[g_j * (nx-1) + (g_i - 1)];
  

  __syncthreads();


  /********** Calcul de Ez **********/
  float ez_ij = ez[g_j * nx + g_i]
              + cex * (shared_hy[s_j * (blockDim.x + 1) + (s_i + 1)] - shared_hy[s_j * (blockDim.x + 1) + s_i])
              - cey * (shared_hx[(s_j + 1) * blockDim.x + s_i] - shared_hx[s_j * blockDim.x + s_i]);

  //set(ez, nx, i, j, ez_ij);
  ez[g_j * nx + g_i] = ez_ij;
}


/****************************************************************************************************************************/


// impose source: set ez[nx/2, ny/2] = sin(2*pi*2.4e9 * t)
__global__ void kernel_impose_source(float *ez, int nx, int ny, float value)
{
  // single thread OK
  if(threadIdx.x == 0 && blockIdx.x == 0) //Au cas où on lance le kernel avec + de un seul thread
  {
    const int g_i = nx/2;
    const int g_j = ny/2;
    ez[g_j* nx + g_i] = value;
  }
}



/****************************************************************************************************************************/
/****************************************************** MAIN HOST ***********************************************************/
/****************************************************************************************************************************/


int main(int argc, char **argv)
{

  float dx = 1., dy = 1., dt = 1.;
  int nx = 1, ny = 1, nt = 1, sampling_rate = 1;
  float eps = 8.854187817e-12f;
  float mu  = 1.2566370614359173e-06f;

  dx = dy = (3.e8 / 2.4e9) / 40.; // wavelength / 40
  nx = ny = 4096;
  dt = 0.5 / (3.e8 * sqrt(1. / (dx * dx) + 1. / (dy * dy))); // cfl / 2
  nt = 500;
  sampling_rate = 0; // don't save

//Constantes dont on a besoin (en float, comme demandé)
float c = (float)(1.0 / sqrt(eps * mu));
float chx = (float)(dt / (dx * mu));
float chy = (float)(dt / (dy * mu));
float cex = (float)(dt / (dx * eps));
float cey = (float)(dt / (dy * eps));


  printf("Solving problem :\n");
  printf(" - space %gm x %gm (dx=%g, dy=%g; nx=%d, ny=%d)\n",
         dx * nx, dy * ny, dx, dy, nx, ny);
  printf(" - time %gs (dt=%g, nt=%d)\n", dt * nt, dt, nt);



//Allocation des tableaux des 3 champs sur le host (CPU)
struct data_host host_hx, host_hy, host_ez;
  if(init_host_data(&host_hx, "host_hx", nx, ny - 1, dx, dy, 0.) ||
     init_host_data(&host_hy, "host_hy", nx - 1, ny, dx, dy, 0.) ||
     init_host_data(&host_ez, "host_ez", nx, ny, dx, dy, 0.)) {
    printf("Error: could not allocate data\n");
    return 1;
  }


  
//Alocation des tableaux des 3 champs sur le device (GPU)
float *d_hx, *d_hy, *d_ez;
size_t size_in_bytes_hx = (size_t)nx * (size_t)(ny-1) * sizeof(float);
size_t size_in_bytes_hy = (size_t)(nx-1) * (size_t)ny * sizeof(float);
size_t size_in_bytes_ez = (size_t)nx * (size_t)ny * sizeof(float);

CHECK_CUDA(cudaMalloc(&d_hx, size_in_bytes_hx));
CHECK_CUDA(cudaMalloc(&d_hy, size_in_bytes_hy));
CHECK_CUDA(cudaMalloc(&d_ez, size_in_bytes_ez));

//On initialise à 0 tous les tableaux des champs sur le device
CHECK_CUDA(cudaMemset(d_hx, 0, size_in_bytes_hx));
CHECK_CUDA(cudaMemset(d_hy, 0, size_in_bytes_hy));  // J'utilise cudaMemset(..., 0, ...) au lieu de cudaMemcpy(d_hy, host_hy.values, ...) car ce dernier est blocking (synchronous) par rapport au host. Je suppose donc
CHECK_CUDA(cudaMemset(d_ez, 0, size_in_bytes_ez));  // que cudaMemset est plus optimisé, comme on initialise sans copie ?

//On déclare les grids
dim3 block(BLOCK_X, BLOCK_Y);
dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

//Tailles des tableaux shared pour les kernels
size_t size_in_bytes_shared_ez = size_t(BLOCK_X + 1) * size_t(BLOCK_Y + 1) * sizeof(float);

//Pour mesurer le temps d'exécution
cudaEvent_t start, stop;
float elapsed; // Le temps est en millisecondes !
CHECK_CUDA(cudaEventCreate(&start));
CHECK_CUDA(cudaEventCreate(&stop));

CHECK_CUDA(cudaEventRecord(start)); //horodatage de l'event start



//time loop
for(int n = 0; n < nt; n++)
{
  //On update les champs les uns après les autres
  kernel_update_hx_hy<<<grid, block, size_in_bytes_shared_ez>>>(d_hx, d_hy, d_ez, nx, ny, chy, chx);
  kernel_update_ez<<<grid, block>>>(d_ez, d_hx, d_hy, nx, ny, cex, cey);

  //Impose source
  float t = n * dt;
  float source = sin(2.f * M_PI * 2.4e9f * t);

  kernel_impose_source<<<1,1>>>(d_ez, nx, ny, source); //Un seul thread est nécessaire pour écrire une valeur dans le tableau d_ez

  //Output VTK (copie vers l'host)
  if(sampling_rate && !(n % sampling_rate)) {

    CHECK_CUDA(cudaDeviceSynchronize()); // On s'assure que tous les kernels soient exécutés avant d'écrire dans le fichier

    //Copie GPU -> CPU 
    CHECK_CUDA(cudaMemcpy(host_ez.values, d_ez, size_in_bytes_ez, cudaMemcpyDeviceToHost));

    write_data_vtk_float(&host_ez, n, 0);
  }
}//fin loop(n)

CHECK_CUDA(cudaDeviceSynchronize()); // On s'assure que le device a terminer toutes ses précédentes tasks avant de continuer

CHECK_CUDA(cudaEventRecord(stop)); // horodotage de l'event stop
CHECK_CUDA(cudaEventSynchronize(stop));
CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));

float time_sec = elapsed * 1e-3;
double mupdates = 1.e-6 * (double)nx * (double)ny * (double)nt / time_sec;
printf("\nGPU Done: %g seconds (%g MUpdates/s)\n", time_sec, mupdates);

// Si on veut write le manifest
write_manifest_vtk("ez", dt, nt, sampling_rate, 1);

//On n'oublie pas de free toutes les data
free(host_hx.values);
free(host_hy.values);
free(host_ez.values);

CHECK_CUDA(cudaFree(d_hx));
CHECK_CUDA(cudaFree(d_hy));
CHECK_CUDA(cudaFree(d_ez));

CHECK_CUDA(cudaEventDestroy(start));
CHECK_CUDA(cudaEventDestroy(stop));

return 0;

}








