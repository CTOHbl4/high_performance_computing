#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <math.h>
#include <errno.h>
#include <mpi.h>

__constant__ float c_Lx, c_Ly, c_Lz;
__constant__ float c_h, c_inv_h2;
__constant__ float c_tau, c_tau2, c_a2, c_Laplacian;
__constant__ int c_N, c_M, c_K;
__constant__ int c_NLow, c_MLow, c_KLow;
__constant__ int c_NGlobal, c_MGlobal, c_KGlobal;
__constant__ int c_MLow0, c_JRight, c_MK;

#define PI 3.14159265359
#define DIMENTIONS 3

__device__ float analyticalSolution_device(float x, float y, float z, float t) {
    float at = PI/3.0f * sqrtf(4.0f/(c_Lx*c_Lx) + 1.0f/(c_Ly*c_Ly) + 4.0f/(c_Lz*c_Lz));
    return sinf(2.0f*PI*x/c_Lx) * sinf((1.0f + y/c_Ly) * PI) * 
           sinf((1.0f + z/c_Lz) * 2.0f*PI) * cosf(at*t + PI);
}

__device__ float phi_device(float x, float y, float z) {
    return analyticalSolution_device(x, y, z, 0.0f);
}

__global__ void initialize_u0_kernel(float* res) {

    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= c_N || j >= c_M || k >= c_K) return;
    
    int idx = i *c_MK + j * c_K + k;

    if ((c_MLow0 && j == 0) || (c_JRight && j == c_M - 1)) {
        res[idx] = 0.0f;
        return;
    }

    float x = (i + c_NLow) * c_h;
    float y = (j + c_MLow) * c_h;
    float z = (k + c_KLow) * c_h;

    res[idx] = phi_device(x, y, z);
}

__global__ void initialize_u1_kernel(float* res, float* ut,
                                     float* xPrev, float* xNext, float* yPrev, float* yNext,
                                     float* zPrev, float* zNext) {

    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= c_N || j >= c_M || k >= c_K) return;
    int idx = i * c_MK + j * c_K + k;
    if ((c_MLow0 && j == 0) || (c_JRight && j == c_M - 1)) {
        res[idx] = 0.0f;
        return;
    }
    float center = ut[idx];

    float left = (i > 0) ? ut[idx - c_MK] : xPrev[j*c_K + k];
    float right = (i < c_N-1) ? ut[idx + c_MK] : xNext[j*c_K + k];
    float front = (j > 0) ? ut[idx - c_K] : yPrev[i*c_K + k];
    float back = (j < c_M-1) ? ut[idx + c_K] : yNext[i*c_K + k];
    float bottom = (k > 0) ? ut[idx - 1] : zPrev[i*c_M + j];
    float top = (k < c_K-1) ? ut[idx + 1] : zNext[i*c_M + j];

    float laplacian = (left + right + front + back + bottom + top - 6.0f * center);

    float l = c_Laplacian * laplacian / 2.0f;
    res[idx] = center + l;
}

void makeU0_device(float* d_ut0, dim3 blockSize, dim3 gridSize) {
    initialize_u0_kernel<<<gridSize, blockSize>>>(d_ut0);
    cudaDeviceSynchronize();
}

void makeU1_device(float* d_ut1, float* d_ut0,
                     dim3 blockSize, dim3 gridSize,
                     float* d_xPrev, float* d_xNext, float* d_yPrev, float* d_yNext,
                     float* d_zPrev, float* d_zNext) {
    initialize_u1_kernel<<<gridSize, blockSize>>>(d_ut1, d_ut0,
                                                     d_xPrev, d_xNext, d_yPrev, d_yNext,
                                                     d_zPrev, d_zNext);
    cudaDeviceSynchronize();
}

__global__ void step_kernel(float* ut1, float* ut0,
                            float* xPrev, float* xNext, 
                            float* yPrev, float* yNext,
                            float* zPrev, float* zNext) {

    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if ((i >= c_N) || (j >= c_M) || (k >= c_K) || (c_MLow0 && j == 0) || (c_JRight && j == c_M - 1)) return;
    
    int idx = i * c_MK + j * c_K + k;

    float centerx2 = 2.0f * ut1[idx];

    float left = (i > 0) ? ut1[idx - c_MK] : xPrev[j*c_K + k];
    float right = (i < c_N-1) ? ut1[idx + c_MK] : xNext[j*c_K + k];
    float front = (j > 0) ? ut1[idx - c_K] : yPrev[i*c_K + k];
    float back = (j < c_M-1) ? ut1[idx + c_K] : yNext[i*c_K + k];
    float bottom = (k > 0) ? ut1[idx - 1] : zPrev[i*c_M + j];
    float top = (k < c_K-1) ? ut1[idx + 1] : zNext[i*c_M + j];

    float laplacian = (left + right + front + back + bottom + top - 3.0f * centerx2);

    float l = c_Laplacian * laplacian;

    float dt2 = centerx2 - ut0[idx];
    ut0[idx] = dt2 + l;
}

void makeStep_device(float* d_ut1, float* d_ut0,
                     dim3 blockSize, dim3 gridSize,
                     float* d_xPrev, float* d_xNext, float* d_yPrev, float* d_yNext,
                     float* d_zPrev, float* d_zNext) {
    step_kernel<<<gridSize, blockSize>>>(d_ut1, d_ut0,
                                            d_xPrev, d_xNext, d_yPrev, d_yNext,
                                            d_zPrev, d_zNext);
}

__global__ void pack_z_boundary_kernel(float* local_data, float* z_send, int exchange_idx) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= c_N || j >= c_M) return;
    
    int dst_idx = i * c_M + j;
    int src_idx = i * c_MK + j * c_K + exchange_idx;
    
    z_send[dst_idx] = local_data[src_idx];
}

__global__ void pack_y_boundary_kernel(float* local_data, float* y_send, int j_offset) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= c_N || k >= c_K) return;
    
    int src_idx = (i * c_M + j_offset) * c_K + k;
    int dst_idx = i * c_K + k;
    
    y_send[dst_idx] = local_data[src_idx];
}

void exchange_cuda(float* d_local_data, int N, int M, int K, MPI_Comm cart_comm,
                   float* h_xPrev, float* h_xNext, float* h_yPrev, float* h_yNext,
                   float* h_zPrev, float* h_zNext,
                   float* h_xPrevSend, float* h_xNextSend, float* h_yPrevSend, float* h_yNextSend,
                   float* h_zPrevSend, float* h_zNextSend,
                   float* d_xPrev, float* d_xNext, float* d_yPrev, float* d_yNext,
                   float* d_zPrev, float* d_zNext,
                   int left_rank, int right_rank, int front_rank, int back_rank,
                   int down_rank, int up_rank, int NLow, int NGlobal, int MLow, 
                   int MGlobal, int KLow, int KGlobal) {
    
    MPI_Request requests[12];
    int count = 0;
    const int MK = M*K;
    const int NM = N*M;
    const int NK = N*K;

    // x
    int exchangeIdxXPrev = 0;
    int exchangeIdxXNext = (N - 1) * MK;
    if (NLow == 0) {
        exchangeIdxXPrev = MK;
    }
    if (NLow + N >= NGlobal) {
        exchangeIdxXNext = (N - 2) * MK;
    }

    cudaMemcpyAsync(h_xPrevSend, d_local_data + exchangeIdxXPrev, MK * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_xNextSend, d_local_data + exchangeIdxXNext, MK * sizeof(float), cudaMemcpyDeviceToHost);

    MPI_Isend(h_xPrevSend, MK, MPI_FLOAT, left_rank, 0, cart_comm, &requests[count++]);
    MPI_Irecv(h_xPrev, MK, MPI_FLOAT, left_rank, 1, cart_comm, &requests[count++]);

    MPI_Isend(h_xNextSend, MK, MPI_FLOAT, right_rank, 1, cart_comm, &requests[count++]);
    MPI_Irecv(h_xNext, MK, MPI_FLOAT, right_rank, 0, cart_comm, &requests[count++]);
    
    // z
    int exchangeIdxZPrev = 0;
    int exchangeIdxZNext = K - 1;
    if (KLow == 0) {
        exchangeIdxZPrev = 1;
    }
    if (KLow + K >= KGlobal) {
        exchangeIdxZNext = K - 2;
    }
    dim3 blockSize_z(8, 8);
    dim3 gridSize_z((M + blockSize_z.x - 1) / blockSize_z.x,
                    (N + blockSize_z.y - 1) / blockSize_z.y);

    pack_z_boundary_kernel<<<gridSize_z, blockSize_z>>>(d_local_data, d_zPrev, exchangeIdxZPrev);
    pack_z_boundary_kernel<<<gridSize_z, blockSize_z>>>(d_local_data, d_zNext, exchangeIdxZNext);

    cudaMemcpyAsync(h_zPrevSend, d_zPrev, NM * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_zNextSend, d_zNext, NM * sizeof(float), cudaMemcpyDeviceToHost);

    MPI_Isend(h_zPrevSend, NM, MPI_FLOAT, down_rank, 4, cart_comm, &requests[count++]);
    MPI_Irecv(h_zPrev, NM, MPI_FLOAT, down_rank, 5, cart_comm, &requests[count++]);
    
    MPI_Isend(h_zNextSend, NM, MPI_FLOAT, up_rank, 5, cart_comm, &requests[count++]);
    MPI_Irecv(h_zNext, NM, MPI_FLOAT, up_rank, 4, cart_comm, &requests[count++]);

    dim3 blockSize_y(8, 8);
    dim3 gridSize_y((K + blockSize_y.x - 1) / blockSize_y.x,
                    (N + blockSize_y.y - 1) / blockSize_y.y);
    if (front_rank != MPI_PROC_NULL) {

        pack_y_boundary_kernel<<<gridSize_y, blockSize_y>>>(d_local_data, d_yPrev, 0);

        cudaMemcpy(h_yPrevSend, d_yPrev, NK * sizeof(float), cudaMemcpyDeviceToHost);

        MPI_Isend(h_yPrevSend, NK, MPI_FLOAT, front_rank, 2, cart_comm, &requests[count++]);
        MPI_Irecv(h_yPrev, NK, MPI_FLOAT, front_rank, 3, cart_comm, &requests[count++]);
    }

    if (back_rank != MPI_PROC_NULL) {
        pack_y_boundary_kernel<<<gridSize_y, blockSize_y>>>(d_local_data, d_yNext, M - 1);

        cudaMemcpy(h_yNextSend, d_yNext, NK * sizeof(float), cudaMemcpyDeviceToHost);

        MPI_Isend(h_yNextSend, NK, MPI_FLOAT, back_rank, 3, cart_comm, &requests[count++]);
        MPI_Irecv(h_yNext, NK, MPI_FLOAT, back_rank, 2, cart_comm, &requests[count++]);
    }

    MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);

    cudaMemcpyAsync(d_xPrev, h_xPrev, MK * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_xNext, h_xNext, MK * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMemcpyAsync(d_zPrev, h_zPrev, NM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_zNext, h_zNext, NM * sizeof(float), cudaMemcpyHostToDevice);

    if (front_rank != MPI_PROC_NULL) {
        cudaMemcpyAsync(d_yPrev, h_yPrev, NK * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    if (back_rank != MPI_PROC_NULL) {
        cudaMemcpyAsync(d_yNext, h_yNext, NK * sizeof(float), cudaMemcpyHostToDevice);
    }
}

__global__ void compute_errors_kernel(float* d_u, float* d_errors, float t) {
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= c_N || j >= c_M || k >= c_K) return;

    int idx = i * c_MK + j * c_K + k;

    float x = (i + c_NLow) * c_h;
    float y = (j + c_MLow) * c_h;
    float z = (k + c_KLow) * c_h;
    float analytical = analyticalSolution_device(x, y, z, t);

    float numerical = d_u[idx];

    d_errors[idx] = fabsf(numerical - analytical);
}

void compute_errors_device(float* d_u, float* d_errors,
                           dim3 blockSize, dim3 gridSize, float t) {
    compute_errors_kernel<<<gridSize, blockSize>>>(d_u, d_errors, t);
}

__global__ void pairwise_max_kernel(float* array, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= stride) return;
    
    float val1 = array[idx];
    float val2 = array[stride + idx];
    array[idx] = fmaxf(val1, val2);
}

float find_max_device(float* d_array, int size) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
        int threads = min(1024, stride);
        int blocks = (stride + threads - 1) / threads;
        
        pairwise_max_kernel<<<blocks, threads>>>(d_array, stride);
        cudaDeviceSynchronize();
    }
    
    float max_value;
    cudaMemcpy(&max_value, d_array, sizeof(float), cudaMemcpyDeviceToHost);
    return max_value;
}

void saveFloatsBinary(const char* filename, float* array, int size) {
    FILE* file = fopen(filename, "wb");
    if (file == NULL) {
        printf("Error opening file %s!\n", filename);
        return;
    }
    size_t written = fwrite(array, sizeof(float), size, file);
    fclose(file);
    if (written != size) {
        printf("Error: only wrote %zu/%d elements to %s\n", written, size, filename);
    }
}

void freeBuffersAndTerminate(float** hostBuffers, int numHostBuffers, float** deviceBuffers, int numDeviceBuffers,
                             cudaEvent_t* events, int numEvents, MPI_Comm cart_comm) {
    for (int i = 0; i < numHostBuffers; i++)
        if (hostBuffers[i] != NULL)
            cudaFreeHost(hostBuffers[i]);
    for (int i = 0; i < numDeviceBuffers; i++)
        if (deviceBuffers[i] != NULL)
            cudaFree(deviceBuffers[i]);
    for (int i = 0; i < numEvents; i++)
        cudaEventDestroy(events[i]);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
}

void setup_constants(int N, int M, int K, int NLow, int MLow, int KLow,
                     int NGlobal, int MGlobal, int KGlobal,
                     float Lx, float Ly, float Lz, 
                     float h, float tau, float a2) {

    float inv_h2 = 1.0f / (h * h);
    float tau2 = tau * tau;
    float laplacian = a2 * inv_h2 * tau2;

    int JRight = (M + MLow >= MGlobal), MLow0 = (MLow == 0), MK = M*K;

    cudaMemcpyToSymbol(c_JRight, &JRight, sizeof(int));
    cudaMemcpyToSymbol(c_MLow0, &MLow0, sizeof(int));
    cudaMemcpyToSymbol(c_MK, &MK, sizeof(int));

    cudaMemcpyToSymbol(c_Lx, &Lx, sizeof(float));
    cudaMemcpyToSymbol(c_Ly, &Ly, sizeof(float));
    cudaMemcpyToSymbol(c_Lz, &Lz, sizeof(float));
    
    cudaMemcpyToSymbol(c_Laplacian, &laplacian, sizeof(float));
    cudaMemcpyToSymbol(c_h, &h, sizeof(float));
    cudaMemcpyToSymbol(c_inv_h2, &inv_h2, sizeof(float));
    
    cudaMemcpyToSymbol(c_tau, &tau, sizeof(float));
    cudaMemcpyToSymbol(c_tau2, &tau2, sizeof(float));
    cudaMemcpyToSymbol(c_a2, &a2, sizeof(float));
    
    cudaMemcpyToSymbol(c_N, &N, sizeof(int));
    cudaMemcpyToSymbol(c_M, &M, sizeof(int));
    cudaMemcpyToSymbol(c_K, &K, sizeof(int));
    
    cudaMemcpyToSymbol(c_NLow, &NLow, sizeof(int));
    cudaMemcpyToSymbol(c_MLow, &MLow, sizeof(int));
    cudaMemcpyToSymbol(c_KLow, &KLow, sizeof(int));
    
    cudaMemcpyToSymbol(c_NGlobal, &NGlobal, sizeof(int));
    cudaMemcpyToSymbol(c_MGlobal, &MGlobal, sizeof(int));
    cudaMemcpyToSymbol(c_KGlobal, &KGlobal, sizeof(int));
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Usage: %s NGlobal T L\n", argv[0]);
        return 1;
    }

    MPI_Init(&argc, &argv);
    float time_init;
    cudaEvent_t start, startFull, startLoop, startExchange, stop, stopFull, stopExchange, stopLoop;
    cudaEventCreate(&start);
    cudaEventCreate(&startFull);
    cudaEventCreate(&startLoop);
    cudaEventCreate(&startExchange);
    cudaEventCreate(&stop);
    cudaEventCreate(&stopFull);
    cudaEventCreate(&stopLoop);
    cudaEventCreate(&stopExchange);
    cudaEvent_t events[8] = {start, startFull, startLoop, startExchange, stop, stopFull, stopExchange, stopLoop};
    cudaEventRecord(startFull, 0);
    cudaEventRecord(start, 0);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int save = argc == 5 && argv[4][0] == 'a';
    if (rank == 0) {
        if (save) {
            struct stat st = {0};
            if (stat("floats", &st) == -1) {
                mkdir("floats", 0700);
            }
            if (stat("errors", &st) == -1) {
                mkdir("errors", 0700);
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    int NGlobal = atoi(argv[1]);
    float T = strtof(argv[2], NULL), L = strtof(argv[3], NULL);
    float h = L/(NGlobal-1), a2 = 1.0/9.0, tau = h / (sqrt(a2 * 12));
    int filenameLength = 100;
    char filename[filenameLength];

    int dims[DIMENTIONS] = {0, 0, 0};
    MPI_Dims_create(size, DIMENTIONS, dims);
    
    int periods[DIMENTIONS] = {1, 0, 1};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, DIMENTIONS, dims, periods, 1, &cart_comm);
    
    int coords[DIMENTIONS];
    MPI_Cart_coords(cart_comm, rank, DIMENTIONS, coords);
    
    int left_rank, right_rank, front_rank, back_rank, down_rank, up_rank;
    MPI_Cart_shift(cart_comm, 0, 1, &left_rank, &right_rank);
    MPI_Cart_shift(cart_comm, 1, 1, &front_rank, &back_rank);
    MPI_Cart_shift(cart_comm, 2, 1, &down_rank, &up_rank);

    int N = NGlobal / dims[0];
    int M = NGlobal / dims[1];
    int K = NGlobal / dims[2];

    int NLow = coords[0] * N;
    int MLow = coords[1] * M;
    int KLow = coords[2] * K;
    
    if (coords[0] == dims[0]-1) N = NGlobal - NLow;
    if (coords[1] == dims[1]-1) M = NGlobal - MLow; 
    if (coords[2] == dims[2]-1) K = NGlobal - KLow;

    setup_constants(N, M, K, NLow, MLow, KLow,
                     NGlobal, NGlobal, NGlobal,
                     L, L, L, h, tau, a2);

    int uShape = N*M*K;

    float *h_ut = NULL, *h_errors = NULL;
    float *h_xPrev = NULL, *h_xPrevSend = NULL, *h_xNext = NULL, *h_xNextSend = NULL;
    float *h_yPrev = NULL, *h_yPrevSend = NULL, *h_yNext = NULL, *h_yNextSend = NULL;
    float *h_zPrev = NULL, *h_zPrevSend = NULL, *h_zNext = NULL, *h_zNextSend = NULL;

    cudaMallocHost(&h_ut, uShape * sizeof(float)); cudaMallocHost(&h_errors, uShape * sizeof(float));
    cudaMallocHost(&h_xPrev, M*K * sizeof(float)); cudaMallocHost(&h_xPrevSend, M*K * sizeof(float)); cudaMallocHost(&h_xNext, M*K * sizeof(float)); cudaMallocHost(&h_xNextSend, M*K * sizeof(float));
    
    cudaMallocHost(&h_yPrev, N*K * sizeof(float)); cudaMallocHost(&h_yPrevSend, N*K * sizeof(float)); cudaMallocHost(&h_yNext, N*K * sizeof(float)); cudaMallocHost(&h_yNextSend, N*K * sizeof(float));
    
    cudaMallocHost(&h_zPrev, N*M * sizeof(float)); cudaMallocHost(&h_zPrevSend, N*M * sizeof(float)); cudaMallocHost(&h_zNext, N*M * sizeof(float)); cudaMallocHost(&h_zNextSend, N*M * sizeof(float));

    float *d_errors = NULL, *d_ut0 = NULL, *d_ut1 = NULL, *d_xPrev = NULL, *d_xNext = NULL, *d_yPrev = NULL, *d_yNext = NULL, *d_zPrev = NULL, *d_zNext = NULL;
    cudaMalloc(&d_ut0, uShape * sizeof(float)); cudaMalloc(&d_ut1, uShape * sizeof(float)); cudaMalloc(&d_errors, uShape * sizeof(float));
    cudaMalloc(&d_xPrev, M*K * sizeof(float)); cudaMalloc(&d_xNext, M*K * sizeof(float)); cudaMalloc(&d_yPrev, N*K * sizeof(float)); cudaMalloc(&d_yNext, N*K * sizeof(float));
    cudaMalloc(&d_zPrev, N*M * sizeof(float)); cudaMalloc(&d_zNext, N*M * sizeof(float));

    float* hostBuffers[14] = {
        h_ut, h_errors, h_xPrev, h_xPrevSend, h_xNext, h_xNextSend, 
        h_yPrev, h_yPrevSend, h_yNext, h_yNextSend, 
        h_zPrev, h_zPrevSend, h_zNext, h_zNextSend
    };
    
    float* deviceBuffers[9] = {
        d_errors, d_ut0, d_ut1, d_xPrev, d_xNext, d_yPrev, d_yNext, d_zPrev, d_zNext
    };

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    cudaSetDevice(rank % 2);

    int success = 1;
    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        success = 0;
    }
    int globalSuccess;
    MPI_Allreduce(&success, &globalSuccess, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if (!globalSuccess) {
        if (rank == 0) {
            printf(cudaGetErrorString(cuda_err));
        }
        freeBuffersAndTerminate(hostBuffers, 14, deviceBuffers, 9, events, 8, cart_comm);
        return -1;
    }

    if (rank == 0) {
        printf("Grid: %dx%dx%d, Processes: %dx%dx%d\n", NGlobal, NGlobal, NGlobal, dims[0], dims[1], dims[2]);
    }

    dim3 blockSize(64, 4, 2);
    dim3 gridSize(
        (K + blockSize.x - 1) / blockSize.x,
        (M + blockSize.y - 1) / blockSize.y,
        (N + blockSize.z - 1) / blockSize.z
    );
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_init, start, stop);
    time_init /= 1000.0;

    float timeU0;
    cudaEventRecord(start, 0);

    makeU0_device(d_ut0, blockSize, gridSize);
    success = 1;
    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        success = 0;
    }
    MPI_Allreduce(&success, &globalSuccess, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if (!globalSuccess) {
        if (rank == 0) {
            printf(cudaGetErrorString(cuda_err));
        }
        freeBuffersAndTerminate(hostBuffers, 14, deviceBuffers, 9, events, 8, cart_comm);
        return -1;
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeU0, start, stop);
    timeU0 /= 1000.0;

    float timeExchanges;
    float timeExchange;
    cudaEventRecord(startExchange, 0);
    
    exchange_cuda(d_ut0, N, M, K, cart_comm,
                  h_xPrev, h_xNext, h_yPrev, h_yNext, h_zPrev, h_zNext,
                  h_xPrevSend, h_xNextSend, h_yPrevSend, h_yNextSend, h_zPrevSend, h_zNextSend,
                  d_xPrev, d_xNext, d_yPrev, d_yNext, d_zPrev, d_zNext,
                  left_rank, right_rank, front_rank, back_rank,
                  down_rank, up_rank, NLow, NGlobal, MLow, NGlobal, KLow, NGlobal);

    cudaEventRecord(stopExchange, 0);
    cudaEventSynchronize(stopExchange);
    cudaEventElapsedTime(&timeExchanges, startExchange, stopExchange);

    float timeU1;
    cudaEventRecord(start, 0);
    makeU1_device(d_ut1, d_ut0, blockSize, gridSize,
                          d_xPrev, d_xNext, d_yPrev, d_yNext, d_zPrev, d_zNext);

    success = 1;
    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        success = 0;
    }
    MPI_Allreduce(&success, &globalSuccess, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if (!globalSuccess) {
        if (rank == 0) {
            printf(cudaGetErrorString(cuda_err));
        }
        freeBuffersAndTerminate(hostBuffers, 14, deviceBuffers, 9, events, 8, cart_comm);
        return -1;
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeU1, start, stop);
    timeU1 /= 1000.0;

    float timeLoop;
    cudaEventRecord(startLoop, 0);
    int t = 2;
    
    float nextTime = tau * t;
    float* temp;
    float timeSteps = 0;
    float timeStep;

    for (; nextTime < T; ) {
        cudaEventRecord(startExchange, 0);
        exchange_cuda(d_ut1, N, M, K, cart_comm,
                      h_xPrev, h_xNext, h_yPrev, h_yNext, h_zPrev, h_zNext,
                      h_xPrevSend, h_xNextSend, h_yPrevSend, h_yNextSend, h_zPrevSend, h_zNextSend,
                      d_xPrev, d_xNext, d_yPrev, d_yNext, d_zPrev, d_zNext,
                      left_rank, right_rank, front_rank, back_rank,
                      down_rank, up_rank, NLow, NGlobal, MLow, NGlobal, KLow, NGlobal);
        cudaEventRecord(stopExchange, 0);
        cudaEventSynchronize(stopExchange);
        cudaEventElapsedTime(&timeExchange, startExchange, stopExchange);
        timeExchanges += timeExchange;
        
        cudaEventRecord(start, 0);

        makeStep_device(d_ut1, d_ut0, blockSize, gridSize, d_xPrev, d_xNext, d_yPrev, d_yNext, d_zPrev, d_zNext);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&timeStep, start, stop);
        timeSteps += timeStep;
        
        if (t % 10 == 0) {
            compute_errors_device(d_ut0, d_errors, blockSize, gridSize, nextTime);
            if (save) {
                cudaMemcpy(h_errors, d_errors, uShape * sizeof(float), cudaMemcpyDeviceToHost);
            }
            float maxError = find_max_device(d_errors, uShape);
            float maxErrorGlobal;
            MPI_Reduce(&maxError, &maxErrorGlobal, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0)
                printf("Analytical error at time %f: %.8f\n", nextTime, maxErrorGlobal);
            if (save) {
                cudaMemcpy(h_ut, d_ut0, uShape * sizeof(float), cudaMemcpyDeviceToHost);
                snprintf(filename, filenameLength, "floats/floats_t%dNLow%dMLow%dKLow%d.bin", t, NLow, MLow, KLow);
                saveFloatsBinary(filename, h_ut, uShape);
                snprintf(filename, filenameLength, "errors/errors_t%dNLow%dMLow%dKLow%d.bin", t, NLow, MLow, KLow);
                saveFloatsBinary(filename, h_errors, uShape);
            }
        }
        temp = d_ut1;
        d_ut1 = d_ut0;
        d_ut0 = temp;

        nextTime = tau * ++t;
    }
    cuda_err = cudaGetLastError();
    success = (cuda_err == cudaSuccess);
    MPI_Allreduce(&success, &globalSuccess, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if (!globalSuccess) {
        if (rank == 0) {
            printf(cudaGetErrorString(cuda_err));
        }
        freeBuffersAndTerminate(hostBuffers, 14, deviceBuffers, 9, events, 8, cart_comm);
        return -1;
    }
    cudaEventRecord(stopLoop, 0);
    cudaEventSynchronize(stopLoop);
    cudaEventElapsedTime(&timeLoop, startLoop, stopLoop);
    timeLoop /= 1000.0;

    float timeFull;
    cudaEventRecord(stopFull, 0);
    cudaEventSynchronize(stopFull);
    cudaEventElapsedTime(&timeFull, startFull, stopFull);
    timeFull /= 1000.0;

    if (rank == 0) {
        printf("Simulation completed: %d time steps, Time: init-%f, U0-%f, exchanges-%f, U1-%f, steps-%f, Loop-%f, Full-%f\n", t, time_init, timeU0, timeExchanges / 1000.0f, timeU1, timeSteps / 1000.0f, timeLoop, timeFull);
    }

    freeBuffersAndTerminate(hostBuffers, 14, deviceBuffers, 9, events, 8, cart_comm);
    return 0;
}
