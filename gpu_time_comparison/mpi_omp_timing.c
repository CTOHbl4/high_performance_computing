#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <math.h>
#include <errno.h>
#include <mpi.h>
#include <omp.h>

#define PI 3.14159265359
#define DIMENTIONS 3

float analyticalSolution(float Lx, float Ly, float Lz, float x, float y, float z, float t) {
    float at = PI/3.0f * sqrtf(4.0f/(Lx*Lx) + 1.0f/(Ly*Ly) + 4.0f/(Lz*Lz));
    return sinf(2.0f*PI*x/Lx) * sinf((1.0f + y/Ly) * PI) * 
           sinf((1.0f + z/Lz) * 2.0f*PI) * cosf(at*t + PI);
}

float phi(float Lx, float Ly, float Lz, float x, float y, float z) {
    return analyticalSolution(Lx, Ly, Lz, x, y, z, 0.0f);
}

float aaLaplasian(float* ut, int N, int M, int K, int i, int j, int k, float h, float a2, 
                  float* xPrev, float* xNext, float* yPrev, float* yNext,
                  float* zPrev, float* zNext, int MLow, int MGlobal) {
    int MK = M*K;
    int jK = j*K;
    int imk = MK * i;
    int idx = imk + jK + k;
    float centerx6 = 6.0f * ut[idx];

    float left = (i > 0) ? ut[idx - MK] : xPrev[jK + k];
    float right = (i < N-1) ? ut[idx + MK] : xNext[jK + k];
    float front = (j > 0) ? ut[idx - K] : yPrev[i*K + k];
    float back = (j < M-1) ? ut[idx + K] : yNext[i*K + k];
    float bottom = (k > 0) ? ut[idx - 1] : zPrev[i*M + j];
    float top = (k < K-1) ? ut[idx + 1] : zNext[i*M + j];
    
    float d2x = left + right;
    float d2y = front + back;
    float d2z = bottom + top;
    
    return a2 * (d2x + d2y + d2z - centerx6) / (h*h);
}

float* makeU0(int NLow, int MLow, int KLow, int N, int M, int K, int NGlobal, int MGlobal, int KGlobal, float Lx, float Ly, float Lz, float h) {
    float* res = calloc(N*M*K, sizeof(float));
    if (res == NULL) {
        return NULL;
    }
    float x, y, z;
    int idxi, idxj;

    #pragma omp parallel for private(idxi, idxj, x, y, z)
    for (int i = 0; i < N; i++) {
        idxi = i*M*K;
        x = (i + NLow) * h;
        for (int j = 0; j < M; j++) {
            y = (j + MLow) * h;
            idxj = idxi + j*K;
            for (int k = 0; k < K; k++) {
                z = (k + KLow) * h;
                if ((MLow == 0 && j == 0) || (MLow + M >= MGlobal && j == M-1)) {
                    res[idxj+k] = 0.0;
                } else {
                    res[idxj+k] = phi(Lx, Ly, Lz, x, y, z);
                }
            }
        }
    }
    return res;
}

float* makeU1(float* ut0, int NLow, int MLow, int KLow, int N, int M, int K, int NGlobal, int MGlobal, int KGlobal, 
              float Lx, float Ly, float Lz, float h, float tau, float a2,
              float* xPrev, float* xNext, float* yPrev, float* yNext,
              float* zPrev, float* zNext) {
    float* res = calloc(N*M*K, sizeof(float));
    if (res == NULL) {
        return NULL;
    }
    int idxi, idxj, jk;
    float l;

    #pragma omp parallel for private(idxi, idxj, jk, l)
    for (int i = 0; i < N; i++) {
        idxi = i*M*K;
        for (int j = 0; j < M; j++) {
            if ((MLow == 0 && j == 0) || (MLow + M >= MGlobal && j == M-1)) continue;
            jk = j*K;
            idxj = idxi + jk;
            for (int k = 0; k < K; k++) {
                l = tau * tau * aaLaplasian(ut0, N, M, K, i, j, k, h, a2, 
                                          xPrev, xNext, yPrev, yNext,
                                          zPrev, zNext, MLow, MGlobal) / 2;
                res[idxj+k] = ut0[idxj + k] + l;
            }
        }
    }
    return res;
}

void step(float** ut1_ptr, float** ut0_ptr, int NLow, int MLow, int KLow, int N, int M, int K, int NGlobal, int MGlobal, int KGlobal, 
          float h, float tau, float a2, 
          float* xPrev, float* xNext, float* yPrev, float* yNext,
          float* zPrev, float* zNext) {
    float* ut1 = *ut1_ptr;
    float* ut0 = *ut0_ptr;
    int idxi, idxj, jk;
    float l, dt2;

    #pragma omp parallel for private(idxi, idxj, jk, l, dt2)
    for (int i = 0; i < N; i++) {
        idxi = i*M*K;
        for (int j = 0; j < M; j++) {
            if ((MLow == 0 && j == 0) || (MLow + M >= MGlobal && j == M-1)) continue;
            jk = j*K;
            idxj = idxi + jk;
            for (int k = 0; k < K; k++) {
                l = tau * tau * aaLaplasian(ut1, N, M, K, i, j, k, h, a2,
                                          xPrev, xNext, yPrev, yNext,
                                          zPrev, zNext, MLow, MGlobal);
                dt2 = 2*ut1[idxj+k] - ut0[idxj+k];
                ut0[idxj+k] = dt2 + l;
            }
        }
    }
    float* swap = *ut1_ptr;
    *ut1_ptr = *ut0_ptr;
    *ut0_ptr = swap;
}

void exchange(float* local_data, int N, int M, int K, MPI_Comm cart_comm, 
                        float* xPrev, float* xNext, float* yPrev, float* yNext,
                        float* zPrev, float* zNext, float* yPrevSend, float* yNextSend,
                        float* zPrevSend, float* zNextSend,
                        int left_rank, int right_rank, int front_rank, int back_rank, 
                        int down_rank, int up_rank, int NLow, int NGlobal, int MLow, int MGlobal, int KLow, int KGlobal) {
    MPI_Request requests[12];
    int count = 0;
    // x
    int exchangeIdxXPrev = 0;
    int exchangeIdxXNext = (N-1)*M*K;
    if (NLow == 0) {
        exchangeIdxXPrev = M*K;
    }
    if (NLow + N >= NGlobal) {
        exchangeIdxXNext = (N-2)*M*K;
    }
    MPI_Isend(&local_data[exchangeIdxXPrev], M*K, MPI_FLOAT, left_rank, 0, cart_comm, &requests[count++]);
    MPI_Irecv(xPrev, M*K, MPI_FLOAT, left_rank, 1, cart_comm, &requests[count++]);
    MPI_Isend(&local_data[exchangeIdxXNext], M*K, MPI_FLOAT, right_rank, 1, cart_comm, &requests[count++]);
    MPI_Irecv(xNext, M*K, MPI_FLOAT, right_rank, 0, cart_comm, &requests[count++]);

    // z
    int exchangeIdxZPrev = 0;
    int exchangeIdxZNext = K-1;
    if (KLow == 0) {
        exchangeIdxZPrev = 1;
    }
    if (KLow + K >= KGlobal) {
        exchangeIdxZNext = K-2;
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            zPrevSend[i*M + j] = local_data[i*M*K + j*K + exchangeIdxZPrev];
        }
    }
    MPI_Isend(zPrevSend, N*M, MPI_FLOAT, down_rank, 4, cart_comm, &requests[count++]);
    MPI_Irecv(zPrev, N*M, MPI_FLOAT, down_rank, 5, cart_comm, &requests[count++]);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            zNextSend[i*M + j] = local_data[i*M*K + j*K + exchangeIdxZNext];
        }
    }
    MPI_Isend(zNextSend, N*M, MPI_FLOAT, up_rank, 5, cart_comm, &requests[count++]);
    MPI_Irecv(zNext, N*M, MPI_FLOAT, up_rank, 4, cart_comm, &requests[count++]);
    
    // y
    if (front_rank != MPI_PROC_NULL) {
        for (int i = 0; i < N; i++) {
            for (int k = 0; k < K; k++) {
                yPrevSend[i*K + k] = local_data[i*M*K + 0*K + k];
            }
        }
        MPI_Isend(yPrevSend, N*K, MPI_FLOAT, front_rank, 2, cart_comm, &requests[count++]);
        MPI_Irecv(yPrev, N*K, MPI_FLOAT, front_rank, 3, cart_comm, &requests[count++]);
    }
    if (back_rank != MPI_PROC_NULL) {
        for (int i = 0; i < N; i++) {
            for (int k = 0; k < K; k++) {
                yNextSend[i*K + k] = local_data[i*M*K + (M-1)*K + k];
            }
        }
        MPI_Isend(yNextSend, N*K, MPI_FLOAT, back_rank, 3, cart_comm, &requests[count++]);
        MPI_Irecv(yNext, N*K, MPI_FLOAT, back_rank, 2, cart_comm, &requests[count++]);
    }

    MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
}

float checkAgainstAnalytical(float* u, float* errors, int NLow, int MLow, int KLow, int N, int M, int K, float Lx, float Ly, float Lz, float t, float h) {
    float max_error = 0.0f, x, y, z, analytical, numerical, error;
    int idx;

    #pragma omp parallel for reduction(max:max_error) private(x, y, z, analytical, numerical, error, idx)
    for (int i = 0; i < N; i++) {
        x = (i + NLow) * h;
        for (int j = 0; j < M; j++) {
            y = (j + MLow) * h;
            for (int k = 0; k < K; k++) {
                z = (k + KLow) * h;
                idx = i*M*K + j*K + k;
                analytical = analyticalSolution(Lx, Ly, Lz, x, y, z, t);
                numerical = u[idx];
                error = fabs(numerical - analytical);
                errors[idx] = error;
                if (error > max_error) max_error = error;
            }
        }
    }
    return max_error;
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

void freeBuffersAndTerminate(float** buffers, int numBuffers, MPI_Comm cart_comm) {
    for (int i = 0; i < numBuffers; i++)
        if (buffers[i] != NULL)
            free(buffers[i]);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Usage: %s NGlobal T L\n", argv[0]);
        return 1;
    }

    int required = MPI_THREAD_FUNNELED;
    int provided;
    MPI_Init_thread(&argc, &argv, required, &provided);
    double startFull = MPI_Wtime();
    double startInit = MPI_Wtime();
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);

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

    int uShape = N*M*K;

    float* xPrev = calloc(M*K, sizeof(float));
    float* xNext = calloc(M*K, sizeof(float));
    float* yPrev = calloc(N*K, sizeof(float));
    float* yPrevSend = calloc(N*K, sizeof(float));
    float* yNext = calloc(N*K, sizeof(float));
    float* yNextSend = calloc(N*K, sizeof(float));
    float* zPrev = calloc(N*M, sizeof(float));
    float* zPrevSend = calloc(N*M, sizeof(float));
    float* zNext = calloc(N*M, sizeof(float));
    float* zNextSend = calloc(N*M, sizeof(float));
    int success = (xPrev && xNext && yPrev && yNext && zPrev && zNext && yPrevSend && yNextSend && zPrevSend && zNextSend);
    int globalSuccess;
    MPI_Allreduce(&success, &globalSuccess, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if (!globalSuccess) {
        printf("Error allocating borders, terminating...\n");
        float* borders[10] = {xPrev, xNext, yPrev, yPrevSend, yNext, yNextSend, zPrev, zPrevSend, zNext, zNextSend};
        freeBuffersAndTerminate(borders, 10, cart_comm);
        return -1;
    }

    if (rank == 0) {
        printf("Grid: %dx%dx%d, Processes: %dx%dx%d, omp threads per processor: %d\n", NGlobal, NGlobal, NGlobal, dims[0], dims[1], dims[2], num_threads);
    }

    double timeInit = MPI_Wtime() - startInit;
    double startU0 = MPI_Wtime();

    float* ut0 = makeU0(NLow, MLow, KLow, N, M, K, NGlobal, NGlobal, NGlobal, L, L, L, h);
    float* errors = calloc(uShape, sizeof(float));
    success = ut0 != NULL && errors != NULL;
    MPI_Allreduce(&success, &globalSuccess, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if (!globalSuccess) {
        printf("Error allocating ut0, terminating...\n");
        float* borders[11] = {ut0, xPrev, xNext, yPrev, yPrevSend, yNext, yNextSend, zPrev, zPrevSend, zNext, zNextSend};
        freeBuffersAndTerminate(borders, 11, cart_comm);
        return -1;
    }

    double timeU0 = MPI_Wtime() - startU0;
    double startExchanges = MPI_Wtime();
    
    exchange(ut0, N, M, K, cart_comm, xPrev, xNext, yPrev, yNext,
                      zPrev, zNext, yPrevSend, yNextSend,
                      zPrevSend, zNextSend, left_rank, right_rank, front_rank, back_rank,
                      down_rank, up_rank, NLow, NGlobal, MLow, NGlobal, KLow, NGlobal);
    
    double timeExchanges = MPI_Wtime() - startExchanges;
    double startU1 = MPI_Wtime();
    
    float* ut1 = makeU1(ut0, NLow, MLow, KLow, N, M, K, NGlobal, NGlobal, NGlobal, L, L, L, h, tau, a2,
                       xPrev, xNext, yPrev, yNext, zPrev, zNext);
    success = ut1 != NULL;
    MPI_Allreduce(&success, &globalSuccess, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if (!globalSuccess) {
        printf("Error allocating ut1, terminating...\n");
        float* borders[12] = {ut0, ut1, xPrev, xNext, yPrev, yPrevSend, yNext, yNextSend, zPrev, zPrevSend, zNext, zNextSend};
        freeBuffersAndTerminate(borders, 12, cart_comm);
        return -1;
    }
    double timeU1 = MPI_Wtime() - startU1;
    double startLoop = MPI_Wtime();
    double timeSteps = 0.0;
    int t = 2;
    
    float nextTime = tau * t;
    for (; nextTime < T; ) {
        double startExchanges = MPI_Wtime();
        exchange(ut1, N, M, K, cart_comm, xPrev, xNext, yPrev, yNext,
                          zPrev, zNext, yPrevSend, yNextSend,
                          zPrevSend, zNextSend, left_rank, right_rank, front_rank, back_rank,
                          down_rank, up_rank, NLow, NGlobal, MLow, NGlobal, KLow, NGlobal);
        timeExchanges += MPI_Wtime() - startExchanges;
        double startSteps = MPI_Wtime();
        step(&ut1, &ut0, NLow, MLow, KLow, N, M, K, NGlobal, NGlobal, NGlobal, h, tau, a2,
             xPrev, xNext, yPrev, yNext, zPrev, zNext);
        timeSteps += MPI_Wtime() - startSteps;
        
        if (t % 10 == 0) {
            float maxError = checkAgainstAnalytical(ut1, errors, NLow, MLow, KLow, N, M, K, L, L, L, nextTime, h);
            float maxErrorGlobal;
            MPI_Reduce(&maxError, &maxErrorGlobal, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0)
                printf("Analytical error at time %f: %.8f\n", nextTime, maxErrorGlobal);
            if (save) {
                snprintf(filename, filenameLength, "floats/floats_t%dNLow%dMLow%dKLow%d.bin", t, NLow, MLow, KLow);
                saveFloatsBinary(filename, ut1, uShape);
                snprintf(filename, filenameLength, "errors/errors_t%dNLow%dMLow%dKLow%d.bin", t, NLow, MLow, KLow);
                saveFloatsBinary(filename, errors, uShape);
            }
        }
        nextTime = tau * ++t;
    }
    
    double timeLoop = MPI_Wtime() - startLoop;
    double timeFull = MPI_Wtime() - startFull;

    if (rank == 0) {
        printf("Simulation completed: %d time steps, Time: init-%f, U0-%f, exchanges-%f, U1-%f, steps-%f, Loop-%f, Full-%f\n", t, timeInit, timeU0, timeExchanges, timeU1, timeSteps, timeLoop, timeFull);
    }

    float* borders[12] = {ut0, ut1, xPrev, xNext, yPrev, yPrevSend, yNext, yNextSend, zPrev, zPrevSend, zNext, zNextSend};
    freeBuffersAndTerminate(borders, 12, cart_comm);
    return 0;
}
