# Команды компиляции

```
xlc_r -mcpu=power8 -qsmp=omp -O2 no_parallel.c -o no_parallel.out

xlc_r -mcpu=power8 -qsmp=omp -O2 openmp.c -o openmp.out

module load SpectrumMPI/10.1.0

OMPI_CC=gcc mpicc -I/opt/ibm/spectrum_mpi/include -L/opt/ibm/spectrum_mpi/lib -lmpiprofilesupport -lmpi_ibm -lm -O2 -std=c99 mpi.c -o mpi.out

OMPI_CC=gcc mpicc -I/opt/ibm/spectrum_mpi/include -L/opt/ibm/spectrum_mpi/lib -lmpiprofilesupport -lmpi_ibm -lm -O2 -std=c99 -fopenmp mpi_omp.c -o mpi_omp.out

nvcc -I/opt/ibm/spectrum_mpi/include -L/opt/ibm/spectrum_mpi/lib -lmpiprofilesupport -lmpi_ibm -lm -O3 -std=c++11 mpi_cuda.cu -o mpi_cuda.out

OMPI_CC=gcc mpicc -I/opt/ibm/spectrum_mpi/include -L/opt/ibm/spectrum_mpi/lib -lmpiprofilesupport -lmpi_ibm -lm -O3 -std=c99 -fopenmp mpi_omp_timing.c -o mpi_omp_t.out

nvcc -I/opt/ibm/spectrum_mpi/include -L/opt/ibm/spectrum_mpi/lib -lmpiprofilesupport -lmpi_ibm -lm -O3 -std=c++11 mpi_cuda_timing.cu -o mpi_cuda_t.out
```

(Для mpi_cuda.cu и mpi_cuda_timing.cu также есть Makefile в соответствии с требованиями к ГПУ заданию).

# Команды запуска экспериментов

Последовательная программа:

```
mpisubmit.pl -p 1 -t 1 no_parallel.out 128 1.0 1.0
mpisubmit.pl -p 1 -t 1 no_parallel.out 128 3.14159265359 3.14159265359
mpisubmit.pl -p 1 -t 1 no_parallel.out 256 1.0 1.0
mpisubmit.pl -p 1 -t 1 no_parallel.out 256 3.14159265359 3.14159265359
```

OpenMP программа:

```
mpisubmit.pl -p 1 -t 1 openmp.out 128 1.0 1.0
mpisubmit.pl -p 1 -t 1 openmp.out 128 3.14159265359 3.14159265359
mpisubmit.pl -p 1 -t 2 openmp.out 128 1.0 1.0
mpisubmit.pl -p 1 -t 2 openmp.out 128 3.14159265359 3.14159265359
mpisubmit.pl -p 1 -t 4 openmp.out 128 1.0 1.0
mpisubmit.pl -p 1 -t 4 openmp.out 128 3.14159265359 3.14159265359
mpisubmit.pl -p 1 -t 8 openmp.out 128 1.0 1.0
mpisubmit.pl -p 1 -t 8 openmp.out 128 3.14159265359 3.14159265359
mpisubmit.pl -p 1 -t 1 openmp.out 256 1.0 1.0
mpisubmit.pl -p 1 -t 1 openmp.out 256 3.14159265359 3.14159265359
mpisubmit.pl -p 1 -t 2 openmp.out 256 1.0 1.0
mpisubmit.pl -p 1 -t 2 openmp.out 256 3.14159265359 3.14159265359
mpisubmit.pl -p 1 -t 4 openmp.out 256 1.0 1.0
mpisubmit.pl -p 1 -t 4 openmp.out 256 3.14159265359 3.14159265359
mpisubmit.pl -p 1 -t 8 openmp.out 256 1.0 1.0
mpisubmit.pl -p 1 -t 8 openmp.out 256 3.14159265359 3.14159265359
bsub <omp_job_32_256_314.lsf
bsub <omp_job_32_256_1.lsf
bsub <omp_job_16_256_314.lsf
bsub <omp_job_16_256_1.lsf
bsub <omp_job_16_128_314.lsf
bsub <omp_job_16_128_1.lsf
```

MPI программа:

```
mpisubmit.pl -p 1 -t 1 mpi.out 128 1.0 1.0
mpisubmit.pl -p 1 -t 1 mpi.out 128 3.14159265359 3.14159265359
mpisubmit.pl -p 2 -t 1 mpi.out 128 1.0 1.0
mpisubmit.pl -p 2 -t 1 mpi.out 128 3.14159265359 3.14159265359
mpisubmit.pl -p 4 -t 1 mpi.out 128 1.0 1.0
mpisubmit.pl -p 4 -t 1 mpi.out 128 3.14159265359 3.14159265359
mpisubmit.pl -p 8 -t 1 mpi.out 128 1.0 1.0
mpisubmit.pl -p 8 -t 1 mpi.out 128 3.14159265359 3.14159265359
mpisubmit.pl -p 16 -t 1 mpi.out 128 1.0 1.0
mpisubmit.pl -p 16 -t 1 mpi.out 128 3.14159265359 3.14159265359
mpisubmit.pl -p 32 -t 1 mpi.out 128 1.0 1.0
mpisubmit.pl -p 32 -t 1 mpi.out 128 3.14159265359 3.14159265359
mpisubmit.pl -p 1 -t 1 mpi.out 256 1.0 1.0
mpisubmit.pl -p 1 -t 1 mpi.out 256 3.14159265359 3.14159265359
mpisubmit.pl -p 2 -t 1 mpi.out 256 1.0 1.0
mpisubmit.pl -p 2 -t 1 mpi.out 256 3.14159265359 3.14159265359
mpisubmit.pl -p 4 -t 1 mpi.out 256 1.0 1.0
mpisubmit.pl -p 4 -t 1 mpi.out 256 3.14159265359 3.14159265359
mpisubmit.pl -p 8 -t 1 mpi.out 256 1.0 1.0
mpisubmit.pl -p 8 -t 1 mpi.out 256 3.14159265359 3.14159265359
mpisubmit.pl -p 16 -t 1 mpi.out 256 1.0 1.0
mpisubmit.pl -p 16 -t 1 mpi.out 256 3.14159265359 3.14159265359
mpisubmit.pl -p 32 -t 1 mpi.out 256 1.0 1.0
mpisubmit.pl -p 32 -t 1 mpi.out 256 3.14159265359 3.14159265359
```

MPI_OMP программа:

```
mpisubmit.pl -p 4 -t 1 mpi_omp.out 128 1.0 1.0
mpisubmit.pl -p 4 -t 1 mpi_omp.out 128 3.14159265359 3.14159265359
mpisubmit.pl -p 4 -t 2 mpi_omp.out 128 1.0 1.0
mpisubmit.pl -p 4 -t 2 mpi_omp.out 128 3.14159265359 3.14159265359
mpisubmit.pl -p 4 -t 4 mpi_omp.out 128 1.0 1.0
mpisubmit.pl -p 4 -t 4 mpi_omp.out 128 3.14159265359 3.14159265359
mpisubmit.pl -p 4 -t 8 mpi_omp.out 128 1.0 1.0
mpisubmit.pl -p 4 -t 8 mpi_omp.out 128 3.14159265359 3.14159265359
mpisubmit.pl -p 4 -t 1 mpi_omp.out 256 1.0 1.0
mpisubmit.pl -p 4 -t 1 mpi_omp.out 256 3.14159265359 3.14159265359
mpisubmit.pl -p 4 -t 2 mpi_omp.out 256 1.0 1.0
mpisubmit.pl -p 4 -t 2 mpi_omp.out 256 3.14159265359 3.14159265359
mpisubmit.pl -p 4 -t 4 mpi_omp.out 256 1.0 1.0
mpisubmit.pl -p 4 -t 4 mpi_omp.out 256 3.14159265359 3.14159265359
mpisubmit.pl -p 4 -t 8 mpi_omp.out 256 1.0 1.0
mpisubmit.pl -p 4 -t 8 mpi_omp.out 256 3.14159265359 3.14159265359
mpisubmit.pl -p 8 -t 1 mpi_omp.out 128 1.0 1.0
mpisubmit.pl -p 8 -t 1 mpi_omp.out 128 3.14159265359 3.14159265359
mpisubmit.pl -p 8 -t 2 mpi_omp.out 128 1.0 1.0
mpisubmit.pl -p 8 -t 2 mpi_omp.out 128 3.14159265359 3.14159265359
mpisubmit.pl -p 8 -t 4 mpi_omp.out 128 1.0 1.0
mpisubmit.pl -p 8 -t 4 mpi_omp.out 128 3.14159265359 3.14159265359
mpisubmit.pl -p 8 -t 8 mpi_omp.out 128 1.0 1.0
mpisubmit.pl -p 8 -t 8 mpi_omp.out 128 3.14159265359 3.14159265359
mpisubmit.pl -p 8 -t 1 mpi_omp.out 256 1.0 1.0
mpisubmit.pl -p 8 -t 1 mpi_omp.out 256 3.14159265359 3.14159265359
mpisubmit.pl -p 8 -t 2 mpi_omp.out 256 1.0 1.0
mpisubmit.pl -p 8 -t 2 mpi_omp.out 256 3.14159265359 3.14159265359
mpisubmit.pl -p 8 -t 4 mpi_omp.out 256 1.0 1.0
mpisubmit.pl -p 8 -t 4 mpi_omp.out 256 3.14159265359 3.14159265359
mpisubmit.pl -p 8 -t 8 mpi_omp.out 256 1.0 1.0
mpisubmit.pl -p 8 -t 8 mpi_omp.out 256 3.14159265359 3.14159265359
```

MPI_GPU программа:

```
mpisubmit.pl -p 1 -t 1 -g 1 mpi_cuda.out 128 1.0 1.0
mpisubmit.pl -p 1 -t 1 -g 1 mpi_cuda.out 128 3.14159265359 3.14159265359
mpisubmit.pl -p 1 -t 1 -g 1 mpi_cuda.out 256 1.0 1.0
mpisubmit.pl -p 1 -t 1 -g 1 mpi_cuda.out 256 3.14159265359 3.14159265359
mpisubmit.pl -p 2 -t 1 -g 2 mpi_cuda.out 256 1.0 1.0
mpisubmit.pl -p 2 -t 1 -g 2 mpi_cuda.out 256 3.14159265359 3.14159265359
```

Дополнительное задание - MPI_OMP и MPI_GPU программы.

```
mpisubmit.pl -p 20 -t 8 mpi_omp_t.out 512 1.0 1.0
mpisubmit.pl -p 20 -t 8 mpi_omp_t.out 512 3.14159265359 3.14159265359
mpisubmit.pl -p 1 -t 1 -g 1 mpi_cuda_t.out 512 1.0 1.0
mpisubmit.pl -p 1 -t 1 -g 1 mpi_cuda_t.out 512 3.14159265359 3.14159265359
mpisubmit.pl -p 2 -t 1 -g 2 mpi_cuda_t.out 512 1.0 1.0
mpisubmit.pl -p 2 -t 1 -g 2 mpi_cuda_t.out 512 3.14159265359 3.14159265359
```

# Скрипты omp_job.lsf для OpenMP части:

omp_job_32_256_314.lsf

```
#BSUB -n 4
#BSUB -W 00:15
#BSUB -o "omp_job.%J.out"
OMP_NUM_THREADS=32 ./openmp.out 256 3.14159265359 3.14159265359
```

omp_job_32_256_1.lsf

```
#BSUB -n 4
#BSUB -W 00:15
#BSUB -o "omp_job.%J.out"
OMP_NUM_THREADS=32 ./openmp.out 256 1.0 1.0
```

omp_job_16_256_314.lsf

```
#BSUB -n 4
#BSUB -W 00:15
#BSUB -o "omp_job.%J.out"
OMP_NUM_THREADS=16 ./openmp.out 256 3.14159265359 3.14159265359
```

omp_job_16_256_1.lsf

```
#BSUB -n 4
#BSUB -W 00:15
#BSUB -o "omp_job.%J.out"
OMP_NUM_THREADS=16 ./openmp.out 256 1.0 1.0
```

omp_job_16_128_314.lsf

```
#BSUB -n 4
#BSUB -W 00:15
#BSUB -o "omp_job.%J.out"
OMP_NUM_THREADS=16 ./openmp.out 128 3.14159265359 3.14159265359
```

omp_job_16_128_1.lsf

```
#BSUB -n 4
#BSUB -W 00:15
#BSUB -o "omp_job.%J.out"
OMP_NUM_THREADS=16 ./openmp.out 128 1.0 1.0
```
