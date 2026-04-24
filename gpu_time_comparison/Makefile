ARCH = sm_60
HOST_COMP = mpicc

MPI_INC = /opt/ibm/spectrum_mpi/include
MPI_LIB = /opt/ibm/spectrum_mpi/lib

SRC = mpi_cuda_timing.cu
TARGET = mpi_cuda_t.out

NVCC_FLAGS = -I$(MPI_INC) -arch=$(ARCH) -O3 -std=c++11 -ccbin $(HOST_COMP)
NVCC_LIBS = -L$(MPI_LIB) -lmpiprofilesupport -lmpi_ibm -lm

$(TARGET): $(SRC)
	nvcc $(NVCC_FLAGS) $(SRC) $(NVCC_LIBS) -o $(TARGET)

clean:
	rm -f $(TARGET) *.o

.PHONY: clean
