# Makefile for some simple examples in this directory.
  # On bridges, you need to run `source prep.sh` to set
  # your environment before invoking this Makefile.
  BCL_HOME=$(PWD)/../../../../
  SOURCES += $(wildcard sparse_test.cu)
  TARGETS := $(patsubst %.cu, %, $(SOURCES))
  CXX=nvcc
 # NVSHMEM_FLAGS=-DNVSHMEM_TARGET -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_37,code=sm_37 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=  sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -ccbin g++  -I$(CUDA_HOME)/include/nvprefix -I$(NVSHMEM_HOME)/include/nvprefix -I$(MPI_HOME)/  include -DENABLE_MPI_SUPPORT -rdc=true  -L$(NVSHMEM_HOME)/lib/nvprefix -lnvshmem -lcuda -L$(CUDA_HOME)/lib64 -lcudart -L$(MPI_HOME)/lib -lmpi -lopen-rte -lopen-pal -lm -ln  uma -ldl -lrt -lutil
 NSPARSE_FLAGS = -I/ccs/home/b2v/src/research/nsparse/cuda-cpp/inc -I$(CUDAPATH)/samples/common/inc
 CUSPARSE_FLAGS = -I/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/math_libs/include -L/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/math_libs/lib64 -lcusparse -lcublas
 CUSP_FLAGS = -I/global/homes/j/jbellav/HnS-SpGEMM/cusplibrary
 NVSHMEM_FLAGS=-DNVSHMEM_TARGET -arch=sm_80  -I$(CUDA_HOME)/include -I$(NVSHMEM_HOME)/include -I$(MPI_HOME)/include -DENABLE_MPI_SUPPORT -rdc=true  -L$(NVSHMEM_HO  ME)/lib -lnvshmem -lcuda -L$(CUDA_HOME)/lib64 -lcudart -L$(MPI_HOME)/lib -lm -lnuma -ldl -lrt -lutil  -L$(OLCF_ESSL_ROOT)/lib64  -lmpich -L/usr/lib64 -lgdrapi -L/opt/cray/pe/lib64 -lmpi_gtl_cuda

CXXFLAGS = -std=c++17 -O3 -I$(BCL_HOME) -DMPI --expt-extended-lambda $(NVSHMEM_FLAGS) $(CUSPARSE_FLAGS) $(CUSP_FLAGS)
all: $(TARGETS)
%: %.cu
	$(CXX) -o $@ $^ $(CXXFLAGS)
clean: 
	rm $(TARGETS)
