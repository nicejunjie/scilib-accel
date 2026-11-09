


CC = pgcc
FC = pgf90

NVHOME := $(shell pgf90_path=$$(which pgf90) && dirname "$$(dirname "$$(dirname "$$pgf90_path")")")
CUDA_VER := $(shell nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')

# Locate cuBLAS/cuSolver. First directory containing libcublas.so wins:
#   user override (make CUDA_MATH_LIB=...) > site module var > NVHPC versioned
#   > NVHPC unversioned > standalone CUDA toolkit
CUDA_MATH_CANDIDATES := $(CUDA_MATH_LIB) $(TACC_NVIDIA_MATH_LIB) \
    $(if $(CUDA_VER),$(NVHOME)/math_libs/$(CUDA_VER)/lib64) \
    $(NVHOME)/math_libs/lib64 \
    $(if $(CUDA_HOME),$(CUDA_HOME)/lib64) /usr/local/cuda/lib64
CUDA_MATH_LIB := $(firstword $(foreach d,$(CUDA_MATH_CANDIDATES),$(if $(wildcard $(d)/libcublas.so),$(d))))
ifeq ($(CUDA_MATH_LIB),)
  $(error libcublas.so not found in: $(CUDA_MATH_CANDIDATES). Set CUDA_MATH_LIB=/path/to/lib64)
endif

CUBLAS = -L$(CUDA_MATH_LIB) -Wl,-rpath,$(CUDA_MATH_LIB) -lcublas
CUSOLVER = -lcusolver
CURT =
#CUINCLUDE = # $(NVHOME)/cuda/include

GPUARCH=NVIDIA

FRIDA_DIR := frida

INCLUDE = -I. -I./blas/$(GPUARCH) -I./utils -I./$(FRIDA_DIR)

BLAS = -Mnvpl
LD_FLAGS = -ldl -lrt -lresolv -lm -pthread -Wl,-z,noexecstack,--gc-sections -lnuma
LIBS = $(CUBLAS) $(CURT) $(CUSOLVER)

TARGET1 = scilib-dbi.so
TARGET2 = scilib-dl.so

#MEMMODEL= -gpu=unified 
#MEMMODEL= -gpu=nomanaged
CPPFLAGS += -D$(GPUARCH)


CFLAGS = -O2 -mp -fPIC -w  -g $(INCLUDE) $(CPPFLAGS) $(MEMMODEL) -cuda
CFLAGS1 = $(CFLAGS) -DDBI -I./$(FRIDA_DIR)
FFLAGS = -O2 -mp -g -mcmodel=large #$(MEMMODEL)

#----------------------------

BLAS_SRCS = $(wildcard blas/$(GPUARCH)/*.c)
UTIL_SRCS = $(wildcard utils/*.c)
UTIL_CU_SRCS = utils/gpu_migrate.cu
COMMON_SRCS = init.c nvidia.c $(UTIL_SRCS) $(BLAS_SRCS)
#COMMON_SRCS = init.c nvidia.c utils.c blas/$(GPUARCH)/sgemm.c blas/$(GPUARCH)/dgemm.c blas/$(GPUARCH)/cgemm.c blas/$(GPUARCH)/zgemm.c 

UTIL_CU_OBJ = $(UTIL_CU_SRCS:.cu=.o)

SRCS1 = main-dbi.c
SRCS1ALL = $(COMMON_SRCS) $(SRCS1)
OBJ1 = $(patsubst %.c,%-dbi.o,$(COMMON_SRCS)) $(SRCS1:.c=.o) $(UTIL_CU_OBJ)

SRCS2 = main-dl.c
SRCS2ALL = $(COMMON_SRCS) $(SRCS2)
OBJ2 = $(patsubst %.c,%-dl.o,$(COMMON_SRCS))  $(SRCS2:.c=.o) $(UTIL_CU_OBJ)



all: dbi dl test_dgemm.x

dbi: versions $(FRIDA_DIR) $(TARGET1)
$(TARGET1): $(OBJ1) 
	@echo "Building DBI based SCILIB-Accel"
	$(CC) -o $@ -shared -ffunction-sections -fdata-sections $^ ./$(FRIDA_DIR)/libfrida-gum.a ${LD_FLAGS} ${CFLAGS} $(LIBS)	
	chmod go+rx $(TARGET1)

dl: versions $(TARGET2)
$(TARGET2): $(OBJ2)
	@echo "Building DL based SCILIB-Accel"
	$(CC) -o $@ -shared  $^  ${LD_FLAGS} ${CFLAGS} $(LIBS)
	chmod go+rx $(TARGET2)

%-dbi.o: %.c
	$(CC) $(CFLAGS1) -c $< -o $@

%-dl.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	nvcc -O2 -c $< -o $@ -Xcompiler -fPIC

# ------------------------- setup frida for DBI --------------------------

# Determine the architecture
CPUARCH := $(shell uname -m)
# Set architecture-specific variables
FRIDA_HOME := https://github.com/frida/frida/releases/download/
FRIDA_VERSION := 16.2.1
ifeq ($(CPUARCH),x86_64)
    FRIDA_DEVKIT_URL := $(FRIDA_HOME)$(FRIDA_VERSION)/frida-gum-devkit-$(FRIDA_VERSION)-linux-x86_64.tar.xz
    FRIDA_DEVKIT_FILE := frida-gum-devkit-$(FRIDA_VERSION)-linux-x86_64.tar.xz
else ifeq ($(CPUARCH),aarch64)
    FRIDA_DEVKIT_URL := $(FRIDA_HOME)$(FRIDA_VERSION)/frida-gum-devkit-$(FRIDA_VERSION)-linux-arm64.tar.xz
    FRIDA_DEVKIT_FILE := frida-gum-devkit-$(FRIDA_VERSION)-linux-arm64.tar.xz
else ifeq ($(CPUARCH),arm64)
    FRIDA_DEVKIT_URL := $(FRIDA_HOME)$(FRIDA_VERSION)/frida-gum-devkit-$(FRIDA_VERSION)-linux-arm64.tar.xz
    FRIDA_DEVKIT_FILE := frida-gum-devkit-$(FRIDA_VERSION)-linux-arm64.tar.xz
else
    $(error Unsupported architecture: $(CPUARCH))
endif

$(FRIDA_DIR): $(FRIDA_DIR)/$(FRIDA_DEVKIT_FILE)
	@if [ ! -e "$(FRIDA_DIR)/frida-gum.h" ]; then \
        tar -xvf $(FRIDA_DIR)/$(FRIDA_DEVKIT_FILE) -C $(FRIDA_DIR); \
        fi

$(FRIDA_DIR)/$(FRIDA_DEVKIT_FILE):
	mkdir -p $(FRIDA_DIR)
	curl -s  -L $(FRIDA_DEVKIT_URL) -o $(FRIDA_DIR)/$(FRIDA_DEVKIT_FILE)

# ------------------------------------------------------------------------

NVPL_INC := $(firstword $(wildcard $(TACC_NVPL_INC) $(NVHOME)/math_libs/nvpl/include))

# Print the toolchain and every dependency version (runs before dbi/dl).
# Library versions come from the resolved .so filename (libX.so -> libX.so.M.m.p.b).
.PHONY: versions print-nvhome
print-nvhome: versions
versions:
	@echo "==================== scilib-accel build configuration ===================="
	@echo "  NVHOME         : $(NVHOME)"
	@echo "  CUDA_MATH_LIB  : $(CUDA_MATH_LIB)"
	@echo "  CC             : $$($(CC) --version 2>/dev/null | sed -n 's/^\(.*[0-9][0-9]*\.[0-9][0-9]*-[0-9]*\).*/\1/p' | head -1)"
	@echo "  FC             : $$($(FC) --version 2>/dev/null | sed -n 's/^\(.*[0-9][0-9]*\.[0-9][0-9]*-[0-9]*\).*/\1/p' | head -1)"
	@echo "  nvcc / CUDA    : $$(nvcc --version 2>/dev/null | sed -n 's/.*release [^,]*, V\(.*\)/\1/p') ($$(which nvcc 2>/dev/null))"
	@echo "  cuBLAS         : $$(basename $$(readlink -f $(CUDA_MATH_LIB)/libcublas.so) | sed 's/libcublas\.so\.//')"
	@echo "  cuSolver       : $$(basename $$(readlink -f $(CUDA_MATH_LIB)/libcusolver.so) | sed 's/libcusolver\.so\.//')"
	@echo "  NVPL BLAS      : $$(awk '/define NVPL_BLAS_VERSION_(MAJOR|MINOR|PATCH)/{v=v (v?".":"") $$3} END{print v?v:"not found"}' $(NVPL_INC)/nvpl_blas_version.h 2>/dev/null)"
	@echo "  NVPL LAPACK    : $$(awk '/define NVPL_LAPACK_VERSION_(MAJOR|MINOR|PATCH)/{v=v (v?".":"") $$3} END{print v?v:"not found"}' $(NVPL_INC)/nvpl_lapack_version.h 2>/dev/null)"
	@echo "  FRIDA gum      : $(FRIDA_VERSION) ($(CPUARCH)) $$([ -e $(FRIDA_DIR)/libfrida-gum.a ] && echo '[present]' || echo '[will download]')"
	@echo "  libnuma        : $$(rpm -q --qf '%{VERSION}' numactl-libs 2>/dev/null || basename $$(readlink -f /usr/lib64/libnuma.so 2>/dev/null) 2>/dev/null)"
	@echo "  glibc          : $$(ldd --version 2>/dev/null | head -1 | awk '{print $$NF}')"
	@echo "  GPU driver     : $$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 || true)$$(command -v nvidia-smi >/dev/null || echo 'n/a (no nvidia-smi on this node)')"
	@echo "=========================================================================="


# ---------------------- run tests ---------------------------------------
test_dgemm.x: test_dgemm.f90
	pgf90 -o $@ $^ $(BLAS) ${FFLAGS} ${MEMMODEL}

run: run1 run2

run1: test_dgemm.x $(TARGET1)
	export OMP_NUM_THREADS=72
	time echo 32 2400 93536 5 | LD_PRELOAD=./$(TARGET1) ./test_dgemm.x
	#time echo 6000 4000 8000 5 | LD_PRELOAD=./$(TARGET1) ./test_dgemm.x

run2: test_dgemm.x $(TARGET2)
	export OMP_NUM_THREADS=72
	time echo 20816 2400 32000 10 | LD_PRELOAD=./$(TARGET2) ./test_dgemm.x
# ------------------------------------------------------------------------

.PHONY: clean
clean:
	rm -rf $(OBJ1) $(OBJ2) $(UTIL_CU_OBJ) $(FRIDA_DIR)

.PHONY: veryclean
veryclean:
	rm -rf test_dgemm.x $(TARGET1) $(TARGET2) $(OBJ1) $(OBJ2) $(FRIDA_DIR)
