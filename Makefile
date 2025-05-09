


CC = pgcc
FC = pgf90

NVHOME := $(shell pgf90_path=$$(which pgf90) && dirname "$$(dirname "$$(dirname "$$pgf90_path")")")
CUBLAS = $(NVHOME)/math_libs/lib64/libcublas.so
CUSOLVER = $(NVHOME)/math_libs/lib64/libcusolver.so
CURT = $(NVHOME)/cuda/lib64/libcudart.so
CUINCLUDE = $(NVHOME)/cuda/include

GPUARCH=NVIDIA

FRIDA_DIR := frida

INCLUDE = -I. -I./blas/$(GPUARCH) -I./utils -I$(CUINCLUDE) -I./$(FRIDA_DIR)

BLAS = -Mnvpl
LD_FLAGS = -ldl -lrt -lresolv -lm -pthread -Wl,-z,noexecstack,--gc-sections -lnuma
LIBS = $(CUBLAS) $(CURT) $(CUSOLVER)

TARGET1 = scilib-dbi.so
TARGET2 = scilib-dl.so

#MEMMODEL= -gpu=unified 
#MEMMODEL= -gpu=nomanaged
CPPFLAGS += -D$(GPUARCH)


CFLAGS = -O2 -mp -fPIC -w  -g $(INCLUDE) $(CPPFLAGS) $(MEMMODEL)
CFLAGS1 = $(CFLAGS) -DDBI -I./$(FRIDA_DIR)
FFLAGS = -O2 -mp -g -mcmodel=large #$(MEMMODEL)

#----------------------------

BLAS_SRCS = $(wildcard blas/$(GPUARCH)/*.c)
UTIL_SRCS = $(wildcard utils/*.c)
COMMON_SRCS = init.c nvidia.c scilib_pthread_wrap.c $(UTIL_SRCS) $(BLAS_SRCS)
#COMMON_SRCS = init.c nvidia.c utils.c blas/$(GPUARCH)/sgemm.c blas/$(GPUARCH)/dgemm.c blas/$(GPUARCH)/cgemm.c blas/$(GPUARCH)/zgemm.c 

SRCS1 = main-dbi.c
SRCS1ALL = $(COMMON_SRCS) $(SRCS1)
OBJ1 = $(patsubst %.c,%-dbi.o,$(COMMON_SRCS)) $(SRCS1:.c=.o)

SRCS2 = main-dl.c
SRCS2ALL = $(COMMON_SRCS) $(SRCS2)
OBJ2 = $(patsubst %.c,%-dl.o,$(COMMON_SRCS))  $(SRCS2:.c=.o)



all: dbi dl test_dgemm.x

dbi: print-nvhome $(FRIDA_DIR) $(TARGET1)
$(TARGET1): $(OBJ1) 
	@echo "Building DBI based SCILIB-Accel"
	$(CC) -o $@ -shared -ffunction-sections -fdata-sections $^ ./$(FRIDA_DIR)/libfrida-gum.a ${LD_FLAGS} ${CFLAGS} $(LIBS)	
	chmod go+rx $(TARGET1)

dl: print-nvhome $(TARGET2)
$(TARGET2): $(OBJ2)
	@echo "Building DL based SCILIB-Accel"
	$(CC) -o $@ -shared  $^  ${LD_FLAGS} ${CFLAGS} $(LIBS)
	chmod go+rx $(TARGET2)

%-dbi.o: %.c
	$(CC) $(CFLAGS1) -c $< -o $@

%-dl.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

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

.PHONY: print-nvhome
print-nvhome:
	@echo "NVHOME = $(NVHOME)"


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
	rm -rf $(OBJ1) $(OBJ2) $(FRIDA_DIR)

.PHONY: veryclean
veryclean:
	rm -rf test_dgemm.x $(TARGET1) $(TARGET2) $(OBJ1) $(OBJ2) $(FRIDA_DIR)
