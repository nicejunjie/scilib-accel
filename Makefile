
NV = 24.3
NVHOME = /scratch1/07893/junjieli/grace-hopper/junjieli/soft/nvhpc/24.3/Linux_aarch64/24.3
CUBLAS = $(NVHOME)/math_libs/lib64/libcublas.so
CURT = $(NVHOME)/cuda/lib64/libcudart.so
CUINCLUDE = $(NVHOME)/cuda/include

GPUARCH=nvidia

INCLUDE = -I. -I./blas/$(GPUARCH) -I$(CUINCLUDE)

BLAS = -lblas
LD_FLAGS = -ldl -lrt -lresolv -lm -pthread -Wl,-z,noexecstack,--gc-sections -lnuma
LIBS = $(CUBLAS) $(CURT)

TARGET1 = scilib-dbi.so
TARGET2 = scilib-dl.so

CC = pgcc
FC = pgf90

# CPPFLAGS = -DGPUCOPY
CPPFLAGS = -DAUTO_NUMA
#MEMMODEL= -gpu=unified




CFLAGS = -O2 -mp -fPIC -w  -g $(INCLUDE) $(CPPFLAGS) $(MEMMODEL)
CFLAGS1 = $(CFLAGS) -DDBI -I./$(FRIDA_DIR)
FFLAGS = -O2 -mp -g -mcmodel=large $(MEMMODEL)

COMMON_SRCS = utils.c blas/$(GPUARCH)/dgemm.c blas/$(GPUARCH)/zgemm.c

SRCS1 = main-dbi.c
SRCS1ALL = $(COMMON_SRCS) $(SRCS1)
OBJ1 = $(patsubst %.c,%-dbi.o,$(COMMON_SRCS)) $(SRCS1:.c=.o)

SRCS2 = main-dl.c
SRCS2ALL = $(COMMON_SRCS) $(SRCS2)
OBJ2 = $(patsubst %.c,%-dl.o,$(COMMON_SRCS))  $(SRCS2:.c=.o)


FRIDA_DIR = frida

all: $(TARGET1) $(TARGET2) test_dgemm.x

dbi: $(TARGET1)
$(TARGET1): $(OBJ1) | $(FRIDA_DIR)
	@echo "Building DBI based SCILIB-Accel"
	$(CC) -o $@ -shared -ffunction-sections -fdata-sections $^ ./$(FRIDA_DIR)/libfrida-gum.a ${LD_FLAGS} ${CFLAGS} $(LIBS)

dl: $(TARGET2)
$(TARGET2): $(OBJ2)
	@echo "Building DL based SCILIB-Accel"
	$(CC) -o $@ -shared  $^  ${LD_FLAGS} ${CFLAGS} $(LIBS)

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
	tar -xvf $(FRIDA_DIR)/$(FRIDA_DEVKIT_FILE) -C $(FRIDA_DIR) #--strip-components=1

$(FRIDA_DIR)/$(FRIDA_DEVKIT_FILE):
	mkdir -p $(FRIDA_DIR)
	curl -s  -L $(FRIDA_DEVKIT_URL) -o $(FRIDA_DIR)/$(FRIDA_DEVKIT_FILE)

# ------------------------------------------------------------------------


test_dgemm.x: test_dgemm.f90 utils.o
	pgf90 -o $@ $^  -g -lnuma $(BLAS) ${FFLAGS}

run: run1 run2

run1: test_dgemm.x $(TARGET)
	export OMP_NUM_THREADS=72
	time echo 20816 2400 32 5 | LD_PRELOAD=./$(TARGET1) ./test_dgemm.x

run2: test_dgemm.x $(TARGET)
	export OMP_NUM_THREADS=72
	time echo 20816 2400 32 5 | LD_PRELOAD=./$(TARGET2) ./test_dgemm.x

.PHONY: clean

clean:
	rm -rf test_dgemm.x $(TARGET) $(OBJ1) $(OBJ2) $(FRIDA_DIR)
