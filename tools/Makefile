# Compiler and Flags
CC = gcc
CFLAGS = -O2
SHARE = -g -fPIC -shared

# Targets
TARGETS = getpagesize thpoff gemm-timer.so gemm-timer-debug.so move_page

# Default target
all: $(TARGETS)

# Rules
getpagesize: getpagesize.c
	$(CC) $(CFLAGS) $^ -o $@

thpoff: thpoff.c
	$(CC) $(CFLAGS) $^ -o $@

gemm-timer.so: gemm-timer.c
	$(CC) $(SHARE) $^ -o $@

gemm-timer-debug.so: gemm-timer.c
	$(CC) $(SHARE) -DDEBUG $^ -o $@

move_page: move_page_timer.c
	$(CC) $(CFLAGS) $^ -o $@ -lnuma -fopenmp

# Clean up
clean:
	rm -f $(TARGETS)

