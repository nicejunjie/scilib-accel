CC = gcc

CFLAGS1 =  -O2

all: getpagesize thpoff gemm-timer.so gemm-timer-debug.so move_page

getpagesize: getpagesize.c
	$(CC) $(CFLAGS1) $^ -o $@

thpoff: thpoff.c
	$(CC) $(CFLAGS1) $^ -o $@

# -------- 

#CPPFLAGS2 = -DDEBUG
CFLAGS2 = -g -fPIC -shared $(CPPFLAGS2)  

gemm-timer.so: gemm-timer.c
	$(CC) $(CFLAGS2) $^ -o $@

gemm-timer-debug.so: gemm-timer.c
	$(CC) -DDEBUG $(CFLAGS2) $^ -o $@


# --------

move_page: move_page_timer.c
	$(CC) -O2 $^ -o $@  -lnuma -fopenmp

clean:
	rm -f getpagesize thpoff gemm-timer move_page gemm-timer.so gemm-timer-debug.so

