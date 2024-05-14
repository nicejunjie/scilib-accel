CC = gcc

CFLAGS1 =  -O2

all: getpagesize thpoff gemm-timer.so

getpagesize: getpagesize.c
	$(CC) $(CFLAGS1) $^ -o $@

thpoff: thpoff.c
	$(CC) $(CFLAGS1) $^ -o $@

# -------- 

#CPPFLAGS2 = -DDEBUG
CFLAGS2 = -g -fPIC -shared $(CPPFLAGS2)  

gemm-timer.so: gemm-timer.c
	$(CC) $(CFLAGS2) $^ -o $@


# --------

clean:
	rm -f getpagesize thpoff gemm-timer

