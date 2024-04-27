CC = gcc
CFLAGS = -Wall -Werror

all: getpagesize thpoff

getpagesize: getpagesize.c
	$(CC) $(CFLAGS) $^ -o $@

thpoff: thpoff.c
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm -f getpagesize thpoff

