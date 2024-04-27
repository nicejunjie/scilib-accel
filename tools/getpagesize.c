#include <stdio.h>
#include <unistd.h>

int main() {
    long pageSize = sysconf(_SC_PAGE_SIZE);

    if (pageSize == -1) {
        perror("Failed to get page size");
        return 1;
    }

    printf("Page size: %ld bytes\n", pageSize);

    return 0;
}

