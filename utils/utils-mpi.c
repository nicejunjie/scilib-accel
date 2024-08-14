
#include <unistd.h>
#include <stdio.h>


int check_MPI() {
    char* pmi_rank = getenv("PMI_RANK");
    //char* pmix_rank = getenv("MPIX_RANK");
    char* mvapich_rank = getenv("MV2_COMM_WORLD_RANK");
    char* ompi_rank = getenv("OMPI_COMM_WORLD_RANK");
    //char* slurm_rank = getenv("SLURM_PROCID");

    if (pmi_rank != NULL  || mvapich_rank != NULL || ompi_rank != NULL )
        return 1;
    else
        return 0;
}

int get_MPI_local_rank() {
    char* pmi_rank = getenv("MPI_LOCALRANKID");
    char* mvapich_rank = getenv("MV2_COMM_WORLD_LOCAL_RANK");
    char* ompi_rank = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (pmi_rank != NULL)
        return atoi(pmi_rank);
    else if (mvapich_rank != NULL)
        return atoi(mvapich_rank);
    else if (ompi_rank != NULL)
        return atoi(ompi_rank);
    else
        return -1;
}

