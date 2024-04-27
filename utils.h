

#include <time.h>
#include <sys/time.h>
#include <numaif.h>
#include <numa.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdio.h>
#define NUMA_HBM 1


/* Timer Tools */

// timer at us accuracy
double mysecond();
double mysecond_();

// timer at ns accuracy
double mysecond2();
double mysecond2_();



/*  NUMA TOOLS */
int which_numa(double *var);

/** 
 * @brief move memory page
 *
 * @param  *ptr Pointer to the array to be moved
 * @param  size Size in bytes to be moved
 * @param  target_node The index of target NUMA node
 *
 * @return void
*/
void move_numa(double *ptr, unsigned long size, int target_node);


int check_MPI();
