
/* Timer Tools */

// timer at us accuracy
double mysecond();
double mysecond_();

// timer at ns accuracy
double mysecond2();
double mysecond2_();



/*  NUMA TOOLS */
int which_numa(void *var, size_t bytes);
int which_numa2(void *var);

/** 
 * @brief move memory page
 *
 * @param  *ptr Pointer to the array to be moved
 * @param  size Size in bytes to be moved
 * @param  target_node The index of target NUMA node
 *
 * @return void
*/
void move_numa(void *ptr, unsigned long size, int target_node);


int check_MPI();
int get_MPI_local_rank();

void get_exe_path(char **path);
void get_argv0(char **argv0);


