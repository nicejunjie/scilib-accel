
/* Timer Tools */

// timer at us accuracy
double scilib_second();
double scilib_second_();

// timer at ns accuracy
double scilib_second2();
double scilib_second2_();



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

char** str_split(char* a_str, const char a_delim);
int in_str(const char* s1, char** s2);
