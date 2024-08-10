#!/usr/bin/env python

import re
import sys

header = """
#define func_group "BLAS"
#include "global.h"
#include "complex.h"
#include <math.h>
#include <stdlib.h>
#include <init.h>
#include "utils.h"

"""

fcontent1 = """    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
"""

fcontent2 = """    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

"""


def generate_wrapper(signature, ptype):
    # Extract return type, function name, and arguments
    match = re.match(r'(\w+)\s+(\w+)\((.*)\)', signature)
    if not match:
        return "Invalid function signature"
    
    return_type, func_name, args = match.groups()
    func_name = func_name.rstrip('_')
 
    if ptype == 'dl':
          my_func_name = func_name + '_'
    else: 
          my_func_name = 'my_' + func_name
    
    # Generate argument names
    arg_names = [arg.split()[-1].strip('*') for arg in args.split(', ')]
    
    # Generate wrapper code
    wrapper = f"""{return_type} {my_func_name}({args})
{{
    {return_type} (*orig_f)() = NULL;
    """
    wrapper += f"enum findex fi = {func_name};\n" 
    #wrapper += func_name
    wrapper += fcontent1

    if return_type == "void":
        wrapper += f"    orig_f({', '.join(arg_names)});\n"
    else:
        wrapper += f"    {return_type} result = orig_f({', '.join(arg_names)});\n"

    wrapper += fcontent2

    if return_type != "void":
        wrapper += "    return result;\n"
    else: 
        wrapper += "    return;\n"

    wrapper += "}"

    return wrapper

if __name__ == '__main__':

  ptype = 'dbi'
  if len(sys.argv) >1: 
     arg = sys.argv[1].lower()
     if arg == 'dl': 
           ptype = 'dl'

  fname = "PROTOTYPES"
  inputfile = open(fname, 'r')

  print(header)
  for line in  inputfile:
    line = line.strip()
    if line and not line.startswith('#'):
       print(generate_wrapper(line, ptype))
       print()




