#!/bin/bash

# Print PAGESIZE
grep CONFIG_ARM64_64K_PAGES /boot/config-$(uname -r)
pagesize=$(getconf PAGESIZE)
echo "PAGESIZE: $pagesize"
grep Hugepagesize: /proc/meminfo

# Print transparent_hugepage/defrag
defrag=$(cat /sys/kernel/mm/transparent_hugepage/defrag)
echo "transparent_hugepage/defrag: $defrag"

# Print transparent_hugepage/enabled
enabled=$(cat /sys/kernel/mm/transparent_hugepage/enabled)
echo "transparent_hugepage/enabled: $enabled"

# Print numa_balancing
numa_balancing=$(cat /proc/sys/kernel/numa_balancing)
echo "numa_balancing: $numa_balancing"




# sysctl vm.defrag, sysctl vm.min_free_kbytes, and sysctl vm.nr_hugepages
# numastat, numatop
