#include <sys/syscall.h>
#include <stdlib.h>
#include <unistd.h>
#include <cstring>

#include "sxr_thread.h"


namespace sxr {

namespace {

#ifdef __LP64__
#define CPU_SETSIZE 1024
#else
#define CPU_SETSIZE 32
#endif
#define __CPU_BITTYPE    unsigned long int  /* mandated by the kernel  */
#define __CPU_BITSHIFT   5                  /* should be log2(BITTYPE) */
#define __CPU_BITS       (1 << __CPU_BITSHIFT)
#define __CPU_ELT(x)     ((x) >> __CPU_BITSHIFT)
#define __CPU_MASK(x)    ((__CPU_BITTYPE)1 << ((x) & (__CPU_BITS-1)))
typedef struct {
    __CPU_BITTYPE  __bits[ CPU_SETSIZE / __CPU_BITS ];
} cpu_set_t;

#define CPU_ZERO(set)          CPU_ZERO_S(sizeof(cpu_set_t), set)
#define CPU_SET(cpu, set)      CPU_SET_S(cpu, sizeof(cpu_set_t), set)


#define CPU_ZERO_S(setsize, set)  memset(set, 0, setsize)
#define CPU_SET_S(cpu, setsize, set) \
  do { \
    size_t __cpu = (cpu); \
    if (__cpu < 8 * (setsize)) \
      (set)->__bits[__CPU_ELT(__cpu)] |= __CPU_MASK(__cpu); \
  } while (0)

} // namespace

void setCurrentThreadAffinityMask(int cpu1, int cpu2, int cpu3)
{
    cpu_set_t _mask;
    memset(&_mask, 0, sizeof(cpu_set_t));

    CPU_SET(cpu1, &_mask);
    if (cpu2)
        CPU_SET(cpu2, &_mask);
    if (cpu3)
        CPU_SET(cpu3, &_mask);
    const int tid = gettid();

    // Android
    const int result = syscall(__NR_sched_setaffinity,
                               tid,
                               sizeof(_mask),
                               &_mask);
} 

} // namespace sxr
