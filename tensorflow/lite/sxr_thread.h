#ifndef SXR_THREAD_H_
#define SXR_THREAD_H_

namespace sxr {

void setCurrentThreadAffinityMask(int cpu1, int cpu2 = 0, int cpu3 = 0);

}

#endif
