#ifndef JIANYU_ENERGY_PROFILER_H_
#define JIANYU_ENERGY_PROFILER_H_

#include "tensorflow/lite/ctpl_stl.h"

namespace jianyu {

class EnergyProfiler {
public:
  EnergyProfiler(int interval);

  ~EnergyProfiler();

  void Start();

  void Resume();

  void Pause();

  void Stop();

  double GetAvgPower();

  double GetMovingPower();

private:
  ctpl::thread_pool thread_;

  int count_;

  double total_power_;

  double moving_power_;

  bool profiling_;

  int interval_;

  bool finished_;
};

}

#endif
