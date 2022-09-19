#include <iostream>

#include "tensorflow/lite/energy_profiler.h"


int main() {
  using namespace std::chrono_literals;
  jianyu::EnergyProfiler profiler(100);
  profiler.Start();
  std::this_thread::sleep_for(10s);
  profiler.Pause();
  std::cout << profiler.GetAvgPower();
  profiler.Stop();
}