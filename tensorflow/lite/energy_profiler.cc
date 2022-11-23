#include <string>
#include <fstream>

#include "tensorflow/lite/energy_profiler.h"


namespace jianyu {

namespace {

const std::string USB_CURRENT = "/sys/class/power_supply/usb/current_now";
const std::string USB_CURRENT_FALLBACK = "/sys/class/power_supply/usb/input_current_now";
const std::string USB_VOLTAGE = "/sys/class/power_supply/usb/voltage_now";
const std::string BAT_CURRENT = "/sys/class/power_supply/battery/current_now";
const std::string BAT_VOLTAGE = "/sys/class/power_supply/battery/voltage_now";

int ReadFromFile(std::string filename) {
  std::ifstream fs(filename);
  std::string content;
  fs >> content;
  if (content.empty()) {
    return ReadFromFile(USB_CURRENT_FALLBACK);
  }
  return std::stoi(content);
}

double ReadPower() {
  double usb_current = ReadFromFile(USB_CURRENT);
  double usb_voltage = ReadFromFile(USB_VOLTAGE);
  double bat_current = ReadFromFile(BAT_CURRENT);
  double bat_voltage = ReadFromFile(BAT_VOLTAGE);

  return usb_current / 1000000. * usb_voltage / 1000000.
       + bat_current / 1000000. * bat_voltage / 1000000.;
}

}

EnergyProfiler::EnergyProfiler(int interval):
    thread_(1, 0),
    count_(0),
    total_power_(.0),
    moving_power_(.0),
    profiling_(false),
    finished_(false),
    interval_(interval) {
  thread_.push([this](int){
    while (true) {
      if (finished_) {
        return;
      }
      if (profiling_) {
        count_ += 1;
        double power = ReadPower();
        total_power_ += power;
        moving_power_ = count_ == 0 ? power : (moving_power_ + power) / 2;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(interval_));
    }
  });
}

EnergyProfiler::~EnergyProfiler() {
  Stop();
}

void EnergyProfiler::Start() {
  profiling_ = true;
}

void EnergyProfiler::Resume() {
  profiling_ = true;
}

void EnergyProfiler::Pause() {
  profiling_ = false;
}

void EnergyProfiler::Stop() {
  finished_ = true;
  thread_.stop();
}

double EnergyProfiler::GetAvgPower() {
  return total_power_ / (count_ == 0 ? 1 : count_);
}

double EnergyProfiler::GetMovingPower() {
  return moving_power_;
}

}