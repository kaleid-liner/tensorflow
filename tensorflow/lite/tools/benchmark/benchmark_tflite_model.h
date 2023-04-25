/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_TFLITE_MODEL_H_
#define TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_TFLITE_MODEL_H_

#include <algorithm>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <sys/types.h>
#include <chrono>

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/profiling/profiler.h"
#include "tensorflow/lite/tools/benchmark/benchmark_model.h"
#include "tensorflow/lite/tools/utils.h"


namespace tflite {
namespace benchmark {

// Benchmarks a TFLite model by running tflite interpreter.
class BenchmarkTfLiteModel : public BenchmarkModel {
 public:
  struct InputLayerInfo {
    InputLayerInfo() : has_value_range(false), low(0), high(0) {}

    std::string name;
    std::vector<int> shape;

    // The input value is randomly generated when benchmarking the NN model.
    // However, the NN model might require the value be limited to a certain
    // range [low, high] for this particular input layer. For simplicity,
    // support integer value first.
    bool has_value_range;
    int low;
    int high;

    // The input value will be loaded from 'input_file_path' INSTEAD OF being
    // randomly generated. Note the input file will be opened in binary mode.
    std::string input_file_path;
  };

  explicit BenchmarkTfLiteModel(BenchmarkParams params = DefaultParams());
  ~BenchmarkTfLiteModel() override;

  std::vector<Flag> GetFlags() override;
  void LogParams() override;
  TfLiteStatus ValidateParams() override;
  uint64_t ComputeInputBytes() override;
  TfLiteStatus Init() override;
  TfLiteStatus RunImpl() override;
  static BenchmarkParams DefaultParams();

 protected:
  TfLiteStatus PrepareInputData() override;
  TfLiteStatus ResetInputsAndOutputs() override;

  int64_t MayGetModelFileSize() override;

  virtual TfLiteStatus LoadModel();

  // Allow subclasses to create a customized Op resolver during init.
  virtual std::unique_ptr<tflite::OpResolver> GetOpResolver() const;

  // Allow subclass to initialize a customized tflite interpereter.
  virtual TfLiteStatus InitInterpreter();

  // Create a BenchmarkListener that's specifically for TFLite profiling if
  // necessary.
  virtual std::unique_ptr<BenchmarkListener> MayCreateProfilingListener() const;

  void CleanUp();

  utils::InputTensorData LoadInputTensorData(
      const TfLiteTensor& t, const std::string& input_file_path);

  std::vector<InputLayerInfo> inputs_;
  std::vector<utils::InputTensorData> inputs_data_;
  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  std::unique_ptr<tflite::ExternalCpuBackendContext> external_context_;

  std::unique_ptr<tflite::FlatBufferModel> s_model_;
  std::unique_ptr<tflite::Interpreter> s_interpreter_;
  std::unique_ptr<tflite::ExternalCpuBackendContext> s_external_context_;

 private:
  utils::InputTensorData CreateRandomTensorData(
      const TfLiteTensor& t, const InputLayerInfo* layer_info);

  void AddOwnedListener(std::unique_ptr<BenchmarkListener> listener) {
    if (listener == nullptr) return;
    owned_listeners_.emplace_back(std::move(listener));
    AddListener(owned_listeners_.back().get());
  }

  void Switch(int device) {
    if (device != cur_device_) {
      std::swap(model_, s_model_);
      std::swap(interpreter_, s_interpreter_);
      std::swap(external_context_, s_external_context_);
      cur_device_ = device;
    }
  }

  std::string GetGraph() {
    if (cur_device_ == 0) {
      return params_.Get<std::string>("graph");
    } else if (cur_device_ == 1) {
      return params_.Get<std::string>("dg_graph");
    } else {
      return "";
    }
  }

  bool DGAvailable() {
    return !params_.Get<std::string>("dg_graph").empty();
  }

  std::vector<std::string> SplitWords(const std::string& sentence) {
    std::string word;
    std::vector<std::string> words;

    std::istringstream iss(sentence);
    while (std::getline(iss, word, ' ')) {
      if (!word.empty()) {
        words.push_back(word);
      }
    }

    return words;
  }

  using clock_t = std::chrono::high_resolution_clock;

  void ResetJiffies() {
    work_jiffies_ = 0;
    proc_total_jiffies_ = 0;
    proc_work_jiffies_ = 0;
    cpu_usage_ = 0.0;
    proc_usage_ = 0.0;
  }

  float GetCPUUsage() {
    const int INTERVAL = 100;
    std::ifstream ifs;

    auto now = clock_t::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_tp_).count() >= INTERVAL 
        || work_jiffies_ == 0) {
      // Get CPU usage of this process
      int pid = getpid();
      std::string proc_stat_file = "/proc/" + std::to_string(pid) + "/stat";
      ifs.open(proc_stat_file);
      std::string line;
      std::getline(ifs, line);
      ifs.close();
      std::vector<std::string> words = SplitWords(line);
      long long utime = std::stoll(words[13]);
      long long stime = std::stoll(words[14]);
      long long cutime = std::stoll(words[15]);
      long long cstime = std::stoll(words[16]);
      long long starttime = std::stoll(words[21]);

      // Get uptime
      ifs.open("/proc/uptime");
      double uptime;
      ifs >> uptime;
      ifs.close();

      const int CLK_TCK = 100;
      long long proc_work_jiffies = utime + stime + cutime + cstime;
      long long proc_total_jiffies = (long long)(uptime * CLK_TCK) - starttime;
      long long elapsed_jiffies = proc_total_jiffies - proc_total_jiffies_;
      elapsed_jiffies = elapsed_jiffies ? elapsed_jiffies : 1;
      double proc_usage = (double)(proc_work_jiffies - proc_work_jiffies_) / elapsed_jiffies;

      // Get total CPU usage
      ifs.open("/proc/stat");
      std::getline(ifs, line);
      words = SplitWords(line);
      long long user = std::stoll(words[1]);
      long long nice = std::stoll(words[2]);
      long long sys = std::stoll(words[3]);
      long long work_jiffies = user + nice + sys;
      double cpu_usage;
      cpu_usage = (double)(work_jiffies - work_jiffies_) / (elapsed_jiffies);
      ifs.close();

      work_jiffies_ = work_jiffies;
      proc_total_jiffies_ = proc_total_jiffies;
      proc_work_jiffies_ = proc_work_jiffies;
      cpu_usage_ = cpu_usage;
      proc_usage_ = proc_usage;
      last_tp_ = now;
    }

    return cpu_usage_ - proc_usage_;
  }

  std::vector<std::unique_ptr<BenchmarkListener>> owned_listeners_;
  std::mt19937 random_engine_;
  std::vector<Interpreter::TfLiteDelegatePtr> owned_delegates_;
  // Always TFLITE_LOG the benchmark result.
  BenchmarkLoggingListener log_output_;

  int cur_device_;
  long long work_jiffies_, proc_total_jiffies_, proc_work_jiffies_;
  double cpu_usage_, proc_usage_;
  clock_t::time_point last_tp_;
  double switch_threshold_;
};

}  // namespace benchmark
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_TFLITE_MODEL_H_
