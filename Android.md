- IO stream dramatically degrade performance
- CPU affinity influence GPU delegate a lot, Hexagon delegate a little.
- Threadpool for hexagon. (or even gpu?)
- GPU invoke:
	- gpu_api_delegate: kRegistration.invoke
	- gpu_api_delegate: Delegate.invoke
	- api.cc: InferenceRunnerImpl.run
- Overheating...
- taskset f0 ./benchmark_model_jianyu --graph=R-50-mbtest.quant.tflite --use_gpu=true --num_threads=4 --use_xnnpack=false --enable_op_profiling=true --max_delegated_partitions=100 --warmup_min_secs=0 --min_secs=0 --warmup_runs=5 --num_runs=5
- TwoStepTensorTie
	- defaulttensortie
	- defaultensortie
- BHWCBuffertToTensorConverter
- TensorToBHWCBufferConverter
- CpuCopier
	- BufferInput
	- BufferOutput
- Try reduce clFlush

- for mp_start nodes:
	- output set to gpu buffer and map for write before execution
	- unmap output after execution
- for seq cpu:
	- intput set to gpu buffer and map for read before execution
	- unmap input after execution
- for mp_end nodes:
	- input set to gpu buffer and map for read before execution
	- unmap input after execution


- Edge start:
	- Map mp_start output write

- Unmap mp_start input (if mapped)
- Unmap mp_start output
- Unmap last seq_cpu input (if mapped)
- Enqueue transform_cpu_gpu
- Enqueue seq_gpu compute
- Enqueue transform_gpu_cpu
- Map mp_end input read (event mp_end_input_map)
- Map mp_end output write (event mp_end_output_map)
- Exec seq_cpu
- Wait event mp_end_output_map
- Exec mp_end

- Edge end:
	- Unmap mp_start input (if mapped)
	- Unmap mp_start output
	- Unmap last seq_cpu input (if mapped)


- Use host_ptr while allocating memory
- Use invalid_region


