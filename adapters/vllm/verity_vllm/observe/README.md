# verity_vllm.observe

The capture of an engine run: the observer that writes a raw observation log while the pinned vLLM runs, and the
readers of that log. The package's docstring lists the modules directly under it, which this file covers; `fold/` turns
a log back into a Program.

## Interface

- **Provides:** the log's event vocabulary (`events`), its writer and reader (`log`), the capture (`capture`, over
  `capture_types`, `observer`, `storage`, `pinned` and `triton_adapter`), the log as the Match reads it
  (`capture_log`), the device kernels of a profiler's trace (`kernel_trace`) and the runtime's module tree as a command
  (`runtime_tree`).
- **Loaded by:** the integration's `pipeline`, `check`, `program`, `properties` and `acquire` packages, `fold/`, and
  the benchmarks that read a log or a kernel trace.
- **Loads:** the integration's `records`, `engine` and `config`, and `program` inside one function; torch and numpy in
  the modules that run beside the engine (`capture`, `observer`, `pinned`, `storage`), and Triton inside one function.

What the integration's other packages and the benchmarks import, module by module. From `events`: the event types
`RunHeader`, `StepStart`, `StepEnd`, `StepLink`, `SampleEnd`, `RunEnd`, `DispatchEnter`, `DispatchExit`, `TritonLaunch`,
`StorageAlloc`, `StorageFree`, `Root`, `Note` and `Event`, with `ViewDesc` and `views_in`. From `log`: `read_log`,
`split_steps` and `header_of`. From `capture_log`: `read_steps` and `request_params`. From `kernel_trace`:
`kernels_of_trace_events`, `load_trace` and `summarize_kernels`. From `capture`: `Capture`, `ObserverConfig` and
`make_header`. From `runtime_tree`: `RuntimeTree`, `RUNNER_PATH` and `runtime_tree_of`, which it hands on from
`verity_vllm.records.runtime_tree`. Each of these modules lists its names in `__all__`.
