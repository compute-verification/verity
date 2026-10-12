# Kimi-K3 prep without a GPU

Jobs that read what the pinned vLLM (`d9105ea80`, torch 2.13.0+cu129) runs for `moonshotai/Kimi-K3`'s text model
(`KimiLinearForCausalLM`) on sm_90, sm_100 and sm_103, before any rental.

- `toolkit.sh`: sourced; assembles CUDA 12.9.1 from NVIDIA's redistributable archives (sha256 from `redistrib_12.9.1.json`,
  whose own hash is pinned) and exports `CUDA_HOME`.
- `build_pin.sh`: builds vLLM at the pin from source for `K3_ARCHS` (default `9.0a;10.0a;10.3a`) on a CPU node, then
  `pin_report.py` reads the build and the pinned official cu129 wheel: each extension's cubin and PTX archs, the K3 kernels by
  arch, the registered ops, and whether `nvidia-cutlass-dsl[cu13]` imports and compiles under torch cu129. The launch command
  is in its header. `K3_CONFIGURE_ONLY=1` stops after CMake's configure.
