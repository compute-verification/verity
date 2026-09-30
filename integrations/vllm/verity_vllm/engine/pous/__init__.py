"""POUS (P3) serving: vLLM runs from the P3 encoding `C` of the checkpoint, decoded on the GPU at every use, and answers audit
challenges from `C` with a native responder in the serving process.

`layout` (W from the checkpoint file, stdlib), `codec` (P3 on the GPU), `native` (the JIT build of `pous_native.cu`), `quant` (the
`quantization="pous"` config), `worker` (the challenger worker extension).
"""
