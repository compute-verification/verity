# Pearl-C's shared hashing

Every Pearl-C kernel commits what it computes with SHA-512, in the hashing format -h3. This directory holds that device
hashing, which the kernel directories include, and what lets their tests run on a CPU: host builds of the hashing and a
stand-in for the CUDA driver. It measures nothing, has no runner, and touches no GPU.

## Try it

~~~sh
uv run pytest tests/benchmarks/pouw/test_pearl_c_hash_host.py -q
~~~

This builds the hashing for the host with g++ and holds it to the scheme's bytes. It prints `48 passed`.

A change to `sha512.cuh` or `hash_h3.cuh` changes the build of every kernel that includes it, so it changes only with
those kernels' gates.
