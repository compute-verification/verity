"""R10 acq-native (owner 1): native acquisition at the existing hook boundaries for the complete CMT-3 commitment path.

`native_collect.NativeCollectCommitter` is a drop-in for CMT-3's `native_host_fast+hidden_m1_gputree` committer (same values, same
arena layout, same 256-B chunk tree, same FA2 M1 export) whose serving-thread work per acquired tensor is one C++ call.  The frozen R9
path (`verity_vllm/commit/committer/native_host.py`) is untouched.
"""
