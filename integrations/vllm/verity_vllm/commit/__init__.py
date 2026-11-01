"""verity_vllm.commit: the Commit, which commits a served run's values while it serves.

`committer` is the GPU committer (`native_collect.NativeCollectCommitter`, whose serving-thread work per acquired tensor is one C++
call, the host committer, leaf hashing and their CUDA/C++ sources). `step_rows` is the ``rows-sha512`` scheme every Commit commits
under by default, `scheme` the ``vllm-v1`` one. Around them: the capture map (`capture_map`), the binding map and coverage check
(`binding`), the partition of record (`partition`), canonical identities (`identity`), the challenge's samples (`challenge`), opened
values (`opened`), the executed prefix (`executed_prefix`), padding steps (`padding_steps`), the memory map (`memory_map`), the
served record (`served_record`) and the row tables (`row_tables`, `row_tables_device`).
"""
