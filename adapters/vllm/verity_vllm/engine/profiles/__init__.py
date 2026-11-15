"""The model facts: the architecture families' declared facts (`family_facts`), the checkpoints' HF configs (`dense_generic`,
`hf_configs/`), the devices, the transport, the cases, the admission calibration and the data files beside them. They import
nothing from `observe/`, so `engine/`, `program/`, `check/` and `commit/` read them; what builds a fold `Profile` from them is
`observe.fold.profiles`."""
