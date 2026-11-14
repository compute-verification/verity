"""The capture of an engine run: an observer of the pinned eager vLLM whose only product is a raw observation log, and
that log's readers.  `fold/` turns a log back into an unexpanded Program, with no torch:

    eager vLLM -> raw observations -> resolver + memory fold -> specializations + instances -> Program

Beside the engine (these need torch, Triton or vLLM, some only inside a function):

    observer        the dispatch mode that records every dispatcher call as facts
    triton_adapter  direct Triton launches, observed at the JIT function's `run`
    storage         stable integer ids for tensor storages, never raw pointers in the log
    pinned          pinned host memory for the capture's asynchronous copies off the device
    capture         the model runner observed step by step into the log, and the run header the log opens with
    runtime_tree    the runtime's module tree, as a command (its record is `verity_vllm.records.runtime_tree`)

Anywhere (no torch):

    events          the event vocabulary, schema `verity-capture/log/v1`: the contract between the two sides
    log             the log's writer and reader, one JSON object a line
    capture_log     the log as the Match reads it: its header, each step's rows and each prompt parameter's request
    capture_types   what a capture is given (its configuration) and what it holds of one step until it writes the step
    kernel_trace    the device kernels of a profiler's Chrome trace

The runtime artifact is the *raw observation log*, never a "trace": that word is kept for a circuit's gate values.
"""

from verity_vllm.records.schema_ids import CAPTURE_LOG_SCHEMA

__all__ = ["SCHEMA"]

SCHEMA = CAPTURE_LOG_SCHEMA
