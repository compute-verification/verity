"""With SPOT_TP=N (N > 1), `vllm.EngineArgs` is wrapped with tensor_parallel_size=N once vllm imports, as the replay ladder's
tp_engine does around serve_tax's replay engine, which passes no tensor-parallel width itself."""
import functools
import importlib.abc
import importlib.util
import os
import sys

TP = int(os.environ.get("SPOT_TP") or 0)


class _Wrap(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "vllm":
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        run = spec.loader.exec_module

        def exec_module(m):
            run(m)
            m.EngineArgs = functools.partial(m.EngineArgs, tensor_parallel_size=TP)

        spec.loader.exec_module = exec_module
        return spec


if TP > 1:
    sys.meta_path.insert(0, _Wrap())
