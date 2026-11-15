"""`served_zk/statement.py`'s Statement (PoUW's prover, `kernels/served_zk/`) with its torch parts: the device's tensors
and the gate, `check`.  `pearl_c_vllm/e2e.py`'s ``pearlc-zk`` mode loads it."""
from __future__ import annotations

from typing import Any

import torch

from verity.protocols.resource_accounting.compute.pouw.prover.kernels.served_zk import statement as _statement


class Statement(_statement.Statement):
    def zeros(self, n: int) -> Any:
        return torch.zeros(n, dtype=torch.uint8, device=self.run.device)

    def empty(self, rows: int, k: int) -> Any:
        return torch.empty((rows, k), dtype=torch.float32, device=self.run.device)

    def fill_stages(self, byte: int) -> None:
        with torch.inference_mode():
            for st in self.stages.values():
                st.view(torch.uint8).fill_(byte)
        torch.cuda.synchronize(self.run.device)

    def check(self, weights: dict, m: int = 32) -> dict:
        """The gate (`served_zk/statement.py`'s docstring), with hashing and graphs on for its length."""
        run = self.run
        dev = run.device
        gen = torch.Generator(device=dev).manual_seed(20261006)
        xs = {w: torch.randn(m, W.shape[1], device=dev, generator=gen).to(torch.bfloat16) for w, W in weights.items()}
        forward = lambda: [run.gemm(w, xs[w], W, None) for w, W in weights.items()]  # noqa: E731
        held = run.graphs, run.hashing, self.enabled
        run.graphs, run.hashing, self.enabled = True, True, True
        res: dict = {}
        try:
            run.begin()
            forward()
            run.end()
            run.begin()
            side, graph = torch.cuda.Stream(device=dev), torch.cuda.CUDAGraph()
            run.whole_capture_begin(run.whole_key())
            try:
                forward()
                side.wait_stream(torch.cuda.current_stream(dev))
                with torch.cuda.graph(graph, stream=side):
                    forward()
            finally:
                calls = run.whole_capture_end()
            run.end()
            for name, skip in (("honest", False), ("control", True)):
                run.begin()
                run.poison()
                self.poison()
                at = run.cur.tiles
                run.whole_replay(calls)
                graph.replay()
                self.skip = skip
                try:
                    run.whole_replayed(calls)
                finally:
                    self.skip = False
                torch.cuda.synchronize(dev)
                cms = self._commits(m, sum(c.tiles for c in calls))
                cms[-1].data = run.leaves.data_ptr() + run.nb * at
                same = []
                for cm in cms:
                    want = cm.reference(cm.read_rows())
                    same.append(self.h.read(cm.tree + cm.tree_bytes - 64, 64) == want["root"]
                                and self.h.read(cm.bc, 128 * cm.rows) == want["bc"])
                staged = all(torch.equal(self.stages[(k, m)][i * m:(i + 1) * m], xs[w].float()) for w, (k, i) in self.pos.items())
                res[name] = {"roots_equal": same, "staged_rows_equal": staged}
                run.end()
        finally:
            run.graphs, run.hashing, self.enabled = held
            self.forwards = self.taken = 0
        rows = {f"x/k{k}": self.count[k] * m for k in self.ks}
        rows["tiles"] = sum(c.tiles for c in calls)
        ok = (all(res["honest"]["roots_equal"]) and res["honest"]["staged_rows_equal"] and res["control"]["staged_rows_equal"]
              and not any(res["control"]["roots_equal"]))
        return {"check": "statement", "cubin_sha256": self.cubin_sha256, "calls": len(calls), "rows": rows, **res, "ok": ok}


__all__ = ["Statement"]
