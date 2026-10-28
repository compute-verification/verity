"""The registered weights under rows-sha512: each tensor's rows are hiding leaves in a SHA-512 tree of its own (`step_rows.tensor_root`)
and the weights root is SHA-512 (`step_rows.weights_tree`), hashed on the host or on the tensor's device (native_rows.cu's row hashes,
frame leaves and tree, in the collector's extension, `native_ext`).  `NativeHostCommitter.register_weights` hashes through it."""
from __future__ import annotations

import os
from typing import Any

from verity_vllm.commit import scheme, step_rows


def key(tree: dict | None) -> bytes:
    """The registration's salt key: the one a committer registered under before (a re-registration reproduces its root:
    `c2_replay.live_weights_check`), else fresh from the OS generator."""
    tree = tree or {}
    return bytes.fromhex(tree["key"]) if tree.get("scheme") == step_rows.NAME else os.urandom(step_rows.KEY_BYTES)


class WeightsRows:
    """One registration's rows-sha512 roots under geometry digest ``geo`` and salt key ``wk``."""

    def __init__(self, geo: bytes, wk: bytes):
        self.geo, self.wk, self._dev = geo, wk, None

    def host_root(self, t: int, entry: list, flat: Any) -> bytes:
        return step_rows.tensor_root(self.geo, t, entry, memoryview(flat.cpu().numpy()).cast("B"), self.wk)

    def device_root(self, t: int, entry: list, flat: Any) -> bytes:
        """`host_root` of a CUDA tensor's bytes, on its device."""
        import torch

        from verity.ml.commitments.merkle import merkle_depth
        from verity_vllm.commit.committer.native_ext import ext

        e, RB = ext(), step_rows.ROW_BYTES
        if self._dev is None:
            self._dev = (e.StepHasher(), torch.frombuffer(bytearray(step_rows.t_nibbles()), dtype=torch.uint8).to(flat.device))
        H, tn = self._dev
        dom = step_rows.weights_domain(self.geo, t, entry)
        rs = e.RowStep(dom.domain_id, step_rows.tensor_key(self.wk, t), step_rows.row_prefix(8 * RB), step_rows.INST.salt_prefix,
                       step_rows.INST.leaf_prefix(step_rows.KEY), step_rows.SCHEMA, RB)
        n, nb = dom.count, int(flat.numel())
        full, stream = nb // RB, torch.cuda.current_stream(flat.device).cuda_stream
        src = flat if flat.data_ptr() % 16 == 0 else flat.clone()
        hs = [H.rows_h(rs, src, full * RB, 0, tn, stream)[0]] if full else []
        if full < n:
            tail = torch.zeros(RB, dtype=torch.uint8, device=flat.device)
            tail[:nb - full * RB].copy_(src[full * RB:])
            hs.append(H.rows_h(rs, tail, RB, full, tn, stream)[0])
        leaves = H.rows_frame(rs, torch.cat(hs) if len(hs) > 1 else hs[0], 0, 1 << merkle_depth(n), n, stream)
        return bytes(H.rows_tree(rs, leaves, 0, stream)[1])

    def root(self, t: int, entry: list, flat: Any, mode: str) -> tuple[bytes, int]:
        """Tensor ``t``'s root under `VERITY_WEIGHTS_HASH` ``mode``, and how many device roots it checked against the host's (under
        device-checked; a mismatch is refused)."""
        if mode == "host" or not flat.is_cuda:
            return self.host_root(t, entry, flat), 0
        r = self.device_root(t, entry, flat)
        if mode != "device-checked":
            return r, 0
        rh = self.host_root(t, entry, flat)
        if rh != r:
            raise RuntimeError(f"weights registration: device root != host root for {entry[0]} ({entry[3]} B, {step_rows.NAME}): "
                               f"{r.hex()[:16]} vs {rh.hex()[:16]} -- refusing to commit (device-checked)")
        return r, 1

    def tree(self, names: list, roots: list[bytes]) -> dict:
        return step_rows.weights_tree(self.geo, names, scheme.names_digest(names), roots, self.wk)
