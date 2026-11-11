"""The collector's binding of a Program's `run` values (`query.manifest.format.RUN_FAMILY`) that are a DSA row's top-k order past
INDEX_TOPK: the Waksman settings that route the row's N = P + 1 keys to the order the layer's indexer wrote, one byte (0 or 1) a
leaf, read by `DsaWaksmanOrder_v1{N,TOPK}` (`dsa_topk.waksman_settings`, the settings the Program's `run` field holds).

What is committed is what the rank's manifest requires, under the manifest's own names: its `run_values` rows give each step's
fields by the module their reader is lowered from (`attention_module`, a layer of the attention source) and that reader's N and TOPK
(`statics`), so a step with none (every prompt within INDEX_TOPK keys) commits nothing and keeps its plan key.  At a layer's attend
the rows of `topk_indices_buffer` at a position P whose field the step requires there are copied on the device (the buffer is every
layer's); at the step's end each is read once and turned into its settings, and a row whose order is no Waksman order of its P + 1
keys is a collector error by name, its field then missing from coverage.  Two requests' rows at one position of one layer are
committed in row order under one name: `bind.Binder.owners` gives occurrence k to the k-th of them.

The buffer's indices are request-local, the keys' positions 0..P in their request, on prefill and decode alike: the pinned vLLM's
top-k kernels number a prefill row's keys from its request's first gathered key and a decode row's by the paged logits' logical
columns, and the sparse backend's mapping to KV-cache slots (`triton_convert_req_index_to_global_index`) writes a new tensor, so
they are read as written."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import torch

from verity.definitions.registry.dsa_topk import waksman_settings
from verity_vllm.query.manifest.format import RUN_FAMILY


@dataclass
class Field:
    name: str
    keys: int
    topk: int
    leaves: int
    requests: int = 0


class RunValues:
    def __init__(self, identities: Iterable[Mapping[str, Any]], layers: Mapping[str, int]) -> None:
        """`identities`: the rank's required-manifest rows; `layers`: each attention layer's module path -> its layer, which a row's
        `attention_module` is or is under.  A row this rank can't commit is refused by name."""
        #: (engine step, layer) -> {keys N: the field of the row at position N - 1}
        self.at: dict[tuple[int, int], dict[int, Field]] = {}
        for r in identities:
            if r.get("family") != RUN_FAMILY:
                continue
            name, st, am = f"{r['op_path']}/{r['output_member']}", r.get("statics"), str(r.get("attention_module"))
            layer = next((i for m, i in layers.items() if am == m or am.startswith(m + ".")), None)
            if layer is None or not isinstance(st, Mapping) or not isinstance(st.get("N"), int) or not isinstance(st.get("TOPK"), int):
                raise ValueError(f"run values: {name} is read in module {r.get('attention_module')!r} with statics {st}: no attention "
                                 f"layer of this rank's source, or no N and TOPK to find its top-k row by")
            at = self.at.setdefault((int(r.get("engine_step", r.get("step"))), layer), {})
            f = at.setdefault(st["N"], Field(name, st["N"], st["TOPK"], int(r["element_range"][1])))
            if f.name != name:
                raise ValueError(f"run values: {name} and {f.name} are both layer {layer}'s row of {f.keys} keys at one step")
            f.requests += 1
        self.names = frozenset(f.name for fs in self.at.values() for f in fs.values())
        self._by_step: dict[int, Counter] = {}
        for (step, _layer), fs in self.at.items():
            self._by_step.setdefault(step, Counter()).update({f.name: f.requests for f in fs.values()})
        self._held: list[tuple[list[Field], Any]] = []
        self._positions: tuple[Any, list[int]] | None = None

    def targets_at(self, engine_step: int) -> dict[str, int]:
        """The fields committed at `engine_step`, each once per request that requires it."""
        return dict(self._by_step.get(int(engine_step), {}))

    def begin_step(self) -> None:
        self._held, self._positions = [], None

    def take(self, engine_step: int, layer: int, positions: Any, buf: Any, rows: int) -> None:
        """At layer `layer`'s attend: copy the rows of `buf` (the indexer's top-k indices) among the step's first `rows` whose field
        the step requires there (`positions`: each row's position)."""
        want = self.at.get((int(engine_step), int(layer)))
        if not want or positions is None or buf is None:
            return
        if self._positions is None or self._positions[0] is not positions:
            self._positions = (positions, [int(p) for p in positions[:rows].tolist()])
        picks = [(i, want[p + 1]) for i, p in enumerate(self._positions[1]) if p + 1 in want]
        if picks:
            topk = picks[0][1].topk
            self._held.append(([f for _i, f in picks], buf[[i for i, _f in picks], :topk]))

    def commit(self, acquire: Callable[[str, Any], None]) -> list[str]:
        """Each held row's settings, staged through `acquire(name, uint8 tensor on the buffer's device)` in row order.  -> errors."""
        errors: list[str] = []
        for fields, t in self._held:
            for f, order in zip(fields, t.tolist()):
                try:
                    if len(order) != f.topk:
                        raise ValueError(f"{len(order)} indices, not TOPK {f.topk}")
                    bits = waksman_settings(order, f.keys)
                    if len(bits) != f.leaves:
                        raise ValueError(f"{len(bits)} settings for the manifest's {f.leaves} leaves")
                except ValueError as e:
                    errors.append(f"run values: {f.name}: {e}"[:300])
                    continue
                acquire(f.name, torch.tensor(bits, dtype=torch.uint8, device=t.device))
        self._held = []
        return errors


__all__ = ["Field", "RunValues"]
