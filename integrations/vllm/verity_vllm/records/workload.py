"""The workload (`verity-vllm/workload/v2`): a row's requests, its sampling declaration and its pinned engine, as
`workloads/<row>.json` holds it and a run reads it; and the v1 layout (no `engine` block, the flat keys it replaced), which a
pod's older run directories still hold.

Its writers are `config.RowConfig.to_json` (the row files), `pipeline.workload.workload_doc`, `pipeline.coverage_workloads`,
`observe.prefix_cache` and `properties.holdout`; each writes every field typed here as required.  `config.RowConfig` parses a
row file for running it (v2 only, with the engine's checks); this is the record a check reads.

    w = Workload.load(path)                     # MissingField / UnknownField name the file and the field
    w.requests[0].prompt_length, w.sampling_variant, w.arrive_step(0), w.doc_sha256
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity_vllm.records.fields import RecordError, objects, only, optional, required

SCHEMA = "verity-vllm/workload/v2"
#: the layout before the `engine` block: `engine_args_required`, `env_required`, `target`, `tokenizer_revision` and
#: `checkpoint_revision` at the top level (`config._V1_KEYS`)
V1 = "verity-vllm/workload/v1"
SCHEMAS = (SCHEMA, V1)

#: what the readers don't interpret: provenance, the generator's parameters and notes, kept for `dump` and the document digest
INFO = frozenset({"note", "seed", "seeds", "serving_profile", "context_class", "length_seed", "prompt_seed", "replicate", "feature",
                  "feature_variant", "dtype", "numerical_format", "length_rule", "expected_engine_steps", "schedule_if_full_length",
                  "tokenizer_note", "prompt_lengths", "max_tokens", "protocols_config", "request_origins", "generated_utc", "sha256"})
V1_INFO = frozenset({"engine_args_required", "env_required", "target", "tokenizer_revision", "checkpoint_revision"})
REQUEST_INFO = frozenset({"origin", "prompt_text", "arrival_offset_s", "text_roundtrip", "shared_prefix_tokens", "prefix_role",
                          "shared_prefix_len", "accepted_from", "corpus_seed"})
_NUMBER = (int, float)


def _ints(v: Sequence[Any] | None, where: str, name: str) -> tuple[int, ...] | None:
    if v is None:
        return None
    if any(isinstance(t, bool) or not isinstance(t, int) for t in v):
        raise RecordError(where, name, "holds a value that is not an integer")
    return tuple(v)


def _put(out: dict[str, Any], **kv: Any) -> dict[str, Any]:
    out.update({k: v for k, v in kv.items() if v is not None})
    return out


@dataclass(frozen=True)
class WorkloadRequest:
    request_id: str                                   # the workload's id of the request (a UUID); the Program's is r<index>
    max_tokens: int                                   # the public cap
    index: int | None                                 # its position: `properties.holdout` writes none
    prompt_len: int | None                            # `properties.holdout` writes none: the length of `prompt_token_ids`
    # a lengths-only plan (`pipeline.workload` without the model's tokenizer) has no prompt
    prompt_token_ids: tuple[int, ...] | None
    arrive_step: int | None                           # absent: the sweep's `arrive_steps`, else 0
    seed: int | None                                  # absent: the workload's `sampling.seed`
    variant: str | None                               # a per-request pin into `variants` (vllm_adapter's [ov-sampling])
    n: int | None                                     # absent: the workload's `sampling.n`
    accepted_token_ids: tuple[int, ...] | None        # the accepted continuation of a speculative row; others have none
    info: Mapping[str, Any]                           # the `REQUEST_INFO` fields it carries

    @property
    def prompt_length(self) -> int | None:
        """`prompt_len`, else the length of the prompt; None for neither."""
        if self.prompt_len is not None:
            return self.prompt_len
        return None if self.prompt_token_ids is None else len(self.prompt_token_ids)

    @classmethod
    def of(cls, doc: Mapping[str, Any], where: str, name: str) -> WorkloadRequest:
        only(doc, {"request_id", "max_tokens", "index", "prompt_len", "prompt_token_ids", "arrive_step", "seed", "variant", "n",
                   "accepted_token_ids", *REQUEST_INFO}, where, f"{name}.")
        if "seed" in doc and doc["seed"] is None:
            # the driver serves an explicit null unseeded, and an absent seed with the workload's: no writer writes one
            raise RecordError(where, f"{name}.seed", "is null: a request's seed is an integer, or absent (the workload's sampling.seed)")
        return cls(request_id=required(doc, "request_id", where, str, f"{name}.request_id"),
                   max_tokens=required(doc, "max_tokens", where, int, f"{name}.max_tokens"),
                   index=optional(doc, "index", where, int, f"{name}.index"),
                   prompt_len=optional(doc, "prompt_len", where, int, f"{name}.prompt_len"),
                   prompt_token_ids=_ints(optional(doc, "prompt_token_ids", where, (list, tuple), f"{name}.prompt_token_ids"),
                                          where, f"{name}.prompt_token_ids"),
                   arrive_step=optional(doc, "arrive_step", where, int, f"{name}.arrive_step"),
                   seed=optional(doc, "seed", where, int, f"{name}.seed"),
                   variant=optional(doc, "variant", where, str, f"{name}.variant"),
                   n=optional(doc, "n", where, int, f"{name}.n"),
                   accepted_token_ids=_ints(optional(doc, "accepted_token_ids", where, (list, tuple), f"{name}.accepted_token_ids"),
                                            where, f"{name}.accepted_token_ids"),
                   info={k: doc[k] for k in REQUEST_INFO if k in doc})

    def dump(self) -> dict[str, Any]:
        return _put({"request_id": self.request_id}, index=self.index, prompt_len=self.prompt_len, max_tokens=self.max_tokens,
                    prompt_token_ids=None if self.prompt_token_ids is None else list(self.prompt_token_ids),
                    arrive_step=self.arrive_step, seed=self.seed, variant=self.variant, n=self.n,
                    accepted_token_ids=None if self.accepted_token_ids is None else list(self.accepted_token_ids)) | dict(self.info)


@dataclass(frozen=True)
class WorkloadSampling:
    variant: str                                      # a key of `variants` (`greedy`, `stoch-t<T>-p<P>[-k<K>]`)
    seed: int | None                                  # every writer writes it, null for greedy
    n: int | None                                     # absent: 1 (the driver's default)
    ignore_eos: bool | None                           # true: every request runs to its cap (`config.done_source`)
    # explicit parameters beside the variant's table: no writer writes them; the global match's G1 compares them when a hand edit adds them
    temperature: float | None
    top_p: float | None
    top_k: int | None
    # stops before the cap: no writer writes them; `config.done_source_refusal` refuses them on a workload of record
    stop_token_ids: Sequence[Any] | None
    stop: Sequence[Any] | str | None
    # the EOS set pinned into the workload: no writer writes it; the global Program's stop rule takes it as the declaration of record, and a
    # checkpoint given beside it must agree (`pipeline.global_program.stop_rule`)
    eos_token_ids: tuple[int, ...] | None = None

    @classmethod
    def of(cls, doc: Mapping[str, Any], where: str) -> WorkloadSampling:
        only(doc, set(cls.__dataclass_fields__), where, "sampling.")
        return cls(variant=required(doc, "variant", where, str, "sampling.variant"),
                   seed=optional(doc, "seed", where, int, "sampling.seed"),
                   n=optional(doc, "n", where, int, "sampling.n"),
                   ignore_eos=optional(doc, "ignore_eos", where, bool, "sampling.ignore_eos"),
                   temperature=optional(doc, "temperature", where, _NUMBER, "sampling.temperature"),
                   top_p=optional(doc, "top_p", where, _NUMBER, "sampling.top_p"),
                   top_k=optional(doc, "top_k", where, int, "sampling.top_k"),
                   stop_token_ids=optional(doc, "stop_token_ids", where, (list, tuple), "sampling.stop_token_ids"),
                   stop=optional(doc, "stop", where, (list, tuple, str), "sampling.stop"),
                   eos_token_ids=_ints(optional(doc, "eos_token_ids", where, (list, tuple), "sampling.eos_token_ids"),
                                       where, "sampling.eos_token_ids"))

    def dump(self) -> dict[str, Any]:
        return _put({"variant": self.variant, "seed": self.seed}, n=self.n, ignore_eos=self.ignore_eos, temperature=self.temperature,
                    top_p=self.top_p, top_k=self.top_k, stop_token_ids=self.stop_token_ids, stop=self.stop,
                    eos_token_ids=None if self.eos_token_ids is None else list(self.eos_token_ids))


@dataclass(frozen=True)
class Workload:
    where: str                                        # the file it was read from, or what handed it over: a refusal names it
    schema: str                                       # SCHEMA, or V1 for an older run's copy
    model: str                                        # the checkpoint's HF repo
    variants: Mapping[str, Mapping[str, Any]]         # variant -> its sampling parameters (temperature, top_p, top_k): the driver's table
    requests: tuple[WorkloadRequest, ...]
    case: str | None                                  # the checkpoint's role; `pipeline.workload` writes null for a row without one
    engine: Mapping[str, Any] | None                  # v2's pinned engine block (`config.RowEngine`); a v1 workload has the flat keys
    sampling: WorkloadSampling | None                 # `properties.holdout` writes none: the driver's greedy default
    sweep: Mapping[str, Any] | None                   # the sweep plan the requests were sampled from (`pipeline.workload`), else none
    info: Mapping[str, Any]                           # the `INFO` fields it carries (and `V1_INFO`, for a v1 workload)
    # the document's sha256 as the driver records it (RunHeader.workload.sha256: sort_keys, compact separators), of the JSON as read
    doc_sha256: str = field(default="", compare=False)

    @property
    def sampling_variant(self) -> str | None:
        return None if self.sampling is None else self.sampling.variant

    def arrive_step(self, i: int) -> int:
        """Request `i`'s arrival: its `arrive_step`, else the sweep's `arrive_steps[i]`, else 0."""
        own = self.requests[i].arrive_step
        if own is not None:
            return own
        steps = None if self.sweep is None else self.sweep.get("arrive_steps")
        return int(steps[i]) if steps is not None and i < len(steps) else 0

    @classmethod
    def of(cls, doc: Workload | Mapping[str, Any], where: str) -> Workload:
        """The record of a workload as its writer wrote it (`doc`, the parsed JSON); a record passes through."""
        if isinstance(doc, cls):
            return doc
        if not isinstance(doc, Mapping):
            raise RecordError(where, "schema", f"cannot be read: the workload is a {type(doc).__name__}, not a JSON object")
        schema = required(doc, "schema", where, str)
        if schema not in SCHEMAS:
            raise RecordError(where, "schema", f"is {schema!r}, not {SCHEMA!r} (or the older {V1!r})")
        info = INFO | (V1_INFO if schema == V1 else frozenset())
        only(doc, {"schema", "model", "variants", "requests", "case", "sampling", "sweep", *info, *(("engine",) if schema == SCHEMA else ())},
             where)
        variants = required(doc, "variants", where, Mapping)
        for k, v in variants.items():
            if not isinstance(v, Mapping):
                raise RecordError(where, f"variants.{k}", f"is a {type(v).__name__}, not a Mapping")
        sampling = optional(doc, "sampling", where, Mapping)
        return cls(where=where, schema=schema, model=required(doc, "model", where, str), variants=variants,
                   requests=tuple(WorkloadRequest.of(r, where, f"requests[{i}]") for i, r in enumerate(objects(doc, "requests", where))),
                   case=optional(doc, "case", where, str),
                   engine=required(doc, "engine", where, Mapping) if schema == SCHEMA else None,
                   sampling=None if sampling is None else WorkloadSampling.of(sampling, where),
                   sweep=optional(doc, "sweep", where, Mapping),
                   info={k: doc[k] for k in info if k in doc},
                   doc_sha256=hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest())

    @classmethod
    def load(cls, path: str | Path) -> Workload:
        with open(path) as f:
            return cls.of(json.load(f), str(path))

    def dump(self) -> dict[str, Any]:
        """The JSON its writers write: every field it holds, an absent optional one left out."""
        out = _put({"schema": self.schema, "model": self.model}, case=self.case, engine=self.engine)
        out |= dict(self.info) | {"variants": self.variants}
        return _put(out, sampling=None if self.sampling is None else self.sampling.dump(), sweep=self.sweep) | {
            "requests": [r.dump() for r in self.requests]}
