"""verity.ir — a small implementation of the Verity IR core (veritor ``docs/verity-ir-2026-09-07.md``, §B/§C govern).

Promoted from ``veritor`` ``verity.ir/`` at commit ``eb865ce29``: ``types``, ``refs``, ``defs``, ``program``, ``layout``,
``evaluate``, ``query`` byte for byte (``plan`` moved to ``verity.proofs.plan``); ``codec`` with its hard-coded registry families replaced by the
``register_lazy_family`` hook.  The registry (``b1``, ``hopper``, ``prims``, ...), frontend, and derive tooling stayed behind;
semantics packages register their primitives with this core.

Invariant (verbatim from IR §C.5; do not paraphrase, do not regress):

    Gate -> exactly one Value<w>.
    Composite interfaces -> ordered collections of value references.
    Commitment positions -> physical authentication locations for those values,
                            not semantic components of gate outputs.

Two namespaces, kept apart everywhere in this package and in tools built on it:

    logical value identity   = producing gate (root-scoped gate index; prescribed inputs
                               are `Input` gates and so have their own canonical references)
    authentication location  = commitment/domain + leaf/range/offset  (NOT modelled here;
                               the correspondence tool reports it in a separate field)

What is implemented: representation types (`Value<w>`, `Array<n,T>`, tuples, records), primitive
definitions with total evaluators on encodings, composite definitions whose bodies are explicit dataflow
over calls, `call` / `batch` / `scan` / `bind`, the canonical layout (function-body order, one interval per
call, batch members and scan iterations in structural order, cached gate counts and offsets — gate `i` is
located by division, never by expansion), root-scoped value references, regions, the query language
(`/`, `/Name`, `//Name`, `//A/B`, `//A//B`, `//A ! groups(F, k)`), an explicit-expansion evaluator, and a
JSON descriptor codec with one unambiguous (deterministic, not yet byte-canonical) encoding.
"""

from .types import Value, Array, Tuple, Record, Type, leaves_of  # noqa: F401
from .refs import Coll, array_of, tuple_of, record_of  # noqa: F401
from .defs import (PrimitiveDefinition, CompositeDefinition, SpecializedDefinition, IntrinsicSummary, bind, primitive, composite,  # noqa: F401
                   Registry, REGISTRY)
from .program import Program, input_prim  # noqa: F401
from .layout import Circuit, Subcircuit, Region, FunctionCall, Gate  # noqa: F401
from .query import query, groups, GapError, QueryError  # noqa: F401
from .evaluate import evaluate_region, evaluate_call, ExecutionTranscript, MissingValue  # noqa: F401
from .codec import encode_program, decode_program, program_digest, descriptor_size, canonical_json  # noqa: F401

INVARIANT = """Gate -> exactly one Value<w>.
Composite interfaces -> ordered collections of value references.
Commitment positions -> physical authentication locations for those values,
                        not semantic components of gate outputs."""
