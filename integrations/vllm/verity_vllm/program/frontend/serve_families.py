"""Which serving program an instance pins (`registry.b1`): the `Serve` families by `program.identity.family`, the
attention geometry a semantic profile declares and the `BN` static it binds, and the bound program of an identity."""

from __future__ import annotations

from verity.ml.circuits.defs import CompositeDefinition, bind
from verity_vllm.program.registry import b1
from verity_vllm.program.registry.b1 import BN, Serve, ServeV4, fa2_kblock_n


#: [E7 D54] the pinned attention kernel source the geometry rule below is read from
FA2_KERNEL_SOURCE = ("vllm-project/flash-attention 506341a143fcabd4bb79052a7605ada727d6b3f5 csrc/flash_attn/src/flash_fwd_launch_template.h:172-192 "
                     "run_mha_fwd_splitkv_align (num_splits == 1; paged varlen KV forces the split-KV dispatch, flash_api.cpp:762-764,798)")


def fa2_kernel_geometry(attention_consts: dict, D: int) -> dict:
    """[E7 D54 2026-09-13] The block geometry of the attention kernel a SEMANTIC PROFILE declares, at head dim ``D``.

    ``attention_consts`` is the profile's attention node ``consts`` (`manifests/semantic-profiles/*.json`,
    `layer_nodes[op_type == "attention"].consts`: `backend`, `fa_version`, `num_splits`, `head_size`, ...) -- the pinned
    document (`protocol_instance.json#vocabulary.profile_sha256`) that names the kernel; this function is the dispatch rule of
    the pinned source (`FA2_KERNEL_SOURCE`) applied to it.  Returns ``{"kernel", "kBlockM", "kBlockN", "kNWarps", "visit",
    "num_splits", "source"}``; refuses (ValueError) any declaration it has no rule for (another backend, FA3, `num_splits != 1`:
    the non-aligned split path is a DIFFERENT kernel -- kBlockN 256 at head dim <= 64 -- whose numerics this registry has never
    measured; refusing is the honest answer, not the default constant)."""
    backend, fa, splits = attention_consts.get("backend"), attention_consts.get("fa_version"), attention_consts.get("num_splits")
    if backend != "FlashAttentionBackend" or fa != 2:
        raise ValueError(f"no attention geometry rule for backend={backend!r} fa_version={fa!r} (this registry models vllm-flash-attn FA2 only)")
    if splits != 1:
        raise ValueError(f"no attention geometry rule for num_splits={splits!r}: the num_splits != 1 split-KV path is a different kernel "
                         f"(kBlockN = Headdim <= 64 ? 256 : Headdim <= 128 ? 128 : 64) never measured here; the profiles pin num_splits = 1")
    D = int(D)
    return {"kernel": "flash_fwd_splitkv_kernel", "traits": f"Flash_fwd_kernel_traits<{D}, 64, {fa2_kblock_n(D)}, 4, false, false, bf16>",
            "kBlockM": 64, "kBlockN": fa2_kblock_n(D), "kNWarps": 4, "visit": "key blocks n_block_max-1 -> 0 (last -> first), one online-softmax rescale per block",
            "num_splits": 1, "source": FA2_KERNEL_SOURCE}


def profile_attention_consts(profile_doc: dict) -> dict:
    """The attention node's ``consts`` of a semantic-profile document (the first `layer_nodes` entry with op_type "attention")."""
    for n in profile_doc.get("layer_nodes", []):
        if n.get("op_type") == "attention":
            return dict(n.get("consts") or {})
    raise ValueError(f"profile {profile_doc.get('id')!r} declares no attention node")


#: The serving programs an instance may pin, by `program.identity.family`: `Serve_v1` -- the legacy program (kBlockN 64 as the
#: module constant; every record before D54; an identity that carries NO geometry means this); `Serve_v4` -- the same program with the
#: attention key-block size as a declared static, filled from `program.identity.attention.kBlockN`.
SERVE_FAMILIES: dict[str, CompositeDefinition] = {"Serve_v1": Serve, "Serve_v4": ServeV4}
DEFAULT_SERVE_FAMILY = "Serve_v1"
#: Kernel names an identity geometry may carry (all the same kernel: vllm-flash-attn FA2 `flash_fwd_splitkv_kernel`, `num_splits == 1`)
FA2_SPLITKV_KERNEL_NAMES = ("flash_fwd_splitkv_kernel", "fa2-splitkv-align", "fa2-splitkv")
LEGACY_ATTENTION_GEOMETRY: dict = {"kernel": "flash_fwd_splitkv_kernel", "kBlockM": 64, "kBlockN": BN, "num_splits": 1,
                                   "note": "Serve_v1's module constant (the head-dim-128 trait); implicit in every pre-D54 identity"}


def serve_family(family: str | None):
    """The `Serve` definition of a `program.identity.family` (None -> `DEFAULT_SERVE_FAMILY`); KeyError names the unknown family."""
    fam = family or DEFAULT_SERVE_FAMILY
    if fam not in SERVE_FAMILIES:
        raise KeyError(f"unknown serving program family {fam!r}; known: {sorted(SERVE_FAMILIES)}")
    return SERVE_FAMILIES[fam]


def attention_geometry_static(attention: dict) -> int:
    """[D57] The program static (``BN`` = kBlockN) of an identity's declared ``attention`` geometry
    (`{"kernel", "kBlockM", "kBlockN", "num_splits", ...}`, the shape `fa2_kernel_geometry` produces).  The row program models the FA2
    split-KV kernel's online softmax over kBlockN-key blocks visited last -> first with one rescale per block; that model needs
    `kernel` one of `FA2_SPLITKV_KERNEL_NAMES` (the `num_splits == 1` path), `kBlockM == 64`, `num_splits == 1`, and a kBlockN multiple of 16 --
    anything else is refused (ValueError), never defaulted."""
    if not isinstance(attention, dict):
        raise ValueError(f"program.identity.attention must be a geometry record, got {type(attention).__name__}")
    kernel = attention.get("kernel")
    if kernel not in FA2_SPLITKV_KERNEL_NAMES:
        raise ValueError(f"program.identity.attention.kernel {kernel!r}: this registry models the FA2 split-KV kernel only")
    if attention.get("num_splits", 1) != 1:
        raise ValueError(f"program.identity.attention.num_splits {attention.get('num_splits')!r} != 1: not the aligned split-KV path this registry models")
    if attention.get("kBlockM", 64) != 64:
        raise ValueError(f"program.identity.attention.kBlockM {attention.get('kBlockM')!r} != 64 (the split-KV kernels' query block)")
    bn = attention.get("kBlockN")
    if not isinstance(bn, int) or isinstance(bn, bool) or bn < 16 or bn % 16:
        raise ValueError(f"program.identity.attention.kBlockN {bn!r} must be a positive multiple of 16")
    return bn


def serve_program(C: dict, LP: int, STEPS: int, *, family: str | None = None, attention: dict | None = None):
    """[D57] The bound serving program of an instance identity: ``family`` `Serve_v1` (or None) with no ``attention`` -> `Serve_v1{C,
    LP, STEPS}` (the legacy geometry, BN 64 implicit); ``family`` `Serve_v4` with ``attention`` -> `Serve_v4{C, LP, STEPS, BN =
    attention.kBlockN}`.  A `Serve_v1` identity that carries a geometry, or a `Serve_v4` identity without one, is refused (ValueError):
    the geometry is part of the identity, not something the verifier infers."""
    fam = family or DEFAULT_SERVE_FAMILY
    defn = serve_family(fam)
    if defn is Serve:
        if attention is not None and attention != LEGACY_ATTENTION_GEOMETRY and attention_geometry_static(attention) != BN:
            raise ValueError(f"a Serve_v1 identity carries attention geometry {attention!r}; Serve_v1 IS the kBlockN = {BN} program -- pin Serve_v4 for another geometry")
        return bind(Serve, C=C, LP=LP, STEPS=STEPS)
    if attention is None:
        raise ValueError("a Serve_v4 identity must carry program.identity.attention (kernel, kBlockM, kBlockN, num_splits)")
    return bind(ServeV4, C=C, LP=LP, STEPS=STEPS, BN=attention_geometry_static(attention))


def serve_program_of_identity(identity: dict):
    """`serve_program` from a `protocol_instance.json#program.identity` dict (`case` names the registry config)."""
    case = identity.get("case")
    cfg = getattr(b1, f"{case}_CONFIG", None)
    if not isinstance(cfg, dict):
        raise KeyError(f"registry has no {case}_CONFIG")
    return serve_program(dict(cfg), int(identity["LP"]), int(identity["STEPS"]), family=identity.get("family"), attention=identity.get("attention"))
