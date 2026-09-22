"""[R10 owner 7] Independent verifier for owner 6's `e2e_witness.json` sidecar (schema `veritor/r10/e2e-witness/v1`,
`out/gen/r10/integ/witness_format.md`) — hashlib-only, built on `review_l1_verify.py`; no lane opening code imported.

Checks (each reported PASS / FAIL / SKIP with a reason):
  T  every tree's every leaf opening folds to the tree's bare merkle root under its declared framing
       (`cmt-chunk-fa2h-v7.s<S>` -> L1 step chunk leaf; `cmt-posleaf-v0` -> weights pos_leaf; `frame-v3` -> not implemented here => SKIP)
  S  step wrappers: layout_digest / ctx_digest recomputed, bind_root(program, ctx, geo, layout, N, merkle) == R_s, fold(R_*) -> run_root;
       the tree named `step:<s>` has root == merkle_root_s and count == n_leaves_s
  W  weights wrapper: names_digest, fold(subtree roots) -> weights_root; tree `weights:<name>` root == subtree_root, count == ceil(nbytes/256)
  C  challenge-v1: seed = SHA-256("verity/challenge/v1" || rho || Encode(instance)) recomputed from the record's rho + instance; the instance
       must bind THIS run's roots, Program and query (docs/protocol/challenge-v1.md); roots_written_utc < derived_utc
  O  obligations: x/W/y logical positions -> (rank, word_index) under the L1 position rule (stream_off from the step layout; weights offset 0);
       the ranks listed equal the chunk range of the 576-word vector; y.y_word (u16 int, one convention) == the word in the opened chunk; expected_y == hook_reference
  R  (informational) y == bf16(sequential fp32 dot(x, W))  -- NOT the gate-set order; a mismatch here is not by itself a failure
  G  statement.bin digest == guest.statement_digest (if the file is present)

Usage: review_e2e_verify.py <e2e_witness.json> [--dir <dir with statement.bin>] [--json out.json]
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from pathlib import Path

from verity_capture.bench import review_l1_verify as rv

CH = rv.CHUNK


def _res(out: dict, key: str, ok: bool | None, note: str = "") -> None:
    out[key] = {"verdict": "PASS" if ok else ("SKIP" if ok is None else "FAIL"), "note": note}


def path_from_json(p: list) -> list[bytes | None]:
    out: list[bytes | None] = []
    for st in p:
        if isinstance(st, dict):
            out.append(None if st.get("kind") == "lift" else bytes.fromhex(st["digest"]))
        else:
            out.append(None if st is None else bytes.fromhex(st))
    return out


LAYOUTS = {  # layout_id -> (step framing prefix, step leaf bytes, step tag, run tag, weights tag)
    "L1-chunk256": ("cmt-chunk-fa2h-v7.s", CH, rv.ROOT_TAG_STEP, rv.ROOT_TAG_RUN, rv.ROOT_TAG_WEIGHTS),
    "L2-row": ("cmt-chunk-fa2h-v7.row1152.s", rv.L2_CHUNK, rv.L2_TAG_STEP, rv.L2_TAG_RUN, rv.L2_TAG_WEIGHTS),
}


def leaf_for(layout_id: str, framing: str, rank: int, count: int, value: bytes) -> bytes | None:
    """Leaf digest under the framing, but ONLY framings that belong to `layout_id` are accepted (an L2 framing inside an L1 sidecar, or
    an unknown framing, returns None and is counted as a FAILED leaf, never skipped)."""
    prefix, nbytes, *_ = LAYOUTS.get(layout_id, (None, None))
    if prefix and framing.startswith(prefix) and framing[len(prefix):].isdigit():
        step = int(framing[len(prefix):])
        if len(value) != nbytes:
            return None
        return rv.l1_leaf(step, rank, value, count * (CH // 4)) if layout_id == "L1-chunk256" else rv.l2_leaf(step, rank, value, count)
    if framing == "cmt-posleaf-v0":
        return rv.pos_leaf(value)
    return None


def bf16_to_f32(w: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (w & 0xFFFF) << 16))[0]


def f32_to_bf16_rne(x: float) -> int:
    u = struct.unpack("<I", struct.pack("<f", x))[0]
    if (u & 0x7F800000) == 0x7F800000:              # inf / nan
        return (u >> 16) | (0x40 if (u & 0x7FFFFF) else 0)
    lsb = (u >> 16) & 1
    u = (u + 0x7FFF + lsb) >> 16
    return u & 0xFFFF


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def main(argv: list[str]) -> int:
    path = Path(argv[0])
    d = None
    js_out = None
    if "--dir" in argv:
        d = Path(argv[argv.index("--dir") + 1])
    if "--json" in argv:
        js_out = Path(argv[argv.index("--json") + 1])
    doc = json.loads(path.read_text())
    out: dict = {"file": str(path), "schema": doc.get("schema"), "layout_id": doc.get("layout_id")}
    # owner 6's harness keeps the step/weights tables in roots.json (sidecar.roots.roots_file); merge them in
    rf = doc.get("roots", {}).get("roots_file")
    if rf and (path.parent / rf).exists():
        rj = json.loads((path.parent / rf).read_text())
        doc.setdefault("roots", {})
        for k in ("steps", "weights", "run_root", "weights_root"):
            if k in rj and k not in doc["roots"]:
                doc["roots"][k] = rj[k]
        if "written_utc" in rj and "roots_written_utc" not in doc["roots"]:
            doc["roots"]["roots_written_utc"] = rj["written_utc"]
        out["roots_file"] = rf
        # independent GPU-tree check: recompute every step's merkle root from the retained padded stream (steps/step_<s>.bin)
        sd = path.parent / "steps"
        if sd.exists():
            rec_ok = rec_bad = 0; missing = 0
            for s in rj.get("steps", []):
                f = sd / f"step_{s['step']}.bin"
                if not f.exists():
                    missing += 1; continue
                data = f.read_bytes()
                mr = rv.l1_merkle_root(data, int(s["step"]))
                if mr.hex() == s["merkle_root"] and len(data) // CH == int(s["n_leaves"]):
                    rec_ok += 1
                else:
                    rec_bad += 1
            _res(out, "K_gpu_step_merkle_roots_recomputed_from_streams_hashlib", (rec_bad == 0 and rec_ok > 0) if (rec_ok + rec_bad) else None,
                 f"steps ok={rec_ok} bad={rec_bad} missing={missing}")
    ctx = doc.get("context", {})
    pd = bytes.fromhex(ctx["program_digest"]) if ctx.get("program_digest") else None
    geo = bytes.fromhex(ctx["geo_digest"]) if ctx.get("geo_digest") else None
    run_id = ctx.get("run_id")
    layout_id = str(doc.get("layout_id"))
    _res(out, "L_layout_id_known", layout_id in LAYOUTS, f"layout_id={layout_id!r} known={sorted(LAYOUTS)}")
    roots = doc.get("roots", {})
    trees = {t["id"]: t for t in doc.get("trees", [])}
    leaf_bytes: dict[tuple[str, int], bytes] = {}

    # ---- T: leaf openings fold to the bare roots
    n_leaves_ok = n_leaves_bad = 0
    fails = []
    for tid, t in trees.items():
        root = bytes.fromhex(t["root"]); count = int(t["count"]); fr = t["framing"]
        for lf in t.get("leaves", []):
            rank = int(lf["rank"]); val = bytes.fromhex(lf["bytes_hex"]); pth = path_from_json(lf["path"])
            leaf_bytes[(tid, rank)] = val
            leaf = leaf_for(layout_id, fr, rank, count, val)
            if leaf is None:
                n_leaves_bad += 1; fails.append((tid, rank, f"framing {fr!r}/len {len(val)} not admissible for layout {layout_id!r}")); continue
            try:
                shape = rv.path_shape(count, rank)
            except ValueError:
                n_leaves_bad += 1; fails.append((tid, rank, "rank outside tree")); continue
            if len(pth) != len(shape) or any((p is None) == has for p, has in zip(pth, shape)):
                n_leaves_bad += 1; fails.append((tid, rank, "path shape")); continue
            if rv.fold_path(leaf, rank, pth) == root:
                n_leaves_ok += 1
            else:
                n_leaves_bad += 1; fails.append((tid, rank, "fold != root"))
    _res(out, "T_leaf_openings_fold_to_bare_roots", (n_leaves_bad == 0 and n_leaves_ok > 0) if (n_leaves_ok + n_leaves_bad) else None,
         f"ok={n_leaves_ok} bad={n_leaves_bad} trees={len(trees)} layout={layout_id} fails={fails[:5]}")
    tag_step, tag_run, tag_w = LAYOUTS.get(layout_id, (None, None, rv.ROOT_TAG_STEP, rv.ROOT_TAG_RUN, rv.ROOT_TAG_WEIGHTS))[2:]
    LB = LAYOUTS.get(layout_id, (None, CH))[1]          # leaf bytes of this layout: 256 (L1) / 1152 (L2)

    def position(stream_off: int, flat_word: int) -> tuple[int, int]:
        b = stream_off + 2 * flat_word
        return b // LB, (b % LB) // 2

    # ---- S: step wrappers and run root
    steps = roots.get("steps", [])
    if steps and pd and geo and run_id:
        step_roots = []
        bad = []
        for s in steps:
            st = int(s["step"]); mr = bytes.fromhex(s["merkle_root"]); n = int(s["n_leaves"]); R = bytes.fromhex(s["root"])
            ld = rv.layout_digest(s["layout"]) if "layout" in s else (bytes.fromhex(s["layout_digest"]) if s.get("layout_digest") else None)
            if "layout" in s and s.get("layout_digest") and ld.hex() != s["layout_digest"]:
                bad.append((st, "layout_digest recomputed != stated"))
            if s.get("ctx_digest") and rv.ctx_digest(run_id, st).hex() != s["ctx_digest"]:
                bad.append((st, "ctx_digest != sha256(run_id/step=s)"))
            if ld is None or rv.l1_step_root(pd, run_id, st, geo, ld, n, mr, tag=tag_step) != R:
                bad.append((st, "bind_root != R_s"))
            t = trees.get(f"step:{st}")
            if t is not None and (bytes.fromhex(t["root"]) != mr or int(t["count"]) != n):
                bad.append((st, "tree step:s root/count != merkle_root/n_leaves"))
            step_roots.append(R)
        rr_ok = roots.get("run_root") and rv.l1_run_root(pd, geo, step_roots, tag=tag_run).hex() == roots["run_root"]
        _res(out, "S_step_wrappers_and_run_root", (not bad) and bool(rr_ok), f"steps={len(steps)} bad={bad[:5]} run_root_match={bool(rr_ok)}")
        _res(out, "S_program_digest_is_pinned", pd.hex() == "5bb5ac55c1ee110189c8d43e138ff1b41c1e7d48224777f1ef688d27c535edb9",
             f"program_digest={pd.hex()[:16]}… placeholder={hashlib.sha256(b'cmt-integ/no-program-digest/' + run_id.encode()).hexdigest()[:16]}…")
    else:
        _res(out, "S_step_wrappers_and_run_root", None, "roots.steps / context incomplete")

    # ---- W: weights wrapper
    w = roots.get("weights", {})
    if w.get("tensors") and geo and roots.get("weights_root"):
        names = [[t["name"], t["dtype"], list(t["shape"]), int(t["nbytes"])] for t in w["tensors"]]
        subs = [bytes.fromhex(t["subtree_root"]) for t in w["tensors"]]
        nd = hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).hexdigest()
        nd_ok = (w.get("names_digest") is None) or nd == w["names_digest"]
        wr_ok = rv.weights_root(geo, names, subs, tag=tag_w).hex() == roots["weights_root"]
        bad = []
        for t in w["tensors"]:
            tr = trees.get(f"weights:{t['name']}")
            if tr is not None:
                if bytes.fromhex(tr["root"]) != bytes.fromhex(t["subtree_root"]):
                    bad.append((t["name"], "tree root != subtree_root"))
                if int(tr["count"]) != -(-int(t["nbytes"]) // LB) or (t.get("n_leaves") and int(t["n_leaves"]) != int(tr["count"])):
                    bad.append((t["name"], f"count != ceil(nbytes/{LB})"))
        _res(out, "W_weights_wrapper", nd_ok and wr_ok and not bad, f"tensors={len(names)} names_digest_ok={nd_ok} weights_root_ok={wr_ok} bad={bad[:5]}")
    else:
        _res(out, "W_weights_wrapper", None, "roots.weights incomplete (full tensor list needed to recompute the weights root)")

    # ---- C: challenge
    # ONE derivation (docs/protocol/challenge-v1.md): seed = SHA-256("verity/challenge/v1" || rho || Encode(instance)); the instance's roots
    # must be THIS run's roots and its Program/query digests this run's context.  R10 records (`value`/`preimage`/`tag`) are legacy: FAIL.
    ch = doc.get("challenge", {})
    if ch.get("seed_hex") and ch.get("rho_hex") and isinstance(ch.get("instance"), dict) and roots.get("run_root") and roots.get("weights_root") and pd and ctx.get("query_id"):
        from verity_protocol import challenge as chal_v1

        try:
            inst = chal_v1.instance_from_json(ch["instance"])
            seed = chal_v1.challenge_seed(bytes.fromhex(ch["rho_hex"]), inst)
            binds = (inst.root("run-values-l1") == bytes.fromhex(roots["run_root"]) and inst.root("weights-l1") == bytes.fromhex(roots["weights_root"])
                     and inst.program_digest == pd and inst.query_digest == chal_v1.query_digest(bytes.fromhex(ctx["query_id"]))
                     and tuple(c.role for c in inst.commitments)[:2] == chal_v1.CONFIG_ROLES[chal_v1.CONFIG_L1_V1])
            spec = inst.sampler
            idx_ok = True
            if ch.get("indices") is not None:
                idx_ok = list(map(int, ch["indices"])) == chal_v1.derive_indices(seed, spec.population, spec.batch)
            _res(out, "C_challenge_recomputed", seed.hex() == ch["seed_hex"] and binds and idx_ok,
                 f"challenge-v1 seed_ok={seed.hex() == ch['seed_hex']} instance_binds_this_run={binds} indices_ok={idx_ok} batch={spec.batch} population={spec.population}")
        except (KeyError, ValueError, chal_v1.ChallengeError) as e:
            _res(out, "C_challenge_recomputed", False, f"challenge-v1 instance malformed: {type(e).__name__}: {e}")
        rw, du = roots.get("roots_written_utc"), ch.get("derived_utc")
        _res(out, "C_roots_written_before_challenge", (rw <= du) if (rw and du) else None,
             f"roots_written_utc={rw} derived_utc={du}" + (" (EQUAL at 1-s resolution: order NOT established by timestamps)" if rw == du else ""))
    elif ch.get("value"):
        _res(out, "C_challenge_recomputed", False, "legacy R10 challenge record (tag/preimage/value): not challenge-v1 -- regenerate with --rho")
    else:
        _res(out, "C_challenge_recomputed", None, "challenge/roots/query_id missing")

    # ---- O: obligations
    obls = doc.get("obligations", [])
    o_bad = []; o_ok = 0; r_match = 0; r_total = 0; hex_conv: dict[str, int] = {}
    step_layout = {int(s["step"]): s.get("layout") for s in steps}
    for ob in obls:
        k = ob.get("k")
        try:
            lg = ob["logical"]; st = int(lg["step"])
            lay = step_layout.get(st) or []
            offs = {row[1]: int(row[8]) for row in lay}
            # y
            y = ob["y"]; so = offs.get(y["tensor"])
            if so is None:
                o_bad.append((k, "y tensor not in step layout")); continue
            r, wi = position(so, int(y["flat_word"]))
            if r != int(y["rank"]) or wi != int(y["word_index"]):
                o_bad.append((k, f"y position rule: got ({r},{wi}) stated ({y['rank']},{y['word_index']})")); continue
            chunk = leaf_bytes.get((y["tree"], r))
            if chunk is None:
                o_bad.append((k, "y chunk not opened in trees")); continue
            yw = rv.word_in_chunk(chunk, wi)
            # ONE convention (challenge-v1 cleanup, docs/protocol/challenge-v1.md §0): the claimed y is the u16 integer
            # `y.y_word` / `expected_y`; hex-string variants (R10 owner-3 raw-bytes vs owner-6 u16-int) are rejected.
            bad_field = None
            for fld, val in (("y_word", y.get("y_word")), ("expected_y", ob.get("expected_y")), ("hook_reference", ob.get("hook_reference"))):
                if val is None:
                    if fld == "y_word":
                        bad_field = "y_word missing (legacy hex-string record; regenerate with the challenge-v1 stage 1)"; break
                    continue
                if type(val) is not int or val != yw:
                    bad_field = f"{fld}"; break
                hex_conv["u16-int"] = hex_conv.get("u16-int", 0) + 1
            if bad_field:
                o_bad.append((k, f"{bad_field} != committed word")); continue
            # operand vectors: GEMM coordinate (x: activations, W: weight row, K words each) or the scalar relation (`inputs`: 1 word each)
            vecs = {}
            if "inputs" in ob and "x" not in ob:
                operands = [(f"in{i}", op, None, op["tree"], 1) for i, op in enumerate(ob["inputs"])]
            else:
                Kx = int(doc.get("relation", {}).get("K", 576))
                operands = [("x", ob["x"], None, ob["x"]["tree"], Kx), ("W", ob["W"], 0, ob["W"]["tree"], Kx)]
            for nm, op, base_off, tree_key, K in operands:
                if base_off is None:
                    base_off = offs.get(op["tensor"])
                    if base_off is None:
                        raise KeyError(f"{nm} tensor not in layout")
                f0 = int(op["flat_word0"])
                r0, w0 = position(base_off, f0); r1, _ = position(base_off, f0 + K - 1)
                if sorted(int(x) for x in op["ranks"]) != list(range(r0, r1 + 1)) or int(op.get("word_index0", w0)) != w0:
                    raise ValueError(f"{nm} ranks/word_index0 mismatch: rule {list(range(r0, r1 + 1))},{w0} stated {op['ranks']},{op.get('word_index0')}")
                words = []
                for rk in range(r0, r1 + 1):
                    c = leaf_bytes.get((tree_key, rk))
                    if c is None:
                        raise KeyError(f"{nm} chunk {rk} not opened")
                    lo = w0 if rk == r0 else 0
                    hi = ((base_off + 2 * (f0 + K - 1)) % LB) // 2 + 1 if rk == r1 else LB // 2
                    words += [rv.word_in_chunk(c, i) for i in range(lo, hi)]
                if len(words) != K:
                    raise ValueError(f"{nm} extracted {len(words)} words != K")
                vecs[nm] = words
            o_ok += 1
            # R (informational): sequential fp32 dot (GEMM coordinate) / fp32 silu(gate)*up (scalar relation)
            if "x" in vecs:
                acc = 0.0
                for a, b in zip(vecs["x"], vecs["W"]):
                    acc = f32(acc + f32(bf16_to_f32(a) * bf16_to_f32(b)))
            else:
                g, u = bf16_to_f32(vecs["in0"][0]), bf16_to_f32(vecs["in1"][0])
                acc = f32(f32(g / f32(1.0 + math.exp(-g))) * u)
            r_total += 1; r_match += int(f32_to_bf16_rne(acc) == yw)
        except (KeyError, ValueError, TypeError) as e:
            o_bad.append((k, str(e)))
    _res(out, "O_obligation_positions_and_values", (not o_bad and o_ok > 0) if obls else None,
         f"ok={o_ok} bad={o_bad[:5]} n={len(obls)} leaf_bytes={LB} y_word_fields={hex_conv}")
    _res(out, "R_informational_sequential_fp32_dot_equals_y", (r_match == r_total) if r_total else None,
         f"match={r_match}/{r_total} (sequential fp32 accumulation, NOT the tc-ampere gate order; informational)")

    # ---- G: statement digest
    g = dict(doc.get("guest", {}))
    if d is not None and (d / "statement.bin").exists() and (d / "expected.json").exists():   # owner 6's stage-2 package directory
        exp = json.loads((d / "expected.json").read_text())
        gg = exp.get("guest", {}) if isinstance(exp.get("guest"), dict) else {}
        for k, v in (("statement", "statement.bin"), ("statement_digest", exp.get("statement_digest")), ("statement_format", exp.get("statement_format")),
                     ("witness_format", exp.get("witness_format")), ("vk", gg.get("vk_hash")), ("elf_sha256", gg.get("elf_sha256")), ("mode", exp.get("mode"))):
            if not g.get(k):
                g[k] = v
        # linkage: the package's expected values must be the sidecar's (same run, same challenge, same program digest)
        link_bad = [k for k, v in (("program_digest", ctx.get("program_digest")), ("query_id", ctx.get("query_id")), ("layout_id", doc.get("layout_id")),
                                    ("run_id", ctx.get("run_id"))) if exp.get(k) != v]
        if exp.get("roots", {}).get("run_root_bound") != roots.get("run_root") or exp.get("roots", {}).get("weights_root_bound") != roots.get("weights_root"):
            link_bad.append("bound roots")
        if (exp.get("challenge") or {}).get("value") != (doc.get("challenge") or {}).get("value"):
            link_bad.append("challenge")
        _res(out, "G_expected_json_matches_sidecar", not link_bad, f"mismatched={link_bad}" if link_bad else "program_digest, query_id, layout_id, run_id, bound roots, challenge all equal")
    if d is not None and g.get("statement") and (d / g["statement"]).exists() and g.get("statement_digest"):
        sd = hashlib.sha256((d / g["statement"]).read_bytes()).hexdigest()
        _res(out, "G_statement_digest", sd == g["statement_digest"], f"sha256(statement.bin)={sd[:16]}… stated={g['statement_digest'][:16]}…")
    else:
        _res(out, "G_statement_digest", None, "statement.bin or digest not available")
    out["guest_label"] = {k: g.get(k) for k in ("mode", "approved", "vk", "elf_sha256", "statement_format", "witness_format")}
    if "R_informational_sequential_fp32_dot_equals_y" in out:   # informational: my host semantics are NOT the gate set; never a verdict
        r = out["R_informational_sequential_fp32_dot_equals_y"]
        r["verdict"] = {"PASS": "INFO-MATCH", "FAIL": "INFO-MISMATCH", "SKIP": "SKIP"}[r["verdict"]]
    verdicts = [v["verdict"] for v in out.values() if isinstance(v, dict) and "verdict" in v]
    out["ALL_NON_SKIPPED_PASS"] = all(v != "FAIL" for v in verdicts) and any(v == "PASS" for v in verdicts)
    txt = json.dumps(out, indent=1, default=str)
    print(txt)
    if js_out:
        js_out.write_text(txt)
    return 0 if out["ALL_NON_SKIPPED_PASS"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
