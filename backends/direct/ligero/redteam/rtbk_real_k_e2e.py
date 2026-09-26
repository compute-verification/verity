"""red-team-bligero-real-k: cross-K, mixed-steps, sized-frame and coverage attacks on real ``bench-vu`` dumps.

Input: honest CPU dumps ``<dumps>/honest-<relation>+<leaf>/`` (``bench-vu --auth included-hash --leaf L --dump-dir``,
two sub-batches each, interactive + zk).  Each variant copies a dump, rewrites statement bytes (relation string, header
steps / K, digest elements) or moves files between dumps, fixes the manifest's custody hashes so the checks behind custody
are reached, and runs every verifier the cells rely on:

* Rust ``ligero-verify verify`` per sub-batch (pinned, with the dump's coins file) and ``batch --target-bits 128``;
* Python ``serialize.verify_files``;
* ``reverify.verify_tree`` (custody, system pin vs manifest, commitments recomputed from the regenerated set, R1 / R2 / R4,
  Rust batch per rep).

Variants (every one must be refused by every verifier that sees it; the honest controls accepted by all):
  relabel         a <src> statement renamed <dst>, its own system
  relabel-sys     ... against <dst>'s pinned system
  relabel-shape   ... and its header steps / K rewritten to <dst>'s
  mixed           one rep holding <dst>'s sub_00 and <src>'s sub_01 (as relabel-shape), <dst>'s system
  steps-lie       an honest statement with header steps / K of the other K
  k-lie           header K alone rewritten (steps kept)
  digest-last     one limb of VU 0's LAST chunk-CV slot changed (the slot a 3-slot reader would never read)
  header-3slot    VU 0's x header in the 3-slot encoding (n_chunks + 4 role)
  header-nch      VU 0's x header claiming one chunk fewer, the last slot zeroed
  orphan / drop / dup   R4 / R2 on a real-K dump: a statement without its proof, a sub-batch gone, sub_00 twice

    python -m backends.direct.ligero.redteam.rtbk_real_k_e2e --bin BIN --dumps /tmp/rt/e2e --out DIR

Exit 0 iff every expectation holds."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
from pathlib import Path

PAIRS = [("fp8-ada-x4+blake3-xob", "fp8-ada-x4-k2048+blake3-xob"),
         ("bf16-hopper-x4+blake3-xob", "bf16-hopper-x4-k2048+blake3-xob"),
         ("bf16-hopper-x4-k2048+blake3-xob", "bf16-hopper-x4-k8192+blake3-xob"),
         ("bf16-hopper-x4+sha256", "bf16-hopper-x4-k2048+sha256")]
SIZED = ["bf16-ampere-x4-k2048+blake3-xob", "bf16-hopper-x4-k8192+blake3-xob", "fp8-hopper-x4-k8192+blake3-xob",
         "fp8-ada-x4-k2048+blake3-xob", "bf16-hopper-x4-k2048+sha256"]
OFF_STEPS, OFF_K, OFF_REL = 50, 54, 59          # serialize._header_w: magic 8, mode, l, n, D, t, f64, u8, u32, u8, n_proofs, n_vus


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _fields(b: bytes) -> dict:
    n_rel = struct.unpack_from("<I", b, OFF_REL)[0]
    rel = b[OFF_REL + 4:OFF_REL + 4 + n_rel].decode()
    p = OFF_REL + 4 + n_rel
    n_auth = struct.unpack_from("<I", b, p)[0]
    return {"n_vus": struct.unpack_from("<I", b, 46)[0], "steps": struct.unpack_from("<I", b, OFF_STEPS)[0],
            "K": struct.unpack_from("<I", b, OFF_K)[0], "relation": rel, "digests_at": p + 4 + n_auth}


def rewrite(b: bytes, relation: str | None = None, steps: int | None = None, K: int | None = None,
            patch: list[tuple[int, int]] | None = None) -> bytes:
    """Statement bytes with the header fields / relation string / digest elements (``(element index, value)`` over the
    flattened n_vus x 2 digest_elems u32 table) replaced; nothing else touched."""
    f = _fields(b)
    out = bytearray(b)
    for i, v in patch or []:
        struct.pack_into("<I", out, f["digests_at"] + 4 * i, v)
    if steps is not None:
        struct.pack_into("<I", out, OFF_STEPS, steps)
    if K is not None:
        struct.pack_into("<I", out, OFF_K, K)
    if relation is not None:
        old = OFF_REL + 4 + len(f["relation"].encode())
        out = out[:OFF_REL] + struct.pack("<I", len(relation.encode())) + relation.encode() + out[old:]
    return bytes(out)


class Checker:
    def __init__(self, binary: str, out: Path):
        self.bin, self.out = str(Path(binary).resolve()), out
        from .. import reverify as RV
        self.RV = RV
        self.v = RV.Verifier(Path(self.bin), _sha(Path(self.bin)), None)

    def rust_verify(self, d: Path, stem: str) -> dict:
        args = [self.bin, "verify", "--system", str(d / "system.bin"), "--statement", str(d / "rep1" / f"{stem}.stmt"),
                "--proof", str(d / "rep1" / f"{stem}.proof")]
        if (d / "rep1" / f"{stem}.coins").is_file():
            args += ["--coins", str(d / "rep1" / f"{stem}.coins")]
        r = subprocess.run(args, capture_output=True, text=True)
        try:
            js = json.loads(r.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            return {"accepted": False, "pinned": None, "reason": f"rc {r.returncode}: {r.stderr.strip()[-200:]}"}
        return {"accepted": bool(js.get("accepted")), "pinned": js.get("system_pinned"), "reason": str(js.get("reason"))[:220]}

    def rust_batch(self, d: Path) -> dict:
        js_path = d / "batch.json"
        r = subprocess.run([self.bin, "batch", "--system", str(d / "system.bin"), "--dir", str(d / "rep1"), "--target-bits", "128",
                            "--jobs", "2", "--json", str(js_path)], capture_output=True, text=True)
        if not js_path.is_file():
            return {"accepted": False, "reason": f"rc {r.returncode}: {r.stderr.strip()[-200:]}"}
        js = json.loads(js_path.read_text())
        return {"accepted": bool(js.get("batch_accepted")) and bool(js.get("system_pinned")), "reason": str(js.get("batch_reason"))[:220],
                "n": js.get("n"), "rejected": js.get("rejected")}

    def python_verify(self, d: Path, stem: str) -> dict:
        from ..serialize import verify_files
        try:
            ok, why, _ = verify_files(d / "rep1" / f"{stem}.stmt", d / "rep1" / f"{stem}.proof", device="cpu")
        except Exception as e:  # noqa: BLE001 -- a reader that raises refuses the file
            ok, why = False, f"raised {type(e).__name__}: {e}"
        return {"accepted": bool(ok), "reason": str(why)[:220]}

    def reverify(self, d: Path) -> dict:
        man = json.loads((d / "manifest.json").read_text())
        rep = self.RV.Report(f"art:rtbk-{d.name}")
        try:
            self.RV.verify_tree(d, man, self.v, rep, jobs=2, params={}, asserted=None, out_dir=d / "reverify")
        except Exception as e:  # noqa: BLE001
            return {"status": "ERROR", "why": [f"raised {type(e).__name__}: {e}"], "relation": None}
        return {"status": rep.status, "why": [str(w)[:220] for w in rep.why][:3], "relation": rep.relation}

    def run(self, d: Path, stems: list[str], python: bool = True) -> dict:
        res = {"rust": {s: self.rust_verify(d, s) for s in stems if (d / "rep1" / f"{s}.proof").is_file()},
               "batch": self.rust_batch(d), "reverify": self.reverify(d)}
        if python:
            res["python"] = {s: self.python_verify(d, s) for s in stems if (d / "rep1" / f"{s}.proof").is_file()}
        return res


def clone(src: Path, dst: Path) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("reverify", "batch.json", "verify_python.json"))
    return dst


def fix_manifest(d: Path, relation: str | None = None) -> None:
    """Custody hashes of the files as they now are (the variant's changes are deliberate); drop entries whose files are gone."""
    man = json.loads((d / "manifest.json").read_text())
    files = []
    for f in man.get("files", []):
        for key in ("proof", "stmt", "coins"):
            p = f.get(key)
            if p and (d / p).is_file():
                f[f"{key}_sha256" if key != "stmt" else "stmt_sha256"] = _sha(d / p)
                f[f"{key}_bytes"] = (d / p).stat().st_size
        if f.get("proof") and not (d / f["proof"]).is_file():
            f.pop("proof")
        if f.get("stmt") and (d / f["stmt"]).is_file() or f.get("proof"):
            files.append(f)
    man["files"] = files
    man["system_file"]["sha256"] = _sha(d / "system.bin")
    man["system_file"]["bytes"] = (d / "system.bin").stat().st_size
    if relation is not None:
        man["relation"]["name"] = relation.split("+")[0]
        man["relation"]["statement_relation"] = relation
    (d / "manifest.json").write_text(json.dumps(man, indent=1))


def hashauth_only(stmt: Path, leaf, rel) -> dict:
    """The statement's leaves under its roots (``hashauth.verify_hash_auth``, the multiproof check alone), apart from the
    Ligero proof: whether a changed digest still opens under the committed root."""
    import numpy as np

    from ..hashauth import verify_hash_auth
    from ..serialize import read_statement
    try:
        st = read_statement(stmt.read_bytes())
        ends = np.arange(st.n_vus) * st.steps + st.steps - 1
        tab = np.asarray(st.a, dtype=np.int64)
        ok, why = verify_hash_auth(st.hash_auth, st.hash_sibs, digests=tab[ends, 1:1 + 2 * leaf.digest_elems],
                                   y_words=np.asarray(st.y16, dtype=np.int64)[ends], y_bits=rel.y_bits, leaf=leaf)
        return {"accepted": bool(ok), "detail": why[:200]}
    except Exception as e:  # noqa: BLE001
        return {"accepted": False, "detail": f"raised {type(e).__name__}: {e}"[:200]}


def refused(res: dict) -> bool:
    """Every verifier refuses: no sub-batch accepted by Rust or Python where the variant changed it, the batch and reverify
    not accepted, and (digest variants) the changed leaf does not open under the root."""
    touched = res.get("touched")
    rust = [v["accepted"] for s, v in res["rust"].items() if touched is None or s in touched]
    py = [v["accepted"] for s, v in res.get("python", {}).items() if touched is None or s in touched]
    return (not any(rust) and not res["batch"]["accepted"] and res["reverify"]["status"] != "PASS" and not any(py)
            and not res.get("hashauth_only", {}).get("accepted", False))


def accepted(res: dict) -> bool:
    return bool(res["rust"] and all(v["accepted"] and v["pinned"] for v in res["rust"].values()) and res["batch"]["accepted"]
                and res["reverify"]["status"] == "PASS" and all(v["accepted"] for v in res.get("python", {}).values())
                and res.get("hashauth_only", {}).get("accepted", True))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bin", required=True)
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="comma-separated variant name prefixes")
    a = ap.parse_args()
    from ..leaf import registry
    from ..relations import relation
    dumps, out = Path(a.dumps), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = Checker(a.bin, out)
    stems = ["sub_00", "sub_01"]
    results, bad = [], []

    def want(name: str) -> bool:
        return a.only is None or any(name.startswith(p) for p in a.only.split(","))

    def record(name: str, res: dict, expect: str) -> None:
        ok = accepted(res) if expect == "accept" else res["reverify"]["status"] != "PASS" if expect == "reverify" else refused(res)
        row = {"variant": name, "expect": expect, "ok": ok, **res}
        results.append(row)
        brief = {k: ([v["accepted"] for v in res[k].values()] if k in ("rust", "python") else res[k].get("accepted", res[k].get("status")))
                 for k in ("rust", "python", "batch", "reverify") if k in res}
        print(f"{'OK ' if ok else 'BAD'} {name} (expect {expect}): {json.dumps(brief)}", flush=True)
        if not ok:
            bad.append(name)
        (out / "results.json").write_text(json.dumps({"ok": not bad, "bad": bad, "results": results}, indent=1))

    def honest(n: str) -> Path:
        return dumps / f"honest-{n}"

    def have(n: str) -> bool:
        return (honest(n) / "manifest.json").is_file() and (dumps / f"honest-{n}.json").is_file()

    # honest controls
    for n in sorted({x for p in PAIRS for x in p} | set(SIZED)):
        if have(n) and want("honest"):
            d = clone(honest(n), out / f"honest-{n}")
            res = ck.run(d, stems)
            res["hashauth_only"] = hashauth_only(d / "rep1" / "sub_00.stmt", registry.for_relation(n), relation(n.split("+")[0]))
            record(f"honest {n}", res, "accept")

    for src, dst in PAIRS + [(d, s) for s, d in PAIRS]:
        if not (have(src) and have(dst)):
            continue
        rd = relation(dst.split("+")[0])
        tag = f"{src}->{dst}"
        for v, use_dst_sys, shape in (("relabel", False, False), ("relabel-sys", True, False), ("relabel-shape", True, True)):
            if not want(v):
                continue
            d = clone(honest(src), out / f"{v}--{tag}")
            if use_dst_sys:
                shutil.copy(honest(dst) / "system.bin", d / "system.bin")
            for s in stems:
                p = d / "rep1" / f"{s}.stmt"
                p.write_bytes(rewrite(p.read_bytes(), relation=dst, steps=rd.steps if shape else None,
                                      K=rd.steps * rd.k if shape else None))
            fix_manifest(d, relation=dst)
            record(f"{v} {tag}", ck.run(d, stems), "refuse")
        if want("mixed"):
            d = clone(honest(dst), out / f"mixed--{tag}")
            for ext in ("stmt", "proof", "coins"):
                shutil.copy(honest(src) / "rep1" / f"sub_01.{ext}", d / "rep1" / f"sub_01.{ext}")
            p = d / "rep1" / "sub_01.stmt"
            p.write_bytes(rewrite(p.read_bytes(), relation=dst, steps=rd.steps, K=rd.steps * rd.k))
            fix_manifest(d)
            res = ck.run(d, stems)
            res["touched"] = ["sub_01"]
            record(f"mixed {tag}", res, "refuse")

    for n in SIZED:
        if not have(n):
            continue
        rel = relation(n.split("+")[0])
        leaf = registry.for_relation(n)
        de = leaf.digest_elems
        other_steps = 24 if rel.steps != 24 else 32
        # steps-lie2 keeps the row (K words, so the digest table still parses at its width) and halves steps: the
        # statement reaches the pinned shape check instead of failing the parse
        for v, kw in (("steps-lie", {"steps": other_steps, "K": other_steps * rel.k}), ("steps-lie2", {"steps": rel.steps // 2}),
                      ("k-lie", {"K": 1536 if rel.K != 1536 else 2048})):
            if want(v):
                d = clone(honest(n), out / f"{v}--{n}")
                for s in stems:
                    p = d / "rep1" / f"{s}.stmt"
                    p.write_bytes(rewrite(p.read_bytes(), **kw))
                fix_manifest(d)
                record(f"{v} {n}", ck.run(d, stems), "refuse")
        b0 = (honest(n) / "rep1" / "sub_00.stmt").read_bytes()
        f0 = _fields(b0)
        dg = struct.unpack_from(f"<{2 * de}I", b0, f0["digests_at"])           # VU 0: x digest then W digest
        patches = {"digest-last": [(de - 1, dg[de - 1] ^ 1)]}
        if n.endswith("blake3-xob"):
            from ..leaf.blake3 import N_CHUNKS_MAX, parse_header
            nch, role = parse_header(dg[0], leaf.n_chunks_max)
            if leaf.n_chunks_max > N_CHUNKS_MAX:              # on the registry frame the 3-slot header IS the honest one
                patches["header-3slot"] = [(0, nch + 4 * role)]
            patches["header-nch"] = [(0, dg[0] - 1)] + [(i, 0) for i in range(1 + 16 * (nch - 1), 1 + 16 * nch)]
        for v, patch in patches.items():
            if want(v):
                d = clone(honest(n), out / f"{v}--{n}")
                p = d / "rep1" / "sub_00.stmt"
                p.write_bytes(rewrite(p.read_bytes(), patch=patch))
                fix_manifest(d)
                res = ck.run(d, stems)
                res["hashauth_only"] = hashauth_only(p, leaf, rel)
                res["touched"] = ["sub_00"]
                record(f"{v} {n}", res, "refuse")
        if n == SIZED[0]:
            for v in ("orphan", "drop", "dup"):
                if not want(v):
                    continue
                d = clone(honest(n), out / f"{v}--{n}")
                if v == "orphan":
                    (d / "rep1" / "sub_01.proof").unlink()
                elif v == "drop":
                    for ext in ("stmt", "proof", "coins"):
                        (d / "rep1" / f"sub_01.{ext}").unlink()
                else:
                    for ext in ("stmt", "proof", "coins"):
                        shutil.copy(d / "rep1" / f"sub_00.{ext}", d / "rep1" / f"sub_01.{ext}")
                fix_manifest(d)
                res = ck.run(d, stems, python=False)
                res["touched"] = []
                # R2 / R4 are reverify's (Rust batch verifies what it is given: a duplicated sub-batch is two valid proofs)
                record(f"{v} {n}", res, "reverify" if v == "dup" else "refuse")

    print(f"\n{len(results) - len(bad)}/{len(results)} as expected; unexpected: {bad or 'none'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
