"""Data version v3: a proportional sample of OLMo 2's stage-1 mix, its 32,000-token BPE (`bpe32k`), the training stream
and the held-out sets (CPU, network). Fixed: candidates never run it or read its output.

    python data_v3.py select [--tree JSON]                   # metadata only: the hub's listing -> data_v3.json (in git)
    python data_v3.py build [--workers 24] [--push NODE ...]  # one node: everything below, from data_v3.json
    python data_v3.py verify [--keep]                         # each node: check what is installed, then make it read-only

The sample (`select`): `allenai/olmo-mix-1124` at REVISION, each source (a directory under `data/`) sampled to FRACTION
of its compressed bytes, and so of its tokens: its files in a seeded order (`random.Random("v3/<source>")`), whole while
they fit, and the first that doesn't cut to its first documents that hold its `take` share of the file's text bytes.
data_v3.json lists every chosen file with the hub's LFS sha256; a download that differs is refused.

A document is (file, line), and h is sha256 of `<path>\\n<line>`. Unit u is DCLM's u-th chosen file plus the other
sources' documents whose h mod P is u (P: DCLM's chosen files), its documents in h order, so each unit is a proportional
sample of the mix. Unit 0 is held out; unit u >= 1 is the training stream's part u - 1.

The held-out sets are unit 0's documents in h order, less any whose text's sha256 shares its first 8 bytes with a training
document's or an earlier candidate's, each taken whole in turn until a set holds its bpe32k tokens: the search pool (2^23),
promotion (2^21), final (2^21), then the ladder's validation stream (2^22, `TRAIN/val.bin`, which candidates may not
read). Each set is tokenized by bpe32k and GPT-2 (final also by Pythia's tokenizer), each document prefixed by the
tokenizer's end-of-text, with data.py's manifest fields (`doc_bytes`, `doc_tokens`).

bpe32k is a byte-level BPE (GPT-2's pre-tokenization, no normalizer, the 256 bytes as its alphabet) of VOCAB tokens with
`<|endoftext|>` as id 0, trained by `tokenizers`' BpeTrainer on parts 0-7's documents in h order. Text that spells
`<|endoftext|>` encodes as text, so a document's tokens' bytes are its text's (checked for every part and set).

The training stream TRAIN has ladder_data.py's format: parts/NNN.bin (uint16, each document prefixed by end-of-text 0),
then parts/NNN.json, and manifest.json last. `--push` copies each output to the same path on the named nodes
(/etc/vy/peers) as it is written, records after their data. The raw downloads are deleted as their units are tokenized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import random
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from sealed import SEALED

REPO = "allenai/olmo-mix-1124"
REVISION = "99ee6aaace88779d1ef099d36251b91101c1679b"
URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/"
DATA = "v3"
TOKENIZER = "bpe32k"
VOCAB = 32_000
EOT = {TOKENIZER: 0, "gpt2": 50256, "pythia": 0}
FRACTION = 0.0098
DCLM = "dclm"
SELECTION = Path(__file__).resolve().with_name("data_v3.json")
ROOT = Path(os.environ.get("CW_ROOT", "/workspace/research/scratch/covert-witness"))
TRAIN = ROOT / "train" / DATA
HELD = ROOT / "heldout" / DATA
BUILD = HELD / "build"
TOKENIZER_JSON = TRAIN / "tokenizer.json"
TOKENIZER_PARTS = 8
VAL_TOKENS = 2 ** 22
SETS = {"search-pool": (HELD / "search-pool", 2 ** 23, (TOKENIZER, "gpt2")),
        "promotion": (SEALED / "heldout" / DATA / "promotion", 2 ** 21, (TOKENIZER, "gpt2")),
        "final": (SEALED / "heldout" / DATA / "final", 2 ** 21, (TOKENIZER, "gpt2", "pythia"))}
DATA_JSON = HELD / "data.json"
PYTHIA_REPO = "EleutherAI/pythia-160m"
PEERS = Path("/etc/vy/peers")
BATCH = 4096


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def doc_hash(path: str, line: int) -> bytes:
    return hashlib.sha256(f"{path}\n{line}".encode()).digest()


def text_hash(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


def read_only(*paths: Path) -> None:
    """Each output file is read-only once written, so its copies on other nodes are too (rsync keeps the mode)."""
    for p in paths:
        p.chmod(0o444)


# ---- the sample ---------------------------------------------------------------------------------------------------------

def select(tree: list[dict]) -> dict:
    """Each source's chosen files (`take` < 1 on the last: the share of its text it gives)."""
    by = defaultdict(list)
    for f in tree:
        parts = f["path"].split("/")
        if parts[0] == "data" and len(parts) > 2 and f.get("lfs"):
            by[parts[1]].append(f)
    sources = {}
    for src in sorted(by):
        files = sorted(by[src], key=lambda f: f["path"])
        random.Random(f"{DATA}/{src}").shuffle(files)
        total = sum(f["size"] for f in files)
        target, have, chosen = FRACTION * total, 0, []
        for f in files:
            take = min(1.0, (target - have) / f["size"])
            if take <= 0:
                break
            chosen.append({"path": f["path"], "size": f["size"], "sha256": f["lfs"], "take": take})
            have += f["size"]
            if take < 1:
                break
        sources[src] = {"files": len(files), "bytes": total, "target_bytes": round(target),
                        "chosen_bytes": round(sum(c["size"] * c["take"] for c in chosen)), "chosen": chosen}
    return {"schema": "covert-witness/olmo-sample/v1", "data": DATA, "repo": REPO, "revision": REVISION,
            "fraction": FRACTION, "seed": f"random.Random('{DATA}/<source>').shuffle(the source's files in path order)",
            "rule": "each source's files in the seeded order, whole while their compressed bytes fit FRACTION of the "
                    "source's; the first that doesn't gives its first documents holding `take` of its text's UTF-8 bytes",
            "sources": sources}


def hub_tree() -> list[dict]:
    from huggingface_hub import HfApi

    out = []
    for f in HfApi().list_repo_tree(REPO, repo_type="dataset", revision=REVISION, recursive=True):
        if getattr(f, "lfs", None):
            out.append({"path": f.path, "size": f.size, "lfs": f.lfs.sha256})
    return out


def selection() -> dict:
    return json.loads(SELECTION.read_text())


def plan(sel: dict, units: int | None = None) -> tuple[list[dict], list[dict]]:
    """(DCLM's chosen files, one a unit; every other source's chosen files)."""
    dclm = sel["sources"][DCLM]["chosen"][:units]
    small = [f for s, v in sorted(sel["sources"].items()) if s != DCLM for f in v["chosen"]]
    return dclm, small


# ---- reading --------------------------------------------------------------------------------------------------------------

def texts(path: Path, cap: int | None = None, stop: int | None = None):
    """(line, text, its UTF-8 bytes) of each document of a file before line `stop` (and `cap`), less any whose text is
    not Unicode (JSON can spell a lone surrogate)."""
    cmd = ["zstd", "-dc", str(path)] if path.name.endswith((".zstd", ".zst")) else ["pigz", "-dc", str(path)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1 << 22)
    end = min(x for x in (cap, stop, float("inf")) if x is not None)
    whole = False
    try:
        for i, line in enumerate(p.stdout):
            if i >= end:
                break
            t = json.loads(line)["text"]
            try:
                n = len(t.encode("utf-8"))
            except UnicodeEncodeError:
                continue
            yield i, t, n
        else:
            whole = True
    finally:
        p.stdout.close()
        if not whole:
            p.kill()
        if p.wait() and whole:
            raise SystemExit(f"data_v3: {' '.join(cmd)} failed")


def taken(path: Path, take: float, cap: int | None):
    """The file's documents that `take` keeps: all, or the first whose text holds that share of the file's bytes."""
    limit = take * sum(n for _, _, n in texts(path, cap)) if take < 1 else None
    have = 0
    for i, t, n in texts(path, cap):
        if limit is not None and have >= limit:
            return
        have += n
        yield i, t, n


def fetch(f: dict, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"{f['sha256'][:16]}-{Path(f['path']).name}"
    if out.is_file() and sha256(out) == f["sha256"]:
        return out
    tmp = out.with_name(out.name + ".part")
    subprocess.run(["curl", "-sSfL", "--retry", "8", "--retry-all-errors", "-o", str(tmp), URL + f["path"]], check=True)
    if (got := sha256(tmp)) != f["sha256"]:
        tmp.unlink()
        raise SystemExit(f"data_v3: {f['path']} has sha256 {got}, the hub's LFS id is {f['sha256']}")
    tmp.rename(out)
    return out


def index_unit(u: int, f: dict, cap: int | None) -> dict:
    """Fetch DCLM's file for unit u and record its kept documents' text hashes (BUILD/index/UUUU.npy)."""
    rec_path = BUILD / "index" / f"{u:04d}.json"
    if rec_path.is_file():
        return json.loads(rec_path.read_text())
    path = fetch(f, BUILD / "raw")
    hashes, nbytes, stop = [], 0, 0
    for i, t, n in taken(path, f["take"], cap):
        hashes.append(text_hash(t))
        nbytes += n
        stop = i + 1
    np.save(BUILD / "index" / f"{u:04d}.npy", np.array(hashes, dtype=np.uint64))
    rec = {"unit": u, "file": f["path"], "file_sha256": f["sha256"], "take": f["take"], "documents": len(hashes),
           "lines": stop, "text_bytes": nbytes, "local": str(path)}
    rec_path.write_text(json.dumps(rec) + "\n")
    return rec


def index_small(j: int, f: dict, units: int, cap: int | None) -> dict:
    """Fetch another source's file j, write its kept documents to their units (BUILD/small/UUUU/JJJ.jsonl, [h, text]),
    record their units and text hashes, and delete the download."""
    rec_path = BUILD / "small" / f"{j:03d}.json"
    if rec_path.is_file():
        return json.loads(rec_path.read_text())
    path = fetch(f, BUILD / "raw")
    outs, us, hashes, nbytes = {}, [], [], 0
    for i, t, n in taken(path, f["take"], cap):
        h = doc_hash(f["path"], i)
        u = int.from_bytes(h[:8], "big") % units
        if u not in outs:
            (BUILD / "small" / f"{u:04d}").mkdir(parents=True, exist_ok=True)
            outs[u] = (BUILD / "small" / f"{u:04d}" / f"{j:03d}.jsonl").open("w")
        outs[u].write(json.dumps([h.hex(), t]) + "\n")
        us.append(u)
        hashes.append(text_hash(t))
        nbytes += n
    for fh in outs.values():
        fh.close()
    np.save(BUILD / "small" / f"{j:03d}.npy", np.array([us, hashes], dtype=np.uint64))
    path.unlink()
    rec = {"file": f["path"], "file_sha256": f["sha256"], "take": f["take"], "documents": len(us), "text_bytes": nbytes}
    rec_path.write_text(json.dumps(rec) + "\n")
    return rec


def unit_docs(u: int, rec: dict, cap: int | None) -> tuple[list[str], int]:
    """Unit u's documents in h order, and how many are DCLM's."""
    docs = [(doc_hash(rec["file"], i), t) for i, t, _ in texts(Path(rec["local"]), cap, rec["lines"])]
    n = len(docs)
    if n != rec["documents"]:
        raise SystemExit(f"data_v3: unit {u} read {n} documents, its index {rec['documents']}")
    for p in sorted((BUILD / "small" / f"{u:04d}").glob("*.jsonl")) if (BUILD / "small" / f"{u:04d}").is_dir() else []:
        with p.open() as fh:
            docs += [(bytes.fromhex(h), t) for h, t in map(json.loads, fh)]
    docs.sort(key=lambda x: x[0])
    return [t for _, t in docs], n


# ---- tokenizers ---------------------------------------------------------------------------------------------------------

def train_tokenizer(docs, out: Path) -> str:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB, min_frequency=0, show_progress=False,
                                  special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(docs, trainer=trainer)
    if tok.get_vocab_size() != VOCAB or tok.token_to_id("<|endoftext|>") != EOT[TOKENIZER]:
        raise SystemExit(f"data_v3: the trained tokenizer has {tok.get_vocab_size()} tokens and end-of-text "
                         f"{tok.token_to_id('<|endoftext|>')}, not {VOCAB} and {EOT[TOKENIZER]}")
    out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out.with_name(out.name + ".part")))
    out.with_name(out.name + ".part").rename(out)
    read_only(out)
    return sha256(out)


def load_bpe(path: Path = TOKENIZER_JSON):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(path))
    tok.encode_special_tokens = True
    return tok


def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def token_bytes(path: Path = TOKENIZER_JSON) -> np.ndarray:
    """Each bpe32k id's bytes (end-of-text 0), from tokenizer.json's byte-level vocabulary."""
    dec = {c: b for b, c in bytes_to_unicode().items()}
    vocab = json.loads(path.read_text())["model"]["vocab"]
    out = np.zeros(VOCAB, dtype=np.int64)
    for s, i in vocab.items():
        out[i] = len(s) if i == EOT[TOKENIZER] else len(bytes(dec[c] for c in s))
    out[EOT[TOKENIZER]] = 0
    return out


def stream(ids: list[list[int]], eot: int) -> np.ndarray:
    out = np.empty(sum(map(len, ids)) + len(ids), dtype=np.uint16)
    p = 0
    for x in ids:
        out[p] = eot
        out[p + 1: p + 1 + len(x)] = x
        p += 1 + len(x)
    return out


class Encoders:
    def __init__(self, names: tuple[str, ...] = (TOKENIZER,)):
        self.bpe = load_bpe()
        self.gpt2 = self.pythia = None
        if "gpt2" in names:
            import tiktoken

            self.gpt2 = tiktoken.get_encoding("gpt2")
        if "pythia" in names:
            from huggingface_hub import HfApi
            from transformers import AutoTokenizer

            self.pythia_revision = HfApi().model_info(PYTHIA_REPO).sha
            self.pythia = AutoTokenizer.from_pretrained(PYTHIA_REPO, revision=self.pythia_revision)
            if self.pythia.convert_tokens_to_ids("<|endoftext|>") != EOT["pythia"]:
                raise SystemExit("data_v3: Pythia's end-of-text id is not EOT['pythia']")

    def ids(self, which: str, docs: list[str]) -> list[list[int]]:
        out = []
        for k in range(0, len(docs), BATCH):
            b = docs[k: k + BATCH]
            if which == TOKENIZER:
                enc = getattr(self.bpe, "encode_batch_fast", self.bpe.encode_batch)
                out += [e.ids for e in enc(b, add_special_tokens=False)]
            elif which == "gpt2":
                out += self.gpt2.encode_ordinary_batch(b, num_threads=len(os.sched_getaffinity(0)))
            else:
                out += self.pythia(b, add_special_tokens=False)["input_ids"]
        return out


def _threads(n: int) -> None:
    os.environ["RAYON_NUM_THREADS"] = str(n)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"


def tokenize_part(u: int, rec: dict, tb_path: str, cap: int | None) -> dict:
    """Unit u's documents as part u - 1 (TRAIN/parts/NNN.bin, then .json); the DCLM download is deleted."""
    t0 = time.time()
    docs, n_dclm = unit_docs(u, rec, cap)
    arr = stream(Encoders().ids(TOKENIZER, docs), EOT[TOKENIZER])
    nbytes = sum(len(t.encode("utf-8")) for t in docs)
    if int(np.load(tb_path)[arr].sum()) != nbytes:
        raise SystemExit(f"data_v3: part {u - 1}'s tokens' bytes are not its documents' bytes")
    out = TRAIN / "parts" / f"{u - 1:03d}.bin"
    (TRAIN / "parts" / f"{u - 1:03d}.json").unlink(missing_ok=True)
    arr.tofile(out.with_name(out.name + ".part"))
    out.with_name(out.name + ".part").rename(out)
    part = {"index": u - 1, "unit": u, "file": rec["file"], "file_sha256": rec["file_sha256"], "documents": len(docs),
            "dclm_documents": n_dclm, "dropped": [], "tokens": int(len(arr)), "text_bytes": nbytes,
            "sha256": sha256(out), "seconds": round(time.time() - t0, 1)}
    (TRAIN / "parts" / f"{u - 1:03d}.json").write_text(json.dumps(part) + "\n")
    read_only(out, TRAIN / "parts" / f"{u - 1:03d}.json")
    Path(rec["local"]).unlink(missing_ok=True)
    return part


# ---- held-out sets ------------------------------------------------------------------------------------------------------

def held_out(rec0: dict, train_hashes: np.ndarray, sizes: dict[str, int], cap: int | None, enc: Encoders,
             tb: np.ndarray) -> dict:
    docs, n_dclm = unit_docs(0, rec0, cap)
    th = np.array([text_hash(t) for t in docs], dtype=np.uint64)
    in_train = np.isin(th, train_hashes)
    seen, kept, dropped = set(), [], []
    for i, t in enumerate(docs):
        if in_train[i] or int(th[i]) in seen:
            dropped.append({"candidate": i, "text_sha256_8": f"{int(th[i]):016x}",
                            "why": "training" if in_train[i] else "earlier candidate"})
        else:
            seen.add(int(th[i]))
            kept.append(t)
    dedup = {"candidates": len(docs), "dclm_candidates": n_dclm, "training_documents": int(len(train_hashes)),
             "rule": "a candidate whose text's sha256 shares its first 8 bytes with a training document's or an earlier "
                     "candidate's is left out", "dropped": dropped}
    out, at = {}, 0
    for name, need in sizes.items():
        take, have = [], 0
        while have < need:
            if at >= len(kept):
                raise SystemExit(f"data_v3: unit 0 has too few documents for the {name} set")
            ids = enc.ids(TOKENIZER, kept[at: at + 256])
            for t, x in zip(kept[at: at + 256], ids):
                if have >= need:
                    break
                take.append((t, x))
                have += len(x) + 1
                at += 1
        out[name] = take
    sets = {}
    for name, take in out.items():
        d, which = (SETS[name][0], SETS[name][2]) if name in SETS else (TRAIN, (TOKENIZER,))
        d.mkdir(parents=True, exist_ok=True)
        texts_ = [t for t, _ in take]
        doc_bytes = [len(t.encode("utf-8")) for t in texts_]
        rec = {"schema": "covert-witness/heldout/v4", "set": name, "data": DATA, "unit": 0, "file": rec0["file"],
               "file_sha256": rec0["file_sha256"], "documents": len(take), "bytes": sum(doc_bytes),
               "dedup": {k: v for k, v in dedup.items() if k != "dropped"} | {"dropped": len(dropped)},
               "doc_bytes": doc_bytes, "tokenizers": {}}
        for w in which:
            ids = [x for _, x in take] if w == TOKENIZER else enc.ids(w, texts_)
            arr = stream(ids, EOT[w])
            fname = "val.bin" if name == "val" else f"{w}.bin"
            (d / fname).unlink(missing_ok=True)
            arr.tofile(d / fname)
            if w == TOKENIZER and int(tb[arr].sum()) != rec["bytes"]:
                raise SystemExit(f"data_v3: the {name} set's bpe32k tokens' bytes are not its documents' bytes")
            rec["tokenizers"][w] = {"tokens": int(len(arr)), "sha256": sha256(d / fname), "doc_tokens": [len(x) + 1 for x in ids]}
        if name == "val":
            v = rec["tokenizers"][TOKENIZER]
            rec = {"index": -1, "file": rec0["file"], "file_sha256": rec0["file_sha256"], "documents": rec["documents"],
                   "dropped": [], "tokens": v["tokens"], "text_bytes": rec["bytes"], "sha256": v["sha256"],
                   "rule": "unit 0's kept documents after the three sets', whole, until VAL_TOKENS"}
            (TRAIN / "val.json").unlink(missing_ok=True)
            (TRAIN / "val.json").write_text(json.dumps(rec) + "\n")
            read_only(TRAIN / "val.bin", TRAIN / "val.json")
        else:
            (d / "manifest.json").unlink(missing_ok=True)
            (d / "manifest.json").write_text(json.dumps(rec) + "\n")
            read_only(d / "manifest.json", *(d / f"{w}.bin" for w in which))
        sets[name] = {k: v for k, v in rec.items() if k not in ("doc_bytes", "tokenizers")} | {
            "dir": str(d), "tokenizers": {t: {k: v for k, v in r.items() if k != "doc_tokens"}
                                         for t, r in rec.get("tokenizers", {}).items()}}
    (BUILD / "dedup.json").write_text(json.dumps(dedup) + "\n")
    return {"dedup": {k: v for k, v in dedup.items() if k != "dropped"} | {"dropped": len(dropped)}, "sets": sets}


# ---- other nodes ---------------------------------------------------------------------------------------------------------

def peer(name: str) -> list[str]:
    for line in PEERS.read_text().splitlines():
        f = line.split()
        if f and f[0] == name:
            return f[1:]
    raise SystemExit(f"data_v3: {name} is not in {PEERS}")


class Pusher(threading.Thread):
    """Copies each batch of paths, in order, to the same paths on every peer (rsync over ssh)."""

    def __init__(self, nodes: list[str]):
        super().__init__(daemon=True)
        self.nodes = {n: peer(n) for n in nodes}
        self.q: queue.Queue = queue.Queue()
        self.error: str | None = None
        self.prepared: set[tuple[str, str]] = set()

    def push(self, *paths: Path) -> None:
        if self.nodes:
            self.q.put([Path(p) for p in paths])

    def run(self) -> None:
        while (paths := self.q.get()) is not None:
            for n, opts in self.nodes.items():
                ssh, dest = opts[:-1], opts[-1]
                for p in paths:
                    if (n, str(p.parent)) not in self.prepared:
                        r = subprocess.run(["ssh", *ssh, "-o", "BatchMode=yes", dest, f"mkdir -p {shlex.quote(str(p.parent))}"])
                        if r.returncode:
                            self.error = f"ssh {n} mkdir {p.parent}: rc {r.returncode}"
                            continue
                        self.prepared.add((n, str(p.parent)))
                    r = subprocess.run(["rsync", "-a", "-e", shlex.join(["ssh", *ssh, "-o", "BatchMode=yes"]),
                                        str(p), f"{dest}:{p}"])
                    if r.returncode:
                        self.error = f"rsync {p} to {n}: rc {r.returncode}"
            self.q.task_done()
        self.q.task_done()

    def finish(self) -> None:
        if self.nodes:
            self.q.put(None)
            self.q.join()
        if self.error:
            raise SystemExit(f"data_v3: pushing to the other nodes failed: {self.error}")


def writable(nodes: list[str]) -> None:
    """Earlier versions' sets may be read-only with their parents (prep.sh)."""
    dirs = [ROOT / "heldout", SEALED / "heldout", ROOT / "train"]
    for d in dirs:
        if d.is_dir():
            d.chmod(d.stat().st_mode | 0o200)
    for n in nodes:
        opts = peer(n)
        cmd = "; ".join(f"[ -d {shlex.quote(str(d))} ] && chmod u+w {shlex.quote(str(d))}" for d in dirs) + "; true"
        subprocess.run(["ssh", *opts[:-1], "-o", "BatchMode=yes", opts[-1], cmd], check=True)


# ---- build ---------------------------------------------------------------------------------------------------------------

def log(**kw) -> None:
    print(json.dumps(kw), flush=True)


def build(a: argparse.Namespace) -> int:
    sel = selection()
    dclm, small = plan(sel, a.units)
    small = small[:a.small_files]
    P, cap = len(dclm), a.cap_lines
    tok_parts = min(a.tokenizer_parts, P - 1)
    sizes = {name: s[1] >> a.scale for name, s in SETS.items()} | {"val": VAL_TOKENS >> a.scale}
    writable(a.push)
    for d in (BUILD / "index", BUILD / "small", BUILD / "raw", TRAIN / "parts"):
        d.mkdir(parents=True, exist_ok=True)
    pusher = Pusher(a.push)
    pusher.start()
    log(units=P, small_files=len(small), cap_lines=cap, sizes=sizes, push=a.push,
        disk_free_gb=round(shutil.disk_usage(ROOT).free / 1e9, 1))
    spawn = get_context("spawn")
    with ProcessPoolExecutor(a.workers, mp_context=spawn) as ix:
        small_f = [ix.submit(index_small, j, f, P, cap) for j, f in enumerate(small)]
        unit_f = [ix.submit(index_unit, u, f, cap) for u, f in enumerate(dclm)]
        small_r = [f.result() for f in small_f]
        log(small=[{k: r[k] for k in ("file", "documents", "text_bytes")} for r in small_r])
        units = {u: unit_f[u].result() for u in range(tok_parts + 1)}
        if TOKENIZER_JSON.is_file():
            tok_sha = sha256(TOKENIZER_JSON)
        else:
            t0 = time.time()
            docs = (t for u in range(1, tok_parts + 1) for t in unit_docs(u, units[u], cap)[0])
            tok_sha = train_tokenizer(docs, TOKENIZER_JSON)
            log(tokenizer=str(TOKENIZER_JSON), sha256=tok_sha, parts=tok_parts, seconds=round(time.time() - t0))
        pusher.push(TOKENIZER_JSON)
        tb = token_bytes()
        tb_path = BUILD / "token_bytes.npy"
        np.save(tb_path, tb)
        with ProcessPoolExecutor(a.part_workers, mp_context=spawn, initializer=_threads,
                                 initargs=(a.part_threads,)) as tx:
            def done(u: int) -> dict | None:
                j, b = TRAIN / "parts" / f"{u - 1:03d}.json", TRAIN / "parts" / f"{u - 1:03d}.bin"
                if j.is_file() and b.is_file():
                    r = json.loads(j.read_text())
                    if r["file_sha256"] == dclm[u]["sha256"] and b.stat().st_size == 2 * r["tokens"]:
                        return r
                return None

            parts, futs = {}, {}
            for u in range(1, P):
                if (r := done(u)) is not None:
                    parts[u] = r
                else:
                    futs[u] = tx.submit(tokenize_part, u, unit_f[u].result(), str(tb_path), cap)
            units = {u: f.result() for u, f in enumerate(unit_f)}
            train_hashes = np.unique(np.concatenate(
                [np.load(BUILD / "index" / f"{u:04d}.npy") for u in range(1, P)]
                + [(lambda x: x[1][x[0] != 0])(np.load(BUILD / "small" / f"{j:03d}.npy")) for j in range(len(small))]))
            if not (TRAIN / "held-out.json").is_file():
                enc = Encoders((TOKENIZER, "gpt2", "pythia"))
                ho = held_out(units[0], train_hashes, sizes, cap, enc, tb)
                ho["pythia_tokenizer"] = {"repo": PYTHIA_REPO, "revision": enc.pythia_revision}
                (TRAIN / "held-out.json").write_text(json.dumps(ho) + "\n")
                for name, (d, _, which) in SETS.items():
                    pusher.push(*[d / f"{w}.bin" for w in which], d / "manifest.json")
                pusher.push(TRAIN / "val.bin", TRAIN / "val.json", TRAIN / "held-out.json")
                log(held_out=ho)
            for u in sorted(futs):
                parts[u] = futs[u].result()
                pusher.push(TRAIN / "parts" / f"{u - 1:03d}.bin", TRAIN / "parts" / f"{u - 1:03d}.json")
                log(part={k: v for k, v in parts[u].items() if k != "dropped"},
                    disk_free_gb=round(shutil.disk_usage(ROOT).free / 1e9, 1))
    ho = json.loads((TRAIN / "held-out.json").read_text())
    recs = [parts[u] for u in range(1, P)]
    tokens = sum(r["tokens"] for r in recs)
    text_bytes = sum(r["text_bytes"] for r in recs)
    pool = ho["sets"]["search-pool"]
    man = {"schema": "te-ladder/stream/v1", "data": DATA, "selection": {"file": SELECTION.name, "sha256": sha256(SELECTION)},
           "dataset": {"repo": REPO, "revision": REVISION}, "tokenizer": TOKENIZER, "vocab": VOCAB,
           "tokenizer_json": {"path": str(TOKENIZER_JSON), "sha256": tok_sha, "parts": list(range(tok_parts))},
           "format": f"uint16, each document prefixed by end-of-text {EOT[TOKENIZER]}", "eot": EOT[TOKENIZER],
           "held_out": ho, "rule": "unit u >= 1 is part u - 1; unit 0 is held out (data_v3.py)",
           "parts": [{k: v for k, v in r.items() if k != "dropped"} | {"dropped": 0} for r in recs],
           "tokens": tokens, "text_bytes": text_bytes,
           "bytes_per_token": {"train": text_bytes / tokens,
                               "search_pool": pool["bytes"] / (pool["tokenizers"][TOKENIZER]["tokens"] - pool["documents"])},
           "val": json.loads((TRAIN / "val.json").read_text())}
    (TRAIN / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    data = {"data": DATA, "dataset": man["dataset"], "selection": man["selection"], "tokenizer": man["tokenizer_json"],
            "train": {"dir": str(TRAIN), "parts": len(recs), "tokens": tokens, "text_bytes": text_bytes},
            "bytes_per_token": man["bytes_per_token"], "pythia_tokenizer": ho["pythia_tokenizer"], "sets": ho["sets"]}
    DATA_JSON.write_text(json.dumps(data, indent=1) + "\n")
    pusher.push(TRAIN / "manifest.json", DATA_JSON)
    pusher.finish()
    if a.manifest_out:
        a.manifest_out.mkdir(parents=True, exist_ok=True)
        shutil.copy(TRAIN / "manifest.json", a.manifest_out / "manifest.json")
        shutil.copy(DATA_JSON, a.manifest_out / "data.json")
    shutil.rmtree(BUILD)
    log(tokens=tokens, parts=len(recs), bytes_per_token=man["bytes_per_token"], tokenizer_sha256=tok_sha,
        disk_free_gb=round(shutil.disk_usage(ROOT).free / 1e9, 1))
    return 0


# ---- verify --------------------------------------------------------------------------------------------------------------

def verify(a: argparse.Namespace) -> int:
    """Every installed file against its record (tokenizer.json, parts, val.bin, the sets), then read-only."""
    data = json.loads(DATA_JSON.read_text())
    man = json.loads((TRAIN / "manifest.json").read_text())
    bad = []
    if sha256(TOKENIZER_JSON) != man["tokenizer_json"]["sha256"]:
        bad.append(str(TOKENIZER_JSON))
    for r in man["parts"] + [man["val"]]:
        p = TRAIN / ("val.bin" if r["index"] == -1 else f"parts/{r['index']:03d}.bin")
        if not p.is_file() or p.stat().st_size != 2 * r["tokens"] or (a.full and sha256(p) != r["sha256"]):
            bad.append(str(p))
    for name, s in data["sets"].items():
        if name == "val":
            continue
        for w, t in s["tokenizers"].items():
            p = Path(s["dir"]) / f"{w}.bin"
            if not p.is_file() or sha256(p) != t["sha256"]:
                bad.append(str(p))
    if bad:
        raise SystemExit(f"data_v3: {len(bad)} files differ from their records: {bad[:6]}")
    for d in [TRAIN, *(Path(s["dir"]) for n, s in data["sets"].items() if n != "val")]:
        subprocess.run(["chmod", "-R", "a-w", str(d)], check=True)
    if a.keep:
        subprocess.run([a.research_python, "-m", "research", "keep", str(TRAIN), "--owner", "@training", "--expires",
                        "never", "--priority", "keep", "--regenerate", "python benchmarks/covert_witness/data_v3.py build",
                        "--why", "covert_witness's v3 training stream and tokenizer"], check=True)
    log(verified={"parts": len(man["parts"]), "tokens": man["tokens"], "sets": sorted(data["sets"]),
                  "sha256_checked": "every file" if a.full else "tokenizer and sets; parts by size"},
        train=str(TRAIN), disk_free_gb=round(shutil.disk_usage(ROOT).free / 1e9, 1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--tree", type=Path, default=None, help="the hub's listing as JSON ({files: [{path, size, lfs}]})")
    b = sub.add_parser("build")
    b.add_argument("--workers", type=int, default=24, help="files fetched and indexed at once")
    b.add_argument("--part-workers", type=int, default=4, help="parts tokenized at once")
    b.add_argument("--part-threads", type=int, default=8, help="tokenizer threads a part")
    b.add_argument("--push", nargs="*", default=[], metavar="NODE", help="copy each output to these nodes (/etc/vy/peers)")
    b.add_argument("--manifest-out", type=Path, default=None)
    b.add_argument("--units", type=int, default=None, help="a smoke run: DCLM's first N units only")
    b.add_argument("--small-files", type=int, default=None, help="a smoke run: the other sources' first N files only")
    b.add_argument("--cap-lines", type=int, default=None, help="a smoke run: each file's first N lines only")
    b.add_argument("--scale", type=int, default=0, help="a smoke run: the sets' sizes divided by 2^N")
    b.add_argument("--tokenizer-parts", type=int, default=TOKENIZER_PARTS)
    v = sub.add_parser("verify")
    v.add_argument("--full", action="store_true", help="hash every part, not only check its size")
    v.add_argument("--keep", action="store_true", help="record the retention of the stream and the search pool")
    v.add_argument("--research-python", default="python3", help="a python with the research package")
    a = ap.parse_args()
    if a.cmd == "select":
        tree = json.loads(a.tree.read_text())["files"] if a.tree else hub_tree()
        SELECTION.write_text(json.dumps(select(tree), indent=1) + "\n")
        print(SELECTION)
        return 0
    return build(a) if a.cmd == "build" else verify(a)


if __name__ == "__main__":
    sys.exit(main())
