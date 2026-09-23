"""Bit-exactness and timing of ``encode_simt.LigeroEncoderSIMT`` against the prover's torch encoder (``field.encode``,
``eval_on_coset`` with the coefficient-form masks) on the same rows, plain and ZK, at the prover's shapes.

    python -m backends.direct.ligero.encode_simt_test --device cuda [--rows 3516] [--out results/encode_simt.json]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .encode_simt import LigeroEncoderSIMT
from .field import P, encode, eval_on_coset, intt
from .protocol import ZK_T_PAD, _add_zh_times


def _time(fn, reps: int, device) -> float:
    fn()
    torch.cuda.synchronize(device)
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - t) / reps


def _timing(l: int, R: int, reps: int, dev, g) -> dict:
    n = 4 * l
    W = torch.randint(0, P, (R, l), generator=g).to(dev)
    S = torch.randint(0, P, (R, ZK_T_PAD), generator=g).to(dev)
    enc = LigeroEncoderSIMT(l, n, 0, dev)
    encz = LigeroEncoderSIMT(l, n, ZK_T_PAD, dev)
    out = torch.empty((R, n), dtype=torch.int32, device=dev)
    t_torch = _time(lambda: encode(W, n), reps, dev)
    t_simt = _time(lambda: enc.encode(W, out=out), reps, dev)
    t_simt_c = _time(lambda: enc.encode(W, out=out, want_coefs=True), reps, dev)

    def torch_zk():
        coefs = torch.zeros((R, l + ZK_T_PAD), dtype=torch.int64, device=dev)
        coefs[:, :l] = intt(W)
        _add_zh_times(coefs, S, l)
        return eval_on_coset(coefs, n)
    t_torch_zk = _time(torch_zk, reps, dev)
    t_simt_zk = _time(lambda: encz.encode(W, S, out=out, want_coefs=True), reps, dev)
    gb = R * n * 4 / 1e9
    print(f"l={l}: {R} rows -> {n}: torch int64 {t_torch*1e3:.1f} ms | SIMT {t_simt*1e3:.2f} ms ({gb/t_simt:.0f} GB/s of codeword) "
          f"| SIMT+coefs {t_simt_c*1e3:.2f} | ZK: torch {t_torch_zk*1e3:.1f} ms, SIMT {t_simt_zk*1e3:.2f} ms")
    return {"rows": R, "n": n, "torch_int64_ms": t_torch * 1e3, "simt_ms": t_simt * 1e3,
            "simt_with_coefs_ms": t_simt_c * 1e3, "torch_zk_ms": t_torch_zk * 1e3, "simt_zk_ms": t_simt_zk * 1e3,
            "codeword_GB": gb, "simt_GBps_codeword": gb / t_simt}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rows", type=int, default=3516)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    dev = torch.device(args.device)
    g = torch.Generator(device="cpu").manual_seed(1)
    report = {"exact": {}, "timing": {}}
    ok_all = True
    for l in (256, 4096, 8192, 16384):
        n = 4 * l
        R = 64
        rows = torch.randint(0, P, (R, l), generator=g).to(dev)
        mask = torch.randint(0, P, (R, ZK_T_PAD), generator=g).to(dev)
        # plain
        enc = LigeroEncoderSIMT(l, n, 0, dev)
        U, C = enc.encode(rows, want_coefs=True)
        ref = encode(rows, n)
        ok1 = torch.equal(U.to(torch.int64), ref)
        ok2 = torch.equal(C.to(torch.int64), intt(rows))
        # ZK: interp_H(rows) + Z_H mask, evaluated on the coset
        encz = LigeroEncoderSIMT(l, n, ZK_T_PAD, dev)
        Uz, Cz = encz.encode(rows, mask, want_coefs=True)
        coefs = torch.zeros((R, l + ZK_T_PAD), dtype=torch.int64, device=dev)
        coefs[:, :l] = intt(rows)
        _add_zh_times(coefs, mask, l)
        refz = eval_on_coset(coefs, n)
        ok3 = torch.equal(Uz.to(torch.int64), refz)
        ok4 = torch.equal(Cz.to(torch.int64), intt(rows))
        report["exact"][l] = {"plain_codeword": ok1, "plain_coefs": ok2, "zk_codeword": ok3, "zk_coefs": ok4, "rows": R}
        ok_all &= ok1 and ok2 and ok3 and ok4
        print(f"l={l:6d} n={n:6d}: plain {ok1} coefs {ok2} | zk (mask fold-in) {ok3} coefs {ok4}")
    # timing at the prover's shape
    for l in (8192, 16384):
        report["timing"][l] = _timing(l, args.rows, args.reps, dev, g)
    report["hardware"] = torch.cuda.get_device_name(dev)
    report["verdict"] = "PASS" if ok_all else "FAIL"
    print("bit-exact:", report["verdict"])
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
    return 0 if ok_all else 2


if __name__ == "__main__":
    raise SystemExit(main())
