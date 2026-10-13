"""Each fused launch the stack's install makes, one at a time with a sync after each, logged before it starts: which one
hangs, and whether each one's receipts, all of them, are sketch_receipt's of its store's blocks (a debugging probe;
not part of the stack)."""

import argparse
import sys
import time
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stack as S  # noqa: E402


def log(*a) -> None:
    print(time.strftime("%H:%M:%SZ", time.gmtime()), *a, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wt-lib", required=True)
    ap.add_argument("--reader-lib", required=True)
    ap.add_argument("--paged-lib", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    st = S.Stack(types.SimpleNamespace(k=a.k, wt_lib=a.wt_lib, reader_lib=a.reader_lib, paged_lib=a.paged_lib), True)
    wt = st.wt
    root_prev = S.RD.digest(st.root)
    st.wt.call("wt_pass", 0, st.root.data_ptr(), st.signs.data_ptr(), S.T, S.stream())
    st.wt.call("wt_sign_frags", st.signs.data_ptr(), st.frags.data_ptr(), S.T, S.stream())
    torch.cuda.synchronize()
    seed = S.BW.pass_seed(wt, 0, root_prev)
    nbits = S.BW.sketch_bits(wt, seed, 16, S.SKT.vector_bytes(wt.sb, S.T))
    log("stack up")
    gen = torch.Generator(device="cuda").manual_seed(1)
    bad = 0
    for name, (n, k, bf) in {"kv": (1024, 4096, False), "lm": (2048, 4096, True)}.items():
        w = (torch.randn((n, k), generator=gen, device="cuda") * 0.1).to(torch.bfloat16 if bf else torch.float8_e4m3fn)
        lin = S.StackLinear(st, name, w, None if bf else torch.ones(1, device="cuda"), bf=bf)
        nblk = lin.nbytes // S.SR.BLOCK
        vs = lin.store.view(torch.uint8).view(-1, S.SR.BLOCK).view(torch.int8)[:nblk]
        sk = S.SKT.sketch_torch(torch, vs, nbits).cpu().numpy()
        want = b"".join(S.BW.root_words(y) for y in S.SR.receipts_of(seed, sk, wt.jbase))
        log(name, "encoded", lin.words, "words", nblk, "blocks")
        for m in S.CAPTURE:
            x8 = S.rand_x(lin, m, gen)
            ref = None
            for cfg in lin.candidates(m):
                plan = lin.planned(m, cfg, {})
                log(name, "m", m, "cfg", cfg, "groups", plan[1], "receipts", plan[3], "...")
                part = torch.empty((lin.nch, m, lin.n), dtype=torch.float32, device="cuda")
                tags = lin.own_tags(plan)
                lin.launch(x8, part, plan, 0, tags)
                torch.cuda.synchronize()
                got = S.BW.host_bytes(tags.view(torch.uint8)[:wt.tag_bytes * plan[3]])
                rok = plan[3] == nblk and got == want
                sok = ref is None or torch.equal(part, ref)
                ref = part if ref is None else ref
                bad += not (rok and sok)
                log(name, "m", m, "cfg", cfg, "receipts_ok", rok, "sums_equal", sok)
    log("done", "bad", bad)


if __name__ == "__main__":
    main()
