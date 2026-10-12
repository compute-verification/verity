"""The design's link term by batch: mu_link = (L / sum R) * |E| / min(|z|, |E|).

L is the throughput of the paths that write neither HBM nor host DRAM on the way to memory outside the counted tiers,
sum R the node's HBM read capacity, and |z|/|E| the bytes that must cross per byte of block skipped:
- tile-output binding (Pearl-C tiles, the tile's 4-byte outputs bound): |z|/|E| = 4m/k;
- k-slice binding (the running accumulator after each slice of width k_s): |z|/|E| = 4m/k_s;
- a construction that binds |z| = c bytes per block of |E| bytes: |z|/|E| = c/|E| (--fixed BYTES:BLOCK).
A value of 1 or more means the relay covers every block it skips: no bound from this term.

    python3 mu_link.py --link-gbps 50 --link-gbps 46.5 [--k 4096] [--ks 32 64 128] [--fixed 64:4096] [--batches ...]
"""

import argparse

SUM_R = 8 * 1308.5  # GB/s: node 2's eight GPUs at the measured HBM read rate


def ratio(z_over_e):
    return 1.0 / min(z_over_e, 1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--link-gbps", type=float, action="append", required=True)
    p.add_argument("--k", type=int, default=4096)
    p.add_argument("--ks", type=int, nargs="*", default=[32, 64, 128])
    p.add_argument("--fixed", action="append", default=[], help="BYTES:BLOCK, a fixed |z| per block")
    p.add_argument("--batches", type=int, nargs="*", default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    a = p.parse_args()
    cols = [("tile k=%d" % a.k, lambda m: 4 * m / a.k)]
    cols += [("k-slice k_s=%d" % ks, lambda m, ks=ks: 4 * m / ks) for ks in a.ks]
    for f in a.fixed:
        c, blk = (int(x) for x in f.split(":"))
        cols.append(("fixed %d B per %d B" % (c, blk), lambda m, c=c, blk=blk: c / blk))
    for L in a.link_gbps:
        base = L / SUM_R
        print(f"\nL = {L:g} GB/s, L / sum R = {100 * base:.3f}% (sum R = {SUM_R:g} GB/s)\n")
        print("| m | " + " | ".join(n for n, _ in cols) + " |")
        print("|---" * (len(cols) + 1) + "|")
        for m in a.batches:
            cells = []
            for _, f in cols:
                mu = base * ratio(f(m))
                cells.append("no bound (%.0f%%)" % (100 * mu) if mu >= 1 else "%.2f%%" % (100 * mu))
            print(f"| {m} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
