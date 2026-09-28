"""The state game for the key chain (POUS `internal/p3-instantiation/overwrite-state-game.py`, verbatim but for this
paragraph): three chaining modes, on the pilot's dense rows and on P3's band rows.

Rows: row i absorbs its parents newest first (dense: i - 1, ..., 0; band k: i - 1, ..., i - min(i, k)), then the pad.
The pad call's output is the key. Dense: the label is key xor W, known with the key. Band (P3): the label is
P(x xor key), one more round. One round = one parallel batch of permutation calls. Every label except a missing pair
{a, b} is stored. An item is "worth two labels" if with it both a and b are known within D rounds, while storing either
one of them instead (plus recomputation) does not give the other. Items are priced in label units at l = 8192,
c = 512: an (l + c)-bit item costs 1.06, a (2l + c)-bit item 2.06.

sponge  (the pilot's) s_j = Pi(s_{j-1} xor (blk_j || 0)), width l + c. Items: states s_(i,j); pre-images
        x_(i,j) = s_(i,j-1) xor blk_j. Moves: forward, backward (Pi^-1), extraction of blk_j from s_(i,j-1) and s_(i,j),
        blk_j = x_(i,j) xor s_(i,j-1) (no call), and x_(i,j) -> s_(i,j) in one call.
mp      Miyaguchi-Preneel s_j = Pi(x_j) xor x_j, x_j = s_{j-1} xor (blk_j || 0), width l + c. Same items. Moves:
        forward, x_(i,j) -> s_(i,j) in one call, blk_j = x_(i,j) xor s_(i,j-1). No backward or two-state extraction
        (x -> Pi(x) xor x is one-way).
over    overwrite chain (r_j, h_j) = Pi2(blk_j || h_{j-1}), width 2l + c; h_j (l + c bits) is chained and r_j dropped.
        Items: chain values h_(i,j) (1.06); Feistel half-states of call (i,j), a masked copy of blk_j that needs
        h_(i,j-1) (1.03); full inputs or Feistel states of call (i,j) (2.06: blk_j, h_(i,j-1), and h_(i,j) one call
        later). Moves: forward only.
"""
import itertools
import sys

W1, WH, W2 = 1 + 512 / 8192, 1 + 256 / 8192, 2 + 512 / 8192


def parents(i, k):
    return [i - j for j in range(1, (i if k is None else min(i, k)) + 1)]


def run(mode, B, k, D, lab, item):
    par = [parents(i, k) for i in range(B)]
    ln = [len(p) + 1 for p in par]              # calls per row, pad included; bit ln[i] = key known
    rows = [1] * B
    kind, ii, jj = item if item else (None, -1, -1)
    if kind in ("s", "h"):
        rows[ii] |= 1 << jj
    if kind == "full":
        rows[ii] |= 1 << (jj - 1)
        lab |= 1 << par[ii][jj - 1]
    band = k is not None
    keyed = [False] * B

    def blk(i, j, lab):
        return j == ln[i] or (lab >> par[i][j - 1]) & 1

    def settle(rows, lab):
        if kind in ("x", "half") and (rows[ii] >> (jj - 1)) & 1:
            lab |= 1 << par[ii][jj - 1]
        return lab

    lab = settle(rows, lab)
    if not band:
        for i in range(B):
            if (rows[i] >> ln[i]) & 1:
                lab |= 1 << i
    for _ in range(D):
        new_rows, new_lab = list(rows), lab
        if kind in ("x", "full"):
            new_rows[ii] |= 1 << jj
        for i in range(B):
            if band and keyed[i]:
                new_lab |= 1 << i                 # label = P(x xor key): one round after the key
            A = rows[i]
            for j in range(1, ln[i] + 1):
                bk = blk(i, j, lab)
                if (A >> (j - 1)) & 1 and bk:
                    new_rows[i] |= 1 << j
                if mode == "sponge":
                    if (A >> j) & 1 and bk:
                        new_rows[i] |= 1 << (j - 1)
                    if j < ln[i] and (A >> j) & 1 and (A >> (j - 1)) & 1:
                        new_lab |= 1 << par[i][j - 1]
        rows = new_rows
        for i in range(B):
            if (rows[i] >> ln[i]) & 1:
                if band:
                    keyed[i] = True
                else:
                    new_lab |= 1 << i
        lab = settle(rows, new_lab)
    return lab


def items(mode, B, k):
    out = []
    for i in range(B):
        for j in range(1, len(parents(i, k)) + 1):
            if mode in ("sponge", "mp"):
                out += [("s", i, j, W1), ("x", i, j, W1)]
            else:
                out += [("h", i, j, W1), ("half", i, j, WH), ("full", i, j, W2)]
    return out


def check(mode, B, k, D):
    allL = (1 << B) - 1
    best = None
    its = items(mode, B, k)
    for a, b in itertools.combinations(range(B), 2):
        known = allL & ~(1 << a) & ~(1 << b)
        if (run(mode, B, k, D, known | (1 << a), None) >> b) & 1 or \
           (run(mode, B, k, D, known | (1 << b), None) >> a) & 1:
            continue
        for (kd, i, j, w) in its:
            lab = run(mode, B, k, D, known, (kd, i, j))
            if (lab >> a) & 1 and (lab >> b) & 1 and (best is None or w < best[0]):
                best = (w, (a, b), (kd, i, j))
    return best
