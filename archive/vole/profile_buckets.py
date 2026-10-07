"""Bucket a `callgrind_annotate --inclusive=no` listing of the dietmc prover by protocol phase.

Rules are matched on the Rust v0-mangled symbol plus source file; crate and function tokens survive mangling
(`lpn_mtx_indices`, `send_internal`, `homcom`, `text_parser`, ...).  Usage: profile_buckets.py annotate_self.txt
"""

import collections
import re
import sys

RULES = [
    ("sVOLE: LPN sparse-matrix index sampling (AES-RNG + uniform_int rejection)", r"lpn_mtx_indices"),
    ("sVOLE: LPN sparse product y = A u + e (F61p multiply-add in svole send/receive)", r"svole.*(send_internal|receive_internal)"),
    ("sVOLE: GGM trees / spsVOLE / AES-PRG", r"ggm|spsvole|aesrng|aes_rng|4aes|Aes128|aes\.rs"),
    ("init: base OT (Chou-Orlandi, curve25519), KOS OT extension, COPEe, base sVOLE", r"curve25519|chou_orlandi|ot_alsz_kos|3kos|copee|base_svole|ristretto"),
    ("init/field-switch: F2 / F40b side backend", r"F40b|small_binary_fields|field_binary|4F2E"),
    ("parse: SIEVE-IR text reader", r"sieve_ir_parser|text_parser|sieveir_reader|InputText|bufreader|consume_byte|read_exact|from_str_radix|Number"),
    ("online: QuickSilver commits / mult checks / MAC memory (homcom, backend_multifield, memory)",
     r"homcom|MultCheck|backend_multifield|Evaluator|6memory|3mac3Mac|DietMacAndCheese|check_zero|assert_zero|zero_check|input_private"),
    ("field arithmetic F61p (unattributed callers)", r"field_f61p|F61p"),
    ("libc memcpy / alloc", r"memcpy|memmove|memset|malloc|free|realloc|alloc"),
    ("channel I/O", r"std2net|TcpStream|std2io|write_all|flush|Channel|channel"),
]


def main(path: str) -> None:
    total = None
    buckets: collections.Counter[str] = collections.Counter()
    rows: list[tuple[int, str]] = []
    with open(path, errors="replace") as f:
        for line in f:
            m = re.match(r"\s*([\d,]+)\s+(?:\(\s*[\d.]+%\)\s+)?(\S.*)$", line)
            if not m:
                continue
            ir = int(m.group(1).replace(",", ""))
            fn = m.group(2).strip()
            if "PROGRAM TOTALS" in fn:
                total = ir
                continue
            rows.append((ir, fn))
            b = "other"
            for name, pat in RULES:
                if re.search(pat, fn):
                    b = name
                    break
            buckets[b] += ir
    if total is None:
        total = sum(ir for ir, _ in rows) or 1
    print(f"PROGRAM TOTAL Ir = {total:,}  ({len(rows)} functions listed at the 99% threshold)")
    print("\n== buckets by self instructions (first matching rule on file:symbol)")
    for k, v in buckets.most_common():
        print(f"  {100 * v / total:5.1f}%  {k}")
    print("\n== top 25 functions by self Ir (file, symbol)")
    for ir, fn in rows[:25]:
        parts = fn.split(":", 1)
        sym = parts[1] if len(parts) > 1 else ""
        print(f"  {100 * ir / total:5.1f}%  {parts[0].split('/')[-1][:26]:26s} {sym[:120]}")


if __name__ == "__main__":
    main(sys.argv[1])
