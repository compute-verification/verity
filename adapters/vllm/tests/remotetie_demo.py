"""Which record can remote's tie name? A toy H1 window, checked with this tree's own Python references.

`Core.Remote` ties the inference-only policy, the PoUW period and the one-stage audit to one commitment, the digest
(`registration.digest`: SHA-512 of the format, a zero byte and the canonical record) of one `verity/registration/v2`
record, and every PoUW period window must name that record as its Commit record (`Credit.commitOf?`). So one record must
pass both readers that look inside it:

* credit's ROOTS1 (`credit._arow`) wants the Commit record to register `row-table/<s>/k<K>` and `input-ids/<s>`, each
  with the domain, root and leaves of the window's own `<prefix>a/<K>@<s>` and `<prefix>ids@<s>`;
* one-stage's R1-R7 (`registration.check`) wants exactly the 14 v2 keys and the audited Program, and each registered
  value's domain under that Program and that value's own name (`registered.domain`).

The toy window: one call of one linear, `q`, at K = 2048 with m = 4, and one round, its tables registered under the H1
window Program's domain and the window's open root. The Program digests and roots are stand-ins; the checks are the
references themselves.

1. H1's Commit record, as `CommitTap.record` writes it (H1's own tables, renamed): ROOTS1 passes it (`None`), and R1-R7
   refuse it at `keys`, since it has 5 of the 14 v2 keys. The record credit reads is not a one-stage registration.
2. V3c's registration, over the I2 window circuit's population Program, registering the weights and the scan state, as
   the Commit record: ROOTS1 refuses it (`arow`: it registers no `row-table/0/k2048`). The record one-stage audits doesn't
   carry the tables credit reads.
3. Could a one-stage record carry the entry ROOTS1 wants? R7's own domain for `row-table/0/k2048` binds the audited
   Program and the value's name, so it is never H1's `a/2048@0` domain, under I2's Program or even under H1's: both
   lines print `False`, and no record that registers the value afresh passes both.

As built, no record passes both, so remote's tie can't hold for the one-stage case.

After the fix, the audited circuit reads the window's A tables and input ids. Its registration registers
`row-table/<s>/k<K>` and `input-ids/<s>` as output values that reuse the entries of H1's window registration, which is in
the one-stage log (`outputs`: `{"row-table/0/k2048": {"record": <its digest>, "value": "pouw/h1/a/2048@0"}, ...}`). R7
then takes the earlier entry's domain and root in place of its own (`registration._check_registered`, Lean
`Registration.applyOutput`), and that registration is the window's Commit record. Passing looks like: ROOTS1 on it prints
`None`; `registration.check` on it, with the audit's own verifier copies and those `outputs`, prints `[]`; and the
policy's and one-stage's commitment digests equal its digest. Checks 1-3 keep printing what they print now, which is why
`outputs` is the one way through that neither loosens the tie nor adds a verifier form.

Run from the tree's root: PYTHONPATH=. python3 adapters/vllm/tests/remotetie_demo.py
"""
import hashlib

from verity.protocols.pouw.verifier import credit as C
from verity.protocols.pouw.verifier import weights as W
from verity.protocols.spot.verifier.one_stage import registered as G
from verity.protocols.spot.verifier.one_stage import registration as R

H1 = hashlib.sha512(b"the H1 window Program's descriptor").hexdigest()
I2 = hashlib.sha512(b"V3c's I2 window circuit's population Program").hexdigest()
OPEN = hashlib.sha512(b"the H1 window's open root").hexdigest()
PREFIX, K, M = "pouw/h1/", 2048, 4
layout = {"calls": [["q", 0]], "linears": [["q", K, 64]], "m": M, "tile": [4, 64]}
rows = W.chunks(K) * M


def entry(program, value, leaves, words=1):
    e = {"value": value, "leaves": leaves, "words": words, "root": "ab" * 64}
    return {**e, "domain": C.domain_of(program, OPEN, e)}


# the H1 window's own registered A table and input-ids table (credit's `entries`), domains under the H1 Program
h1_a, h1_ids = entry(H1, f"{PREFIX}a/{K}@0", rows), entry(H1, f"{PREFIX}ids@0", 8)
entries = [h1_a, h1_ids]

h1_commit = {"format": R.FORMAT_V2, "protocol": R.PROTOCOL, "program": H1, "window": {"kind": "served", "refused": []},
             "registered": [{**h1_a, "value": C.commit_rows(0, K)}, {**h1_ids, "value": C.commit_ids(0)}]}
table = "cd" * 64
rounds = [{"request_table": table, "requests": C.requests_root(bytes.fromhex(table), bytes.fromhex(h1_ids["root"])).hex()}]
print("1. credit ROOTS1 (_arow) on H1's Commit record:", C._arow(rounds, entries, h1_commit["registered"], layout, PREFIX))
v = R.check(h1_commit, None, {}, {}, "", {}, registered={C.commit_rows(0, K): 1, C.commit_ids(0): 1}, window_root=OPEN)
print("   one-stage R1-R7 on H1's Commit record:", list(v.codes), "missing", v.detail["keys"])

v3c_registered = [entry(I2, "weights/q@1", 64), entry(I2, "scan/state", 16)]
print("2. credit ROOTS1 (_arow) on V3c's record:", C._arow(rounds, entries, v3c_registered, layout, PREFIX))

for program in (I2, H1):
    r7 = G.domain(program, C.commit_rows(0, K), 1, rows, window=OPEN).domain_id.hex()
    print(f"3. R7's domain for row-table/0/k{K} under {'I2' if program == I2 else 'H1'}'s Program == ROOTS1's:",
          r7 == h1_a["domain"])

print("   the three readers' digest of H1's Commit record:", R.digest(h1_commit)[:16], "...;",
      "of a V3c-shaped record:", R.digest({**h1_commit, "program": I2, "registered": v3c_registered})[:16], "...")
