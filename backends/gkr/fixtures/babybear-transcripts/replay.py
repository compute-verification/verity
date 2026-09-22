"""Replay a transcript vector with nothing but hashlib: re-derives every state and every challenge from the absorbed
bytes and asserts equality with the file (the conventions of README.md, executable).

    python3 replay.py transcript_vu4.json transcript_vu64.json
"""

import hashlib
import json
import sys


def replay(path: str) -> None:
    t = json.load(open(path))
    p, deg, ev = t["modulus"], t["degree"], t["events"]
    state = hashlib.sha256(bytes.fromhex(ev[0]["absorb"])).digest()  # state_0 = sha256(domain)
    assert state.hex() == ev[0]["state"], "domain"
    ctr = n_abs = n_ch = 0
    for i, e in enumerate(ev[1:], 1):
        if "absorb" in e:
            state = hashlib.sha256(state + bytes.fromhex(e["absorb"])).digest()
            assert state.hex() == e["state"], f"event {i}: state"
            n_abs += 1
        elif "challenge" in e:
            assert e["ctr"] == ctr, f"event {i}: ctr {ctr} != {e['ctr']}"
            coords: list[int] = []
            while len(coords) < deg:
                d = hashlib.sha256(state + b"chal" + ctr.to_bytes(8, "little")).digest()
                ctr += 1
                for k in range(8):  # eight 4-byte LE words; the first < p is the coordinate
                    v = int.from_bytes(d[4 * k : 4 * k + 4], "little")
                    if v < p:
                        coords.append(v)
                        break
            assert coords == e["challenge"], f"event {i}: challenge"
            n_ch += 1
    print(f"{path}: {n_abs} absorbs, {n_ch} challenges, {len(t['proof_msgs'])} proof messages replayed; root {t['root'][:16]}...")


if __name__ == "__main__":
    for f in sys.argv[1:] or ["transcript_vu4.json", "transcript_vu64.json"]:
        replay(f)
