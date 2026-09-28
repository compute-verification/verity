#!/usr/bin/env python3
"""The `.FTZ` gate on Pearl-C's shipped SASS (build.sh): no FP32 FADD, FFMA or FMUL with `.FTZ`, except inside nvcc's correctly
rounded divide, reciprocal and square root, matched as whole sequences against the ones the pinned test kernels (pearl_c.cu's
`ieee_ops` and `ieee_q`) produce, under an nvcc and flags whose sequences `ftz_pins.json` pins.  Any other nvcc or flags fail
closed.  run.py's check holds the test kernels' words to the correctly rounded values on subnormal inputs and outputs.

* **A sequence** is nvcc's out-of-line slow path (nvdisasm names it `__cuda_sm…_slowpath`), or the inline fast path that goes
  with it: the divergent block holding its `.FTZ` ops, inside a convergence region (BSSY … BSYNC) that also calls that slow path.
  So a fast path counts only together with its slow-path call.
* **Sequences are compared by their skeleton:** every F*, MUFU and control instruction in order, with its opcode, modifiers,
  immediates and guard polarity, and with registers erased.  ptxas picks registers and equivalent integer and move forms per
  kernel; the floating-point work is the routine.

A pin is keyed by the build: 'nvcc version/arch flags', as build.sh passes it (the flags carry no FTZ or fast math).

  python3 ftz_gate.py NVDISASM_TEXT BUILD        exit 1 with the problems on stderr
  python3 ftz_gate.py NVDISASM_TEXT BUILD --pin  print the test kernels' sequences as ftz_pins.json's entry for BUILD
"""

from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PINS = os.path.join(HERE, "ftz_pins.json")
TEST_KERNELS = ("ieee_ops", "ieee_q")
GATED = ("FADD", "FFMA", "FMUL")
CONTROL = ("BRA", "CALL", "RET", "BSSY", "BSYNC", "EXIT")

_LABEL = re.compile(r"^(\S+):\s*$")
_INS = re.compile(r"^\s+/\*[0-9a-f]{4,}\*/\s+(.*?)\s*;\s*(?:/\*.*)?$")
_SPLIT = re.compile(r"^(@!?U?P[0-9T]\s+)?(\S+)\s*(.*)$")
_SLOW = re.compile(r"^\$__internal_\d+_\$(\w+)$")


def functions(text: str) -> dict[str, dict[str, list[str]]]:
    """{kernel: {function label: instructions}}: nvdisasm's text sections, split at the non-local labels (the kernel, its slow
    paths and device functions).  A local label stays in the list as `.L_x_N:`."""
    out: dict[str, dict[str, list[str]]] = {}
    kernel = label = None
    for line in text.splitlines():
        if line.startswith("\t.section\t.text."):
            kernel = line.split(".text.", 1)[1].split(",")[0].strip('"')
            out[kernel], label = {}, None
            continue
        m = _LABEL.match(line)
        if m and kernel is not None:
            name = m.group(1)
            if name.startswith(".L_"):
                if label is not None:
                    out[kernel][label].append(name + ":")
            elif not name.startswith(".text"):
                label = name
                out[kernel][label] = []
            continue
        m = _INS.match(line)
        if m and kernel is not None and label is not None:
            out[kernel][label].append(m.group(1))
    return out


def split(ins: str) -> tuple[str, str, str]:
    """(guard, opcode with modifiers, operands) of one instruction."""
    m = _SPLIT.match(ins.replace(".reuse", ""))
    return (m.group(1) or "").strip(), m.group(2), m.group(3)


def is_label(ins: str) -> bool:
    return ins.endswith(":")


def gated_ftz(ins: str) -> bool:
    if is_label(ins):
        return False
    op = split(ins)[1].split(".")
    return op[0] in GATED and "FTZ" in op[1:]


def skeleton(ins: list[str]) -> tuple[str, ...]:
    out = []
    for s in ins:
        if is_label(s):
            continue
        guard, op, args = split(s)
        base = op.split(".")[0]
        if not (base.startswith("F") or base == "MUFU" or base in CONTROL):
            continue
        args = re.sub(r"`\([^)]*\)|\.L_x_\d+", "L", args)
        args = re.sub(r"\bU?R(?:\d+|Z)\b", "r", args)
        args = re.sub(r"\bU?P(?:\d|T)\b", "p", args)
        args = re.sub(r"\bB\d+\b", "b", args)
        g = "" if not guard else ("@!p " if guard.startswith("@!") else "@p ")
        out.append((g + op + " " + args).strip())
    return tuple(out)


def _fast(body: list[str], i: int) -> tuple[tuple[str, ...], str | None]:
    """The divergent block holding instruction i and the slow path its convergence region calls, or (the block, None)."""
    a = i
    while a > 0 and not is_label(body[a - 1]) and split(body[a - 1])[1].split(".")[0] not in CONTROL:
        a -= 1
    b = i
    while b < len(body) and not is_label(body[b]) and split(body[b])[1].split(".")[0] not in CONTROL:
        b += 1
    block = skeleton(body[a:b])
    while b < len(body) and is_label(body[b]):
        b += 1
    if b == len(body) or split(body[b])[1] != "BSYNC":
        return block, None
    bar = split(body[b])[2]
    j = a - 1
    while j >= 0 and not (split(body[j])[1] == "BSSY" and split(body[j])[2].startswith(bar + ",")):
        j -= 1
    called = [m.group(1) for s in body[max(j, 0):b] if "CALL" in s for m in [re.search(r"\$__internal_\d+_\$(\w+)", s)] if m]
    return block, called[0] if j >= 0 and len(set(called)) == 1 else None


def sequences(fns: dict[str, list[str]], kernel: str) -> tuple[dict[str, set], dict[str, set], list[str]]:
    """One kernel's slow-path and fast-path skeletons that hold gated `.FTZ` ops, by slow path; and every gated `.FTZ` op that is
    in neither."""
    slow: dict[str, set] = {}
    fast: dict[str, set] = {}
    problems = []
    for label, body in fns.items():
        m = _SLOW.match(label)
        if m:
            if any(gated_ftz(s) for s in body):
                slow.setdefault(m.group(1), set()).add(skeleton(body))
            continue
        for i, s in enumerate(body):
            if gated_ftz(s):
                block, called = _fast(body, i)
                if called is None:
                    problems.append(f"{kernel}: `{s}` is outside nvcc's correctly rounded div, rcp and sqrt sequences")
                else:
                    fast.setdefault(called, set()).add(block)
    return slow, fast, problems


def load_pins(path: str = PINS) -> dict:
    """ftz_pins.json's entries, by 'nvcc version/arch flags' (keys starting with _ are notes)."""
    return {k: v for k, v in json.load(open(path)).items() if not k.startswith("_")}


def pinned_of(entry: dict) -> tuple[dict[str, set], dict[str, set]]:
    return ({k: {tuple(s) for s in v} for k, v in entry["slow"].items()},
            {k: {tuple(s) for s in v} for k, v in entry["fast"].items()})


def check(text: str, nvcc: str, pins: dict) -> list[str]:
    """The gate's problems for one cubin's nvdisasm text built as `nvcc` ('V…/arch flags'); empty when it passes."""
    kernels = functions(text)
    found = {k: sequences(fns, k) for k, fns in kernels.items()}
    problems = [p for _, _, ps in found.values() for p in ps]
    uses = any(slow or fast for slow, fast, _ in found.values())
    tests = [k for k in TEST_KERNELS if k in kernels]
    if not uses and not tests:
        return problems
    if nvcc not in pins:
        return problems + [f"nvcc {nvcc}: no pinned IEEE sequences, so no .FTZ op is admitted (pinned: {sorted(pins)})"]
    slow_pin, fast_pin = pinned_of(pins[nvcc])
    if len(tests) != len(TEST_KERNELS):
        problems.append(f"the pinned test kernels {TEST_KERNELS} are not all in this cubin")
    produced_slow: dict[str, set] = {}
    produced_fast: dict[str, set] = {}
    for k in tests:
        for d, out in ((found[k][0], produced_slow), (found[k][1], produced_fast)):
            for name, sks in d.items():
                out.setdefault(name, set()).update(sks)
    if (produced_slow, produced_fast) != (slow_pin, fast_pin):
        problems.append(f"the test kernels' IEEE sequences under nvcc {nvcc} are not ftz_pins.json's")
    for k, (slow, fast, _) in found.items():
        for kind, d, pin in (("slow path", slow, slow_pin), ("fast path", fast, fast_pin)):
            for name, sks in d.items():
                for sk in sks - pin.get(name, set()):
                    problems.append(f"{k}: a {name} {kind} with .FTZ ops that no test kernel pins ({len(sk)} ops)")
    return problems


def pin_entry(text: str) -> dict:
    kernels = functions(text)
    slow: dict[str, set] = {}
    fast: dict[str, set] = {}
    for k in TEST_KERNELS:
        s, f, _ = sequences(kernels[k], k)
        for d, out in ((s, slow), (f, fast)):
            for name, sks in d.items():
                out.setdefault(name, set()).update(sks)
    return {"slow": {k: sorted(map(list, v)) for k, v in sorted(slow.items())},
            "fast": {k: sorted(map(list, v)) for k, v in sorted(fast.items())}}


def main(argv: list[str]) -> int:
    text, nvcc = open(argv[1]).read(), argv[2]
    if "--pin" in argv:
        print(json.dumps({nvcc: pin_entry(text)}, indent=1))
        return 0
    problems = check(text, nvcc, load_pins())
    for p in problems:
        print(p, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
