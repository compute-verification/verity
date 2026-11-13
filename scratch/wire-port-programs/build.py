"""Four small two-stage programs, as descriptor text, for the second stage's refusal of a wire whose reader port a
link names too. Scratch: never merged.

Each program is a population of two outer blocks (replay blocks). An outer block calls the template `CoreSpotPair_v0`
on its parameter `x`, calls a second template on that call's output, and returns the second template's output. The
four differ only in the second template's parameter names:

* `plain-chain`: one parameter, `y`. The wire's reader port is `(CoreSpotNamed_v0, y)`, which no link names.
* `o0-chain`: one parameter, `o0`, the name of the template's first output field. The link its return makes and the
  wire both name `(CoreSpotNamed_v0, o0)`.
* `plain-both`: two parameters, `a` and `b`, the first reading the outer block's `w`, the second the call's output.
  The link names `(CoreSpotBoth_v0, a)` and the wire `(CoreSpotBoth_v0, b)`.
* `same-both`: two parameters, both `a`. The link and the wire both name `(CoreSpotBoth_v0, a)`.

Run from the repository's root with the workspace's Python:

    python scratch/wire-port-programs/build.py

It writes each descriptor beside itself as `<label>.json` (`codec.encode_program`'s descriptor, written by
`json.dumps`), and `cases.json`: for each label the outer block's descriptor id (`replay`), the links and the wires
`second_stage.replay_links` and `second_stage.replay_wires` derive (the links at no boundary record), and the wires
whose reader port a link names too (`clash`).
"""

from __future__ import annotations

import json
from pathlib import Path

from verity.circuits import codec
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.population import population_program
from verity.circuits.types import Array, Value
from verity.protocols.spot.verifier import second_stage

HERE = Path(__file__).resolve().parent

XOR8 = PrimitiveDefinition("CoreSpotXor8", 1, (("a", Value(8)), ("b", Value(8))), Value(8), lambda a, b: a ^ b,
                           register=False)
PAIR = bind(CompositeDefinition("CoreSpotPair", 0, (), lambda S: ((("x", Array(2, Value(8))),), Value(8)),
                                lambda B, S, x: B.call(XOR8, x[0], x[1]), register=False))


def named(p: str):
    """A template of one parameter, named `p`."""
    return bind(CompositeDefinition("CoreSpotNamed", 0, (), lambda S: (((p, Value(8)),), Value(8)),
                                    lambda B, S, y: B.call(XOR8, y, y), register=False))


def both(p: str, q: str):
    """A template of two parameters, named `p` and `q`."""
    return bind(CompositeDefinition("CoreSpotBoth", 0, (), lambda S: (((p, Value(8)), (q, Value(8))), Value(8)),
                                    lambda B, S, a, b: B.call(XOR8, a, b), register=False))


def chain(t):
    """An outer block: `t` over PAIR's output."""
    return bind(CompositeDefinition("CoreSpotReplayNamed", 0, (), lambda S: ((("x", Array(2, Value(8))),), Value(8)),
                                    lambda B, S, x: B.call(t, B.call(PAIR, x)), register=False))


def beside(t):
    """An outer block: `t` over its parameter `w` and PAIR's output."""
    return bind(CompositeDefinition("CoreSpotReplayBoth", 0, (),
                                    lambda S: ((("x", Array(2, Value(8))), ("w", Value(8))), Value(8)),
                                    lambda B, S, x, w: B.call(t, w, B.call(PAIR, x)), register=False))


CASES = (("plain-chain", lambda: chain(named("y"))), ("o0-chain", lambda: chain(named("o0"))),
         ("plain-both", lambda: beside(both("a", "b"))), ("same-both", lambda: beside(both("a", "a"))))


def main() -> None:
    cases = {}
    for label, make in CASES:
        replay = make()
        program = population_program(replay, 2)
        key = codec.descriptor_id(replay)
        (HERE / f"{label}.json").write_text(json.dumps(codec.encode_program(program)) + "\n")
        links = second_stage.replay_links(program, key, None)
        wires = second_stage.replay_wires(program, key)
        named_ports = {(x.template, x.port) for x in links or ()}
        cases[label] = {
            "replay": key,
            "links": None if links is None else [{"template": x.template, "port": x.port, "value": x.value}
                                                 for x in links],
            "wires": None if wires is None else [{"template": w.template, "port": w.port, "writer": w.writer,
                                                  "wport": w.wport} for w in wires],
            "clash": [{"template": w.template, "port": w.port, "writer": w.writer, "wport": w.wport}
                      for w in wires or () if (w.template, w.port) in named_ports],
        }
        print(f"{label}: replay {key}; {len(links or ())} links, {len(wires or ())} wires, "
              f"{len(cases[label]['clash'])} of the wires at a port a link names")
    (HERE / "cases.json").write_text(json.dumps(cases, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
