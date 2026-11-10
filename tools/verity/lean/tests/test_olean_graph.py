"""`olean_graph.differs`, the sampled recompile's comparison of a module's `.olean` parts (`check.sampled_recompile`): on
regions written here as Lean's compactor lays them out, and with a toolchain, on a module built at two paths, where only
the path of its source in a message differs, and with one declaration or attribute changed, which it refuses."""
from __future__ import annotations

import importlib.util
import os
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lean_olean_graph", HERE.parent / "olean_graph.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)

HEADER = b"olean\x02\x01" + b"4.34.0".ljust(33, b"\0") + b"0" * 40
SOURCE = "P/M.lean"


class Region:
    """One part as the compactor writes it: the header, its base address and root, then each object 8-byte aligned at its
    address, the base plus its offset."""

    def __init__(self, base: int, header: bytes = HEADER):
        self.base, self.header, self.body = base, header, bytearray()

    def _put(self, b: bytes) -> int:
        at = self.base + G.DATA + len(self.body)
        self.body += b + bytes(-len(b) % 8)
        return at

    def string(self, s: str, length: int | None = None) -> int:
        data = s.encode() + b"\0"
        return self._put(struct.pack("<iHBBQQQ", 0, 1, 0, G.STRING, len(data), len(data),
                                     len(s) if length is None else length) + data)

    def ctor(self, tag: int, fields: list[int], scalars: bytes = b"") -> int:
        return self._put(struct.pack("<iHBB", 0, 8 + 8 * len(fields) + len(scalars), len(fields), tag)
                         + b"".join(struct.pack("<Q", f) for f in fields) + scalars)

    def array(self, elems: list[int]) -> int:
        return self._put(struct.pack("<iHBBQQ", 0, 1, 0, G.ARRAY, len(elems), len(elems))
                         + b"".join(struct.pack("<Q", e) for e in elems))

    def write(self, f: Path, root: int) -> Path:
        f.write_bytes(self.header + struct.pack("<QQ", self.base, root) + self.body)
        return f


def module(d: Path, path: str = "/a/P/M.lean", text: str = "unused variable", scalar: bytes = b"\x01\0\0\0\0\0\0\0",
           header: bytes = HEADER, stray: int | None = None, lengths: dict[str, int] | None = None,
           bases: tuple[int, int, int] = (0x1000_0000, 0x2000_0000, 0x3000_0000)) -> dict[str, Path]:
    """A module's three parts, mapped at ``bases``: the public one names it, the server part holds a message (its source's
    ``path``, its ``text``, each of the length ``lengths`` gives, else its own) that points at the public name, and the
    private part a declaration with ``scalar`` that points into the server part, past the message, so a longer path moves
    what it points at."""
    d.mkdir(parents=True, exist_ok=True)
    pub, srv, prv = Region(bases[0], header), Region(bases[1]), Region(bases[2])
    name = pub.string("P.M")
    pub_root = pub.ctor(0, [name, pub.array([name, 1])])
    lengths = lengths or {}
    msg = srv.ctor(1, [srv.string(path, lengths.get(path)), srv.string(text, lengths.get(text)), name])
    entry = srv.ctor(2, [srv.string("declRangeExt"), msg])
    srv_root = srv.array([msg, entry])
    prv_root = prv.ctor(0, [entry, prv.ctor(3, [name], scalar), stray if stray is not None else 1])
    return {"olean": pub.write(d / "M.olean", pub_root), "olean.server": srv.write(d / "M.olean.server", srv_root),
            "olean.private": prv.write(d / "M.olean.private", prv_root)}


def test_a_module_built_at_another_path_is_one_build_and_nothing_else_may_differ(tmp_path):
    """The path is longer, so every object after it in the server part moves and the private part's pointer into it
    differs: one graph. Any other difference, at any depth or across a part, is refused, a string's length alone among
    them, and so is a path that isn't the module's source, isn't absolute, or isn't of its own length."""
    x = module(tmp_path / "x")
    assert G.differs(x, module(tmp_path / "same"), SOURCE) is None
    y = module(tmp_path / "y", path="/a/much/longer/checkout/elsewhere/P/M.lean")
    assert x["olean.private"].read_bytes() != y["olean.private"].read_bytes(), "the pointer into the server part moved"
    assert G.differs(x, y, SOURCE) is None
    refused = {
        "another module's source": module(tmp_path / "other", path="/b/P/N.lean"),
        "a relative path": module(tmp_path / "rel", path="b/P/M.lean"),
        "a path inside a longer name": module(tmp_path / "longer", path="/b/QP/M.lean"),
        "the message's text": module(tmp_path / "text", path="/b/P/M.lean", text="unused variablE"),
        "the length of the message's text": module(tmp_path / "length", lengths={"unused variable": 16}),
        "a path of another length than its text": module(tmp_path / "plength", lengths={"/a/P/M.lean": 12}),
        "a scalar of the private declaration": module(tmp_path / "scalar", scalar=b"\x02" + bytes(7)),
        "a boxed scalar field": module(tmp_path / "boxed", stray=3),
        "the header": module(tmp_path / "header", header=HEADER[:-1] + b"1"),
    }
    for what, z in refused.items():
        assert G.differs(x, z, SOURCE) is not None, what
    assert "string differs" in G.differs(x, refused["another module's source"], SOURCE)
    assert "of length 16 rebuilt" in G.differs(x, refused["the length of the message's text"], SOURCE)
    assert G.differs(x, refused["a scalar of the private declaration"], SOURCE).startswith("olean.private:")
    assert "headers differ" in G.differs(x, refused["the header"], SOURCE)
    assert "parts" in G.differs(x, {p: f for p, f in y.items() if p != "olean.private"}, SOURCE)


def test_an_address_in_no_part_or_in_two_is_malformed(tmp_path):
    x, y = module(tmp_path / "x", stray=0x4000_0000), module(tmp_path / "y", stray=0x4000_0000)
    assert G.differs(x, y, SOURCE).startswith("malformed: address 0x40000000")
    over = module(tmp_path / "over", bases=(0x1000_0000, 0x1000_0000 + 64, 0x3000_0000))
    assert G.differs(module(tmp_path / "ok"), over, SOURCE) == "malformed: olean and olean.server are mapped over each other"
    (tmp_path / "short.olean").write_bytes(b"olean")
    assert G.differs(x, {**x, "olean": tmp_path / "short.olean"}, SOURCE).startswith("malformed:")


BASIC = """module

@[expose] public section

namespace Lab

def fastId (n : Nat) : Nat := n

def fastIe (n : Nat) : Nat := n

@[implemented_by fastId]
def slowId (n : Nat) : Nat := n + 0

private theorem helper (n : Nat) : slowId n = n := by simp [slowId]

theorem slowId_eq (n : Nat) : slowId n = n := helper n

end Lab
"""
LAB = """module

public import Lab.Basic

@[expose] public section

namespace Lab

def unusedArg (m : Nat) (k : Nat) : Nat := k

private theorem two : 1 + 1 = 2 := rfl

end Lab
"""


def built(at: Path, lab: str = LAB, basic: str = BASIC) -> Path:
    """The package's build at ``at``: `Lab` logs a linter warning, which its server part keeps with its source's path."""
    (at / "Lab").mkdir(parents=True)
    (at / "lakefile.toml").write_text('name = "lab"\ndefaultTargets = ["Lab"]\n\n[[lean_lib]]\nname = "Lab"\n')
    shutil.copy(HERE / "fixtures" / "control" / "lean-toolchain", at)
    (at / "Lab.lean").write_text(lab)
    (at / "Lab" / "Basic.lean").write_text(basic)
    subprocess.run(["lake", "build"], cwd=at, check=True, capture_output=True,
                   env={**os.environ, "LAKE_ARTIFACT_CACHE": "false", "LEAN_CACHE": "off"})
    return at / ".lake" / "build" / "lib" / "lean"


def parts(lib: Path, mod: str) -> dict[str, Path]:
    return {p: lib / f"{mod}.{p}" for p in G.PARTS}


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_lean_s_output_at_two_paths_differs_only_in_the_path_and_a_changed_declaration_is_refused(tmp_path):
    """Lean 4.34 writes `Lab`'s warning, with its source's absolute path, into its server part, and the private part points
    past it, so the two builds' bytes differ; `Lab.Basic` logs nothing and is byte-identical. A private theorem changed,
    or an `implemented_by` pointed at another function of the same length, is refused."""
    a, b = built(tmp_path / "a" / "pkg"), built(tmp_path / "elsewhere" / "b" / "pkg")
    assert (a / "Lab.olean.server").read_bytes() != (b / "Lab.olean.server").read_bytes()
    assert (a / "Lab.olean").read_bytes() == (b / "Lab.olean").read_bytes()
    assert G.differs(parts(a, "Lab"), parts(b, "Lab"), "Lab.lean") is None
    assert G.differs(parts(a, "Lab/Basic"), parts(b, "Lab/Basic"), "Lab/Basic.lean") is None
    assert G.differs(parts(a, "Lab"), parts(b, "Lab"), "Other.lean") is not None, "the path must be the module's source"
    proof = built(tmp_path / "c" / "pkg", lab=LAB.replace("1 + 1 = 2", "2 + 0 = 2"))
    assert G.differs(parts(a, "Lab"), parts(proof, "Lab"), "Lab.lean") is not None
    impl = built(tmp_path / "d" / "pkg", basic=BASIC.replace("@[implemented_by fastId]", "@[implemented_by fastIe]"))
    assert G.differs(parts(a, "Lab/Basic"), parts(impl, "Lab/Basic"), "Lab/Basic.lean") is not None
