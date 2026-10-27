"""`importlib.metadata`'s search of `sys.path` is held by the installed distributions, not by each directory's listing."""
from __future__ import annotations

import ast
import subprocess
import sys


def test_importlib_metadata_listings_count_as_the_installed_distributions(tmp_path):
    """Every workspace member is installed, so were `importlib.metadata` held by its listings, every member's root would be in
    every target's reads, and a file added to one member would make every target's entry miss."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "member").mkdir()
    (tmp_path / "member" / "m.py").write_text("")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    code = f"""
import importlib.metadata as md, os, sys
from pathlib import Path
from circuit_check import cache as CC
root = Path({str(tmp_path)!r})
sys.path.insert(0, str(root / "member"))
CC.trace(root)
with CC.recording() as fr:
    list(md.distributions())
    os.listdir(root / "member")
print(sorted(fr.dirs))
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert ast.literal_eval(out.strip().splitlines()[-1]) == ["<distributions>", "member"]
