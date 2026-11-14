"""sites.py REV FILE...: the retired-term sites the lint counts in each FILE as REV holds it, one a line with its line
number, what the lint says of it and the line's text, by the checkout's own lint module (its GLOSSARY.md is the term
source) and no other reader. SITES_ROOT names another worktree whose lint module and glossary read it."""
import importlib.util
import os
import subprocess
import sys

R = "/agent/repos/verity"
spec = importlib.util.spec_from_file_location("rt", f"{os.environ.get('SITES_ROOT', R)}/tests/lint/test_repo_retired_terms.py")
rt = importlib.util.module_from_spec(spec)
sys.modules["rt"] = rt
spec.loader.exec_module(rt)
rev = sys.argv[1]
for path in sys.argv[2:]:
    text = subprocess.run(["git", "-C", R, "show", f"{rev}:{path}"], capture_output=True, text=True, check=True).stdout
    found = rt.sites(text, path)
    print(f"{path} at {rev[:12]}: {len(found)} site(s)")
    lines = text.splitlines()
    for n, what in found:
        print(f"  {n}: {what[:60]} | {lines[n - 1].strip()[:140]}")
