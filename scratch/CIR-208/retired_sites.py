"""retired_sites.py BASE: the retired-terms lint's sites() on each file HEAD changes against BASE, at BASE and at HEAD."""
import subprocess, sys
sys.path.insert(0, "tests/lint")
import test_repo_retired_terms as T
base = sys.argv[1]
files = subprocess.run(["git", "diff", "--name-only", base, "HEAD"], capture_output=True, text=True, check=True).stdout.split()
gained = 0
for f in files:
    new = open(f, encoding="utf-8").read()
    old_p = subprocess.run(["git", "show", f"{base}:{f}"], capture_output=True, text=True)
    old = old_p.stdout if old_p.returncode == 0 else ""
    a, b = (T.sites(old, f) if old else []), T.sites(new, f)
    gained += max(0, len(b) - len(a))
    print(f"{f}: holds={T.holds(f)} sites at base {len(a)}, at head {len(b)}" + (f" | {sorted(set(t for _, t in b))}" if b else ""))
print("files gaining a site:", gained)
