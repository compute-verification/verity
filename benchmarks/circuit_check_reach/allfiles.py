import importlib.util, os, json
from pathlib import Path
from circuit_check import cache as CC
root = Path(".").resolve()
ents = list(Path("/tmp/ccc").glob("*/*/*.json"))
mods = set.intersection(*[set(CC.read(p)["reads"].get("modules", {})) for p in ents])
files = set.intersection(*[set(CC.read(p)["reads"].get("files", {})) for p in ents])
for m in sorted(mods):
    s = importlib.util.find_spec(m)
    if s and s.origin and s.origin.startswith(str(root)):
        files.add(os.path.relpath(s.origin, root))
chk = [str(p) for p in Path("tools/verity/circuit_check/src/circuit_check").rglob("*.py")]
json.dump({"modules_files": sorted(files), "checker": sorted(chk)}, open("/tmp/allfiles.json", "w"), indent=1)
print(len(files), "module/data files;", len(chk), "checker files")
