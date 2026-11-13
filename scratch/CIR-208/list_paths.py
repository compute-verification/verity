"""list_paths.py LIST: whether every `file` of the list names a file of this tree, and where the files are."""
import collections, json, sys
from pathlib import Path
doc = json.load(open(sys.argv[1]))
files = collections.Counter(e["file"] for e in doc["entries"])
gone = sorted(f for f in files if not Path(f).is_file())
roots = collections.Counter("/".join(f.split("/")[:4]) for f in files)
print(f"{len(doc['entries'])} entries in {len(files)} files; {len(gone)} name no file")
for f in gone:
    print("  no file:", f)
for r, n in sorted(roots.items()):
    print(f"  {n:3d} files under {r}")
