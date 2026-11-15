import sys, os, tomllib, importlib.util
root = sys.argv[1]; os.chdir(root); sys.path.insert(0, root + "/tools/move")
spec = importlib.util.spec_from_file_location("tm", root + "/tools/move/tests/test_maps.py"); tm = importlib.util.module_from_spec(spec); spec.loader.exec_module(tm)
found = tm.two_names({p.name: d for p in sorted(tm.MOVE.glob("*.toml")) if "words" in (d := tomllib.loads(p.read_text()))})
for f in found: print(f)
