"""Scratch, never to land: what the data sheets' two loaders find in the environment that runs this file. It prints, and
always exits 0: `verity.hardware.cards.CARDS` and `verity.hardware.nodes.NODES`, whether each is a directory, and how many
sheets each loader lists there."""
import importlib
import os

import verity

print("verity:", verity.__file__)
print("PYTHONPATH is", "set" if os.environ.get("PYTHONPATH") else "unset")
for name, attr, lister in (("verity.hardware.cards", "CARDS", "skus"), ("verity.hardware.nodes", "NODES", "kinds")):
    try:
        m = importlib.import_module(name)
        d = getattr(m, attr)
        print(f"{name}: {m.__file__}")
        print(f"{name}.{attr}: {d} | is a directory: {d.is_dir()} | resolved: {d.resolve()}")
        listed = getattr(m, lister)()
        print(f"len({name}.{lister}()): {len(listed)} {listed}")
    except Exception as e:  # a report, so a loader that raises is printed as what it raised
        print(f"{name}: {type(e).__name__}: {e}")
