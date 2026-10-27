"""The Vercel build step (`buildCommand`): copy the routing code and the registry from the repository into `_vendor/`, so the
deployment runs `main`'s `slack.py` and routes by `main`'s registry, byte for byte.  `_vendor/` is never committed, so the
service has no second copy to keep in step.  It needs the repository around the project's Root Directory, which Vercel's
"Include files outside the root directory in the Build Step" setting provides; without it the build fails here, and Vercel
keeps the deployment before it.

  python3 build.py [OUT]      # OUT defaults to _vendor/ beside this file"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SOURCES = {"research_slack.py": "tools/research/src/research/slack.py",
           "registry.json": ".agents/skills/using-slack/registry.json"}


def main(argv: list[str]) -> int:
    out = Path(argv[0]) if argv else HERE / "_vendor"
    missing = [rel for rel in SOURCES.values() if not (REPO / rel).is_file()]
    if missing:
        print(f"build: {', '.join(missing)} not found under {REPO}: turn on the project's \"Include files outside the root "
              "directory in the Build Step\" setting", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    for name, rel in SOURCES.items():
        shutil.copyfile(REPO / rel, out / name)
        print(f"build: {rel} -> {out / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
