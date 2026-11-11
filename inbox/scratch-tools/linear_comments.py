"""Dump an issue's comments to files, oldest first, and print an index.

    python3 linear_comments.py TOP-394 [OUTDIR] [--since ISO]

Run from /tmp/rchk with PYTHONPATH=tools/research/src RESEARCH_AT_CHECKOUT=1.
Each comment goes to OUTDIR/NN-HHMMZ-<who>.md; the index line is NN, time, who, size.
"""
import re
import sys
from pathlib import Path

from research import secret_hint
from research.linear import Linear

secret_hint.recover()


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    since = sys.argv[sys.argv.index("--since") + 1] if "--since" in sys.argv else ""
    if since:
        args = [a for a in args if a != since]
    issue = args[0]
    out = Path(args[1] if len(args) > 1 else f"/tmp/linear/{issue}")
    out.mkdir(parents=True, exist_ok=True)
    nodes, cursor = [], None
    while True:
        after = f', after: "{cursor}"' if cursor else ""
        got = Linear()._issue(
            issue,
            f"comments(first: 50{after}) {{ nodes {{ id createdAt body parent {{ id }} }} "
            "pageInfo { hasNextPage endCursor } }",
        )["comments"]
        nodes += got["nodes"]
        if not got["pageInfo"]["hasNextPage"]:
            break
        cursor = got["pageInfo"]["endCursor"]
    nodes.sort(key=lambda n: n["createdAt"])
    for i, n in enumerate(nodes, 1):
        body = n["body"]
        head = body.lstrip("_@ \n")[:160]
        who = re.match(r"(?:Posted by top for )?\**([a-z][a-z-]+)", head)
        name = who.group(1) if who else "unknown"
        stamp = n["createdAt"][11:16].replace(":", "") + "Z"
        path = out / f"{i:02d}-{stamp}-{name}.md"
        path.write_text(body)
        if n["createdAt"] >= since:
            reply = " (reply)" if n.get("parent") else ""
            print(f"{i:02d} {n['createdAt'][:16]}Z {name:20s} {len(body):6d} chars{reply}  {path.name}")


main()
