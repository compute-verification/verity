"""List an issue's comments with id, time, Linear author and first words.

    python3 linear_authors.py CI-489 [--only-user NAME]

Run from /tmp/rchk with PYTHONPATH=tools/research/src RESEARCH_AT_CHECKOUT=1.
"""
import sys

from research import secret_hint
from research.linear import Linear

secret_hint.recover()
issue = sys.argv[1]
only = sys.argv[sys.argv.index("--only-user") + 1] if "--only-user" in sys.argv else ""
nodes, cursor = [], None
while True:
    after = f', after: "{cursor}"' if cursor else ""
    got = Linear()._issue(
        issue,
        f"comments(first: 50{after}) {{ nodes {{ id createdAt body user {{ name }} botActor {{ name }} }} "
        "pageInfo { hasNextPage endCursor } }",
    )["comments"]
    nodes += got["nodes"]
    if not got["pageInfo"]["hasNextPage"]:
        break
    cursor = got["pageInfo"]["endCursor"]
nodes.sort(key=lambda n: n["createdAt"])
for n in nodes:
    who = (n.get("user") or {}).get("name") or ("bot:" + ((n.get("botActor") or {}).get("name") or "?"))
    if only and only.lower() not in who.lower():
        continue
    print(n["id"][:8], n["createdAt"][:16] + "Z", f"{who:22s}", " ".join(n["body"].split())[:150])
