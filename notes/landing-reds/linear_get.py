import json, os, sys, urllib.request
q = """query($id: String!) { issue(id: $id) { identifier title state { name } createdAt updatedAt description
  comments(first: 100) { nodes { createdAt body user { name } } } } }"""
for ident in sys.argv[1:]:
    req = urllib.request.Request("https://api.linear.app/graphql", data=json.dumps({"query": q, "variables": {"id": ident}}).encode(),
                                 headers={"Authorization": os.environ["LINEAR_API_KEY"], "Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=60))
    i = (d.get("data") or {}).get("issue")
    if not i:
        print(ident, d.get("errors")); continue
    with open(f"/tmp/reds/linear-{ident}.md", "w") as f:
        f.write(f"# {i['identifier']} {i['title']} [{i['state']['name']}] {i['createdAt']} .. {i['updatedAt']}\n\n{i['description'] or ''}\n")
        for c in sorted(i["comments"]["nodes"], key=lambda c: c["createdAt"]):
            f.write(f"\n---\n{c['createdAt']} {(c.get('user') or {}).get('name')}\n{c['body']}\n")
    print(ident, i["title"][:100], i["state"]["name"], len(i["comments"]["nodes"]), "comments")
