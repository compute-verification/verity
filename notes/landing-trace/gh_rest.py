import json, subprocess, sys, time
out = {}
page = 1
while True:
    for t in range(5):
        r = subprocess.run(["gh","api",f"repos/compute-verification/verity/pulls?state=closed&sort=updated&direction=desc&per_page=100&page={page}"], capture_output=True, text=True)
        if r.returncode == 0: break
        time.sleep(3*(t+1))
    else:
        print("fail page", page, r.stderr[-300:]); break
    d = json.loads(r.stdout)
    if not d: break
    for x in d:
        out[x["number"]] = dict(number=x["number"], title=x["title"], createdAt=x["created_at"], mergedAt=x["merged_at"], closedAt=x["closed_at"],
                                updatedAt=x["updated_at"], headRefOid=x["head"]["sha"], headRefName=x["head"]["ref"], author=(x.get("user") or {}).get("login"),
                                labels=[l["name"] for l in x.get("labels",[])])
    last = d[-1]["updated_at"]
    print(page, len(out), last, flush=True)
    if last < "2026-10-14": break
    page += 1
json.dump(list(out.values()), open("/tmp/trace/gh_closed_rest.json","w"))
