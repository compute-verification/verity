import json

rows = []
with open("/data/vqueue/live/decisions.jsonl") as f:
    for line in f:
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("why") in ("judged pass", "judged fail"):
            rows.append(d)
for d in rows[-40:]:
    rec = d.get("record") or {}
    print(d.get("at"), d.get("why"), (d.get("sha") or "")[:12], (d.get("base") or "")[:12], rec.get("gate"),
          ",".join(rec.get("ids") or []))
