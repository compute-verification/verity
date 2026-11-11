exec(open('/cursor/stores/self/bin/goals_audit.py').read().split("print('total'")[0])
import collections
openg = [g for g in gs if g['status'] != 'done']
handles = ['benchmarks','ci','circuits','comms','compute-accounting','compute-research','console','infra','kernels','lean','memory-accounting','network-accounting','proofs','soundness','training','vllm']
def kids(g): return [w['goal'] for w in (g.get('waitingOn') or []) if isinstance(w, dict) and 'goal' in w]
out = []
for h in handles:
    og = [g for g in openg if g['owner'] == h]
    lines = []
    t1 = [g for g in og if g['_tier'] == 1]
    off = [g for g in og if g['status'] == 'off-track']
    head = f"**{h}** ({len(og)} open, {len(off)} off track)"
    if not og:
        lines.append("no open goal: the page says nothing about what you are working on. Add the one to three outcomes you are on now.")
    cleanup_only = og and all(('cleanup' in g['tags'] or 'slice-clean' in g['id'] or 'overhaul' in g['id']) for g in og)
    if cleanup_only:
        lines.append("every open goal is cleanup: your research or build work this week is not on the page. Add it.")
    for g in t1:
        ks = kids(g)
        if not ks and not g.get('tier'):
            lines.append(f"`{g['id']}` is tier 1 only because nothing waits on it: link it under its project, or say it is one")
    for g in og:
        if g.get('tier') and g['_tier'] > 1 and not any(by[p]['status'] != 'done' for p in waited.get(g['id'], []) if p in by):
            lines.append(f"`{g['id']}` is set tier {g['_tier']} but no open goal waits on it: link it under its project")
        for k in kids(g):
            t = by.get(k)
            if t is None: lines.append(f"`{g['id']}` waits on `{k}`, which no longer exists")
            elif t['status'] != 'done' and t['_due'] > g['_due']:
                lines.append(f"`{g['id']}` (due {g['_due'].astimezone(PST):%a %d %b}) waits on `{k}` (due {t['_due'].astimezone(PST):%a %d %b}): move one date or drop the link")
    old = [g for g in og if now - g['_upd'] > timedelta(hours=48)]
    if old: lines.append("not written in 48 h: " + ", ".join(f"`{g['id']}`" for g in old))
    lg = [g for g in og if len(g['title'].split()) > 8]
    if lg: lines.append("title over eight words: " + ", ".join(f"`{g['id']}`" for g in lg))
    if len(off) >= 2:
        lines.append(f"{len(off)} off track: for each, the date you now believe or the one thing it waits on")
    out.append((h, head, lines))
for h, head, lines in out:
    print('- ' + head + (': nothing found.' if not lines else ''))
    for l in lines: print('  - ' + l)
