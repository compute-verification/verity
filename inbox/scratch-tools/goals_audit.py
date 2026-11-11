import json, sys, re
from datetime import datetime, timezone, timedelta
from collections import Counter, defaultdict
gs = json.load(open('/tmp/goals/all.json'))
now = datetime.now(timezone.utc)  # input: `research goals show --json > /tmp/goals/all.json` first
PST = timezone(timedelta(hours=-8))
def due(g):
    d = g['due']
    if 'T' in d:
        return datetime.fromisoformat(d)
    return datetime.fromisoformat(d + 'T23:59:59-08:00')
def upd(g):
    return datetime.fromisoformat(g['updatedAt'].replace('Z', '+00:00'))
by = {g['id']: g for g in gs}
waited = defaultdict(list)   # id -> goals that wait on it
for g in gs:
    for w in g.get('waitingOn', []) or []:
        if isinstance(w, dict) and 'goal' in w:
            waited[w['goal']].append(g['id'])
def tier(g, seen=()):
    if g.get('tier'): return g['tier']
    ps = [by[p] for p in waited.get(g['id'], []) if p in by and p not in seen]
    if not ps: return 1
    return min(3, 1 + min(tier(p, seen + (g['id'],)) for p in ps))
for g in gs: g['_tier'] = tier(g); g['_due'] = due(g); g['_upd'] = upd(g)
openg = [g for g in gs if g['status'] != 'done']
print('total', len(gs), 'status', dict(Counter(g['status'] for g in gs)))
print('open by tier', dict(sorted(Counter(g['_tier'] for g in openg).items())), ' explicit tier on', sum(1 for g in openg if g.get('tier')), 'of', len(openg))
print('open by priority', dict(sorted(Counter(g['priority'] for g in openg).items())), ' prioritySet:', [(g['id'], g['priority'], g.get('prioritySetBy')) for g in gs if g['prioritySet']])
print('headline open', sum(1 for g in openg if g['headline']), 'of', len(openg))
owners = sorted(set(g['owner'] for g in openg))
print()
print(f"{'owner':20} open t1 t2 t3  off overdue stale3d  oldest-update")
for o in owners:
    og = [g for g in openg if g['owner'] == o]
    t = Counter(g['_tier'] for g in og)
    off = sum(1 for g in og if g['status'] == 'off-track')
    over = sum(1 for g in og if g['_due'] < now)
    stale = sum(1 for g in og if now - g['_upd'] > timedelta(days=3))
    oldest = min(g['_upd'] for g in og)
    print(f"{o:20} {len(og):4} {t[1]:2} {t[2]:2} {t[3]:2}  {off:3} {over:7} {stale:7}  {(now-oldest).days} d")
print()
over = sorted([g for g in openg if g['_due'] < now], key=lambda g: g['_due'])
print('OVERDUE open goals:', len(over))
for g in over:
    print(f"  {g['_due'].astimezone(PST):%a %d %b %H:%M} {g['status']:9} t{g['_tier']} p{g['priority']} {g['id']:55} upd {(now-g['_upd']).total_seconds()/3600:5.0f} h ago")
print()
offnb = [g for g in openg if g['status'] == 'off-track' and not g.get('blockers') and not g.get('waitingOn')]
print('off-track with no blockers and no waits:', [g['id'] for g in offnb])
bad = []
for g in openg:
    for w in g.get('waitingOn', []) or []:
        if isinstance(w, dict) and 'goal' in w:
            t = by.get(w['goal'])
            if t is None: bad.append((g['id'], w['goal'], 'missing'))
            elif t['status'] == 'done': bad.append((g['id'], w['goal'], 'done'))
            elif t['_due'] > g['_due']: bad.append((g['id'], w['goal'], 'due after'))
print('bad waits:', len(bad))
for b in bad: print('  ', b)
long = [g['id'] for g in openg if len(g['title'].split()) > 8]
print('titles over 8 words (open):', len(long))
print('tags (open):', dict(Counter(t for g in openg for t in g['tags']).most_common()))
