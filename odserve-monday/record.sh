#!/usr/bin/env bash
# Read-only: the follow-on FRID's in-run onsite record (its run dir's out/record/onsite.json on FOLLOW_NODE) and the
# checks Monday's result reads off it, for the served run RID (its copy NETACCT/RID there):
#   RECORD-VERDICTS     each verdict
#   RECORD-OF-RECORD    of_record's holds, its composite and verified compute
#   RECORD-PINNED       auditor-listing's registered_files (holds, why, compared: circuit and public, read_by): the
#                       public file is the pin's staged body completed, and θ's layout is the pin's
#   RECORD-CHECKED      every check with holds and why: auditor-listing, auditor-record, scan-state-pinned, ...
#   CERTIFIER-KEY       open or not: where it is open, the network certifier's verdict is NOT_ESTABLISHED
#   RECORD-OPEN         each verdict with open terms or conditions; RECORD-CONDITIONAL of_record's conditions
#   K                   the figure of record (bound.of_record's K, bits a phase) and what is open under it
#   NETWORK-RUN/-PERIOD the network bound over the run and pooled over each monitoring period, for the serving node, a
#                       datacenter and the world (bits and share); SAMPLED-PER-WINDOW/-PERIOD sampled proofs' ε at δ
#                       and its pooled cells
#   MEMBER-OPEN         where the record states the drawn unit's member as open in Lean (onsite_record.MEMBER_OPEN)
#   LAYOUT              v2's member layout from the pre-draw pin: the pin's staged entries state one (a5.pin_layout),
#                       θ (the v2 case) names one, and they are equal; no key or count is printed
# No count leaves the node: the serving field is never read, and a failed check's why (which can name a member's key
# and n) is never posted. Its lines in LOGS/record-FRID.txt.
#   [FOLLOW_NODE=vy-nebius-cpu-11] bash record.sh RID FRID
set -uo pipefail
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$H/env.sh" || exit 3
RID=${1:?the served run id} FRID=${2:?the follow-on run id}
for r in "$RID" "$FRID"; do [[ $r =~ ^r[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$ ]] || { echo "REFUSED: $r is not a store run id"; exit 3; }; done
IP=$(follow_ip "$FOLLOW_NODE") || exit 3
bash "$H/node_probe.sh" "research@$IP" "python3 - /workspace/research/runs/$FRID/out/record/onsite.json $NETACCT/$RID $VCLASS <<'EOF'
import glob, json, sys
from pathlib import Path
rec_path, d, vclass = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
if not rec_path.is_file():
    sys.exit(f'RECORD: none at {rec_path}')
rec = json.loads(rec_path.read_text())
v = rec.get('verdicts') or {}
print('RECORD-VERDICTS:', json.dumps({k: x.get('verdict') for k, x in v.items()}, sort_keys=True))
o = rec.get('of_record') or {}
print('RECORD-OF-RECORD:', json.dumps({k: o.get(k) for k in ('holds', 'composite', 'verified_compute')}, sort_keys=True))
checked = rec.get('checked') or {}
listing = checked.get('auditor-listing') or {}
files = listing.get('registered_files') or {}
print('RECORD-PINNED: registered_files', json.dumps({k: files.get(k) for k in ('holds', 'why', 'compared', 'read_by')}, sort_keys=True),
      '; auditor-listing holds:', listing.get('holds'))
for name, x in sorted(checked.items()):
    print('RECORD-CHECKED:', name, json.dumps({'holds': x.get('holds'), 'why': str(x.get('why') or '')[:300]}, sort_keys=True))
key_open = sorted(k for k, x in v.items() if 'certifier-key' in (x.get('open') or []))
print('CERTIFIER-KEY:', 'open in ' + ', '.join(key_open) if key_open else 'not open')
for name, x in sorted(v.items()):
    if x.get('open') or x.get('conditional_on'):
        print('RECORD-OPEN:', name, json.dumps({k: x.get(k) for k in ('verdict', 'answer', 'open', 'conditional_on')}, sort_keys=True)[:600])
print('RECORD-CONDITIONAL:', json.dumps(o.get('conditional_on'))[:600], '; not established:', json.dumps(o.get('not_established'))[:300])
fig = (rec.get('bound') or {}).get('of_record') or {}
print('K:', fig.get('K'), 'bits a phase; open under it:', json.dumps(fig.get('open'))[:400])
WHERE = ('node', 'datacenter', 'world')
def per_period(label, cells, keys):
    for name, cell in (cells or {}).items():
        if isinstance(cell, dict) and all(isinstance(cell.get(w), dict) for w in WHERE):
            print(label, name, json.dumps({w: {k: cell[w].get(k) for k in keys} for w in WHERE}, default=str)[:600])
periods = rec.get('periods') or {}
run = periods.get('run') if isinstance(periods.get('run'), dict) else {}
print('NETWORK-RUN:', json.dumps({k: run.get(k) for k in ('bits', 'share', 'phases')}, default=str)[:300])
per_period('NETWORK-PERIOD:', periods, ('bits', 'share'))
sp = rec.get('sampled_proofs') or {}
print('SAMPLED-PER-WINDOW:', json.dumps({k: (sp.get('per_window') or {}).get(k) for k in ('epsilon', 'delta', 'law')}, default=str)[:300])
per_period('SAMPLED-PERIOD:', sp.get('pooled'), ('incorrect_at_most', 'share'))
def find(x, path=''):
    if isinstance(x, str) and x.startswith('open, not in Lean: the drawn unit'):
        yield path
    elif isinstance(x, dict):
        for k, y in x.items():
            yield from find(y, f'{path}.{k}')
    elif isinstance(x, list):
        for i, y in enumerate(x):
            yield from find(y, f'{path}[{i}]')
where = list(find(rec))
print('MEMBER-OPEN:', ('stated at ' + ', '.join(where[:3])) if where else 'not stated in the record')
pins = glob.glob(str(d / f'v3c-w*-{vclass}' / 'audit' / 'window' / 'public-pin.json'))
case = d / 'private' / 'sampled-proofs' / 'case.json'
try:
    pin = json.loads(Path(pins[0]).read_text())
    rows = sorted((m['k'], key, m['n']) for key, m in pin['members'].items())
    ok = [k for k, _, _ in rows] == list(range(len(rows))) and all(type(n) is int and n > 0 for _, _, n in rows)
    layout = [[key, n] for _, key, n in rows] if ok else None
    why = '' if ok else 'its staged entries state no k 0 to the member count with a positive n each'
except (IndexError, OSError, KeyError, TypeError, ValueError) as e:
    layout, why = None, f'{type(e).__name__}'
try:
    members = json.loads(case.read_text()).get('proofs', {}).get('members')
except (OSError, ValueError, AttributeError) as e:
    members = None
print('LAYOUT: the pin states one:', 'yes' if layout else f'no ({why})', '; θ names one:', 'yes' if isinstance(members, list) else 'no',
      '; equal:', 'yes' if layout and members == layout else 'no')
EOF" | tee "$LOGS/record-$FRID.txt"
