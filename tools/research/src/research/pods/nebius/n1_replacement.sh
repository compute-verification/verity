#!/usr/bin/env bash
# vy-nebius-1's replacement (INF-57): n1 stops at its deadline, 19 Oct 15:00Z; its CPU work (two train slots, Lean slots, lean's
# cache publisher, the merge mirror, research runs at nice 19) goes to vy-nebius-cpu-4/5, and its GPU and Kueue services retire.
#
#   n1_replacement.sh [plan] [PATH]     print every live step in order, each command as it is to be run; runs nothing
#   n1_replacement.sh preflight [PATH]  run the read-only checks: each node's region quota, its project's rights (the launcher's
#                                       editor, the VM account's power role) and preset, launch.sh --dry-run, the
#                                       deploy install each node would get (no --apply, read against a live node of its kind),
#                                       n1's custody backlog and the data on n1 nothing else holds
#
# PATH is where the nodes come from, each a list of nodes below (NODES_<PATH>):
#   now       two 192vcpu-768gb nodes (cpu-1's shape: one full train slot each), in regions whose quota has room today
#   quota     two 256vcpu-1024gb nodes (cpu-2's shape) in us-central1, once Nebius quota request C291564672 is granted
#   reattach  one 192vcpu-768gb node in uk-south2 that boots on n1's data disk as it is (vy-nebius-1-data: hf, runs, usage, Lean
#             caches, nothing copied); needs n1's VM and cpu-1's deleted after their stop, for the disk, 192 vCPU and two IPv4s
#
# Every create, start, stop and delete here is printed for the coordinator, whose call spend and new instances are; this script
# makes only Nebius list/get calls and read-only ssh. The nodes' deploy entries are deploy.toml's [nodes.vy-nebius-cpu-4/5].
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../../../../../.." && pwd)

cmd=${1:-plan}
path=${2:-now}
REF=${N1R_REF:-origin/main}
N1=vy-nebius-1
DEADLINE=${N1R_DEADLINE:-2026-10-21T21:00:00Z}   # cpu-2/3's, the fleet's current end
MAX_HOURS=${N1R_MAX_HOURS:-72}
CUSTODY_PY=/workspace/custody/python/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12   # n1's custody tree's interpreter

# NAME REGION PROJECT PRESET RATE DATA_GIB DATA_TYPE DATA_DISK
NODES_now="vy-nebius-cpu-4 uk-south1 project-e03yeqz7lc00cg4kesj06a 192vcpu-768gb 6.34 2046 network_ssd_non_replicated -
vy-nebius-cpu-5 eu-west2 project-e04qzbhvbc0093k7dhq008 192vcpu-768gb 6.34 2046 network_ssd_non_replicated -"
NODES_quota="vy-nebius-cpu-4 us-central1 project-u00tvvtxkc00mdpbnexc70 256vcpu-1024gb 8.45 2048 network_ssd -
vy-nebius-cpu-5 us-central1 project-u00tvvtxkc00mdpbnexc70 256vcpu-1024gb 8.45 2048 network_ssd -"
NODES_reattach="vy-nebius-cpu-4 uk-south2 project-e05mvq7kln00dvsa5xj2mg 192vcpu-768gb 6.34 5022 network_ssd_non_replicated vy-nebius-1-data"
# a live node of each preset, which preflight reads each node's deploy install against
TWIN_192vcpu_768gb=vy-nebius-cpu-1
TWIN_256vcpu_1024gb=vy-nebius-cpu-2

var="NODES_$path"
[ -n "${!var:-}" ] || { echo "n1_replacement: no path $path (now | quota | reattach)" >&2; exit 2; }
NODES=${!var}

launch_env() {  # NAME REGION PROJECT PRESET RATE DATA_GIB DATA_TYPE DATA_DISK
  local disk=""; [ "$8" = - ] || disk=" NEBIUS_DATA_DISK=$8"
  echo "NEBIUS_VM_NAME=$1 NEBIUS_REGION=$2 NEBIUS_PROJECT_ID=$3 NEBIUS_PLATFORM_PREFIX=cpu-d3 NEBIUS_PRESET=$4" \
       "NEBIUS_IMAGE_FAMILY=ubuntu24.04-driverless NEBIUS_DATA_GIB=$6 NEBIUS_DATA_TYPE=$7 NEBIUS_RATE_USD_PER_HOUR=$5$disk"
}

slot_line() {  # PRESET: the train slot's line, as on the live node of that shape
  case $1 in
    192vcpu-768gb) echo "b 54-191 only=trains mem=560   # one full check (peak ~584G) in the 700G user slice, as on cpu-1" ;;
    256vcpu-1024gb) echo "t 4-171 only=trains mem=560   # one full check, as on cpu-2; research runs at nice 19 on the rest" ;;
  esac
}

plan() {
  local names first n
  names=$(printf '%s\n' "$NODES" | cut -d' ' -f1 | tr '\n' ' ')
  first=$(printf '%s\n' "$NODES" | head -1 | cut -d' ' -f1)
  quota_note=""
  [ "$path" != quota ] || quota_note=$'\n    On 1024 GB nodes, give cpu-4/5 their own user-slice and session caps too (cpu-1\'s 700G and 600G are for 755 GB).'
  cat <<EOF
# n1's replacement, path "$path": ${names}from $REF
# Each step: who runs it, about how long, then the commands. Steps marked (coordinator) spend or create.

L0  (coordinator, 5 min) decide the path and the spend: a budgets.toml line for $(printf '%s\n' "$NODES" | awk '{s+=$5} END{printf "$%.2f/h", s}') until $DEADLINE,
    and whether vy-nebius-2 and vy-nebius-cpu-1 are renewed past 19 Oct 15:00Z (their deadlines are n1's).
EOF
  [ "$path" = reattach ] && cat <<EOF

L0r (coordinator, after n1 and cpu-1 stop at 15:00Z, 5 min) free the disk, the vCPUs and two IPv4s; the data disks stay:
    (cd tools/research/src/research/pods/nebius && NEBIUS_VM_NAME=vy-nebius-1 bash teardown.sh delete-vm --yes)
    (cd tools/research/src/research/pods/nebius && NEBIUS_VM_NAME=vy-nebius-cpu-1 NEBIUS_PRESET=192vcpu-768gb bash teardown.sh delete-vm --yes)
EOF
  echo
  echo "L1  (coordinator, ~10 min each, in parallel) create, start and register each node, its lease ${MAX_HOURS} h and deadline $DEADLINE:"
  while read -r n r p pre rate gib typ disk; do
    echo "    (cd tools/research/src/research/pods/nebius && $(launch_env "$n" "$r" "$p" "$pre" "$rate" "$gib" "$typ" "$disk") \\"
    echo "      bash launch.sh --max-hours $MAX_HOURS --yes --register --deadline $DEADLINE)"
  done <<< "$NODES"
  cat <<EOF

L2  (infra, 15 min) a PR: drop \`staged\` from [nodes.$first]$( [ "$names" = "$first " ] || echo " and the others'") in deploy.toml; add each node's line
    to pods/nebius/peers (outside uk-south2 a node has no private address there: its public IP, port 22 open to all); mark it ready.
    Until it lands, install from the PR's pushed head (--ref origin/BRANCH): install reads that commit's own deploy.toml.$quota_note

L3  (infra, 5 min a node) install every entry, then prepare the check pod:
EOF
  for n in $names; do
    echo "    research deploy install $n --by infra --ref $REF --apply"
    echo "    research pods ssh $n -- 'cd /etc/systemd/system && for t in vy-*.timer; do sudo systemctl enable --now \$t; done'   # install enables none"
    echo "    research pods ssh vy-nebius-2 -- 'cat ~/.ssh/vy-cluster' | research pods ssh $n -- 'umask 077; cat > ~/.ssh/vy-cluster'   # peers' key, node to node; .pub too"
    echo "    research pods ssh $n -- 'echo from=\\\"IPS\\\" \$(cat ~/.ssh/vy-cluster.pub) >> ~/.ssh/authorized_keys'   # IPS: each node's egress (curl ifconfig.me)"
    echo "    research run --on $n --project verity --source . --cwd source -- bash tools/verity/check/pod_setup.sh"
  done
  echo
  echo "L4  (infra, 2 min a node) the slot files, one train slot a node, Lean slots on $first (cpu-1's form):"
  while read -r n r p pre rate gib typ disk; do
    echo "    research pods ssh $n -- 'printf \"%s\\n\" \"# NAME CPUS [windows=FILE] [lead=MIN] [only=trains] [mem=GB]: check slots slot.py takes (infra)\" \"$(slot_line "$pre")\" > /workspace/research/locks/slots'"
  done <<< "$NODES"
  cat <<EOF
    research pods ssh $first -- 'printf "%s\n" "# NAME N: Lean audit slots (tools/check/lean_slot.py), n1s moved here (INF-57)" "check 3" "audit 2" "build 3" > /workspace/research/locks/lean-slots'
    and every node's /etc/vy/peers from L2's file: research deploy install NODE --by infra --ref $REF --apply on each live node.

L5  (infra for lean, 10 min, then the first build ~40 min in the background) lean's cache publisher on $first: its R2 key goes
    root to root, never through a terminal (it is 401 until lean's re-carded key lands; copy it after):
    research pods ssh $N1 -- 'sudo cat /etc/verity/lean-cache.env' | research pods ssh $first -- 'sudo install -D -m 600 /dev/stdin /etc/verity/lean-cache.env'
    research pods ssh $first -- 'sudo /usr/local/sbin/vy-lean-build-setup && sudo systemctl start vy-lean-cache-daily.service'

L6  (ci, 10 min) the merge mirror beside lean's sandbox: seed the bare repository on $first as n1's was seeded, then
    RESEARCH_MERGE_MIRROR=$first for \`research merge\` (merge.py MIRROR's default changes in the retire PR, L12).

L7  (ci, 2 min) drive's machines: \`research queue drive --on ...\` gains $(echo "$names" | sed 's/ $//; s/ /,/g') and loses $N1 (before 13:00Z, n1's drain).

L8  (infra, 13:00Z-14:30Z) n1 drains: no new check (L7), Kueue admits nothing after 13:00Z (drain_s), the running trains end;
    \`research queue status\` shows nothing on $N1.

L9  (infra, ~20 min, from 14:00Z) n1's custody to backlog 0, then once more after the last run ends:
    research pods ssh $N1 -- 'sudo /usr/local/bin/vy-custody --dry-run' | grep '^would publish'    # until none
    research pods ssh $N1 -- 'sudo systemctl start vy-custody.service'

L10 (compute-accounting, 30 min) the one recorded dir on n1 R2 doesn't hold and nothing regenerates (26.7 GiB, keep):
    research run --on $N1 --project verity -- research data put --kind pouw-pass --tree /workspace/research/cache/pouw-service-draw/passes/r20261006-224411-4a25 --preserve
    and each owner of an unowned dir it wants kept (preflight lists them): glm47-match (vllm/circuits), pouw/ (compute-accounting), cp/ (vllm).

L11 (nobody, 15:00Z) n1 stops itself at its deadline; its data disk is kept (and billed) until teardown.sh delete-all.

L12 (infra, 30 min) the retire PR: infra/cluster/descriptions/nebius-after-n1.toml over nebius.toml; n1's deploy.toml entries
    and node removed; check.py AUTO, post_train.py MACHINE, merge.py MIRROR, lean/check.py ON_NODE, fleet.py JUMP and keeper's help
    to $first or vy-nebius-2; n1's monitors (monitors.py, n1_alerts.py) to node 2's; \`research pods unregister $N1\`.
    The console (verity_console.py's node1 panels, \`cluster usage\`'s record host, publish_pool.py's POOL_DEST_HOST) moves with it.
EOF
}

preflight() {
  local regions
  regions=$(printf '%s\n' "$NODES" | cut -d' ' -f2 | sort -u | tr '\n' ' ')
  echo "== quota (non-GPU vCPU, IPv4, disks) in $regions"
  bash -c ". '$here/common.sh' && nebius_auth && nb quotas quota-allowance list --parent-id ${N1R_TENANT:-tenant-e00g3zymmem3epgq3a} --page-size 999" \
    | python3 -c '
import json, sys
want = set(sys.argv[1].split())
names = ("compute.instance.non-gpu.vcpu", "vpc.ipv4-address.public.count", "compute.disk.size.network-ssd",
         "compute.disk.size.network-ssd-non-replicated")
for q in json.load(sys.stdin).get("items", []):
    n, s, st = q["metadata"].get("name"), q.get("spec") or {}, q.get("status") or {}
    if n in names and s.get("region") in want:
        lim, use = int(s.get("limit") or 0), int(st.get("usage") or 0)
        scale = 2**30 if "disk" in n else 1
        region, unit = s["region"], " GiB" if scale > 1 else ""
        print(f"  {region:12} {n:46} {use // scale:>6} of {lim // scale:>6}{unit}  free {(lim - use) // scale}")' "$regions"
  echo "== rights in each node's project: the launcher needs editor (its VPC writes come first), the VM's account a power role (its lease)"
  projects=$(printf '%s\n' "$NODES" | cut -d' ' -f3 | sort -u | tr '\n' ' ')
  bash -c ". '$here/common.sh' && nebius_auth && ids='import json, sys; [print(x[\"metadata\"][\"id\"]) for x in json.load(sys.stdin).get(\"items\", [])]'
    permits='import json, sys; [print(sys.argv[1], p[\"spec\"].get(\"resource_id\"), p[\"spec\"].get(\"role\")) for p in json.load(sys.stdin).get(\"items\", [])]'
    for who in launcher:\$NEBIUS_SA_ID vm:\${NEBIUS_VM_SA_ID:-\$NEBIUS_SA_ID}; do
      for g in \$(nb iam group-membership list-member-of --subject-id \${who#*:} --page-size 100 | python3 -c \"\$ids\"); do
        nb iam access-permit list --parent-id \$g --page-size 200 | python3 -c \"\$permits\" \${who%%:*}; done; done" | python3 -c '
import sys
have = {tuple(line.split()) for line in sys.stdin if len(line.split()) == 3}
for p in sys.argv[1].split():
    ed = ("launcher", p, "editor") in have
    pw = any(("vm", p, r) in have for r in ("editor", "compute.instance-power-operator"))
    print(f"  {p}: launcher", "editor" if ed else "NOT EDITOR (its creates are refused)", "; VM account",
          "can stop the VM" if pw else "CANNOT STOP THE VM (no power role: launch.sh stops it at the probe)")' "$projects"
  echo "== presets and launch.sh --dry-run"
  while read -r n r p pre rate gib typ disk; do
    bash -c ". '$here/common.sh' && nebius_auth && nb compute platform list --parent-id $p" | python3 -c '
import json, sys
ok = any(x.get("name") == sys.argv[2] for p in json.load(sys.stdin).get("items", []) if p["metadata"]["name"] == "cpu-d3"
         for x in p["spec"].get("presets", []))
print(f"  {sys.argv[1]}: cpu-d3 {sys.argv[2]}", "listed" if ok else "NOT LISTED", f"in {sys.argv[3]}")' "$n" "$pre" "$r"
    if [ "$disk" != - ]; then
      bash -c ". '$here/common.sh' && nebius_auth && nb compute disk get-by-name --parent-id $p --name $disk" | python3 -c '
import json, sys
d = json.load(sys.stdin); st = d.get("status") or {}
print(f"  {sys.argv[1]}: disk {sys.argv[2]}", st.get("state"), "attached to", st.get("read_write_attachment") or "nothing")' "$n" "$disk"
    fi
    env $(launch_env "$n" "$r" "$p" "$pre" "$rate" "$gib" "$typ" "$disk") NEBIUS_SSH_PUBKEY="ssh-ed25519 AAAA-preflight" \
      bash "$here/launch.sh" --dry-run | python3 -c '
import json, sys
text = sys.stdin.read(); disk = [l for l in text.splitlines() if l.startswith("# data disk")]
spec = json.loads(text.split("(ids are placeholders)\n")[1])
print("  " + spec["metadata"]["name"] + ": launch.sh --dry-run ok,", spec["spec"]["resources"], ";", disk[0][2:])'
  done <<< "$NODES"
  echo "== deploy install, no --apply, each node read against its twin (a live node of its preset)"
  while read -r n r p pre rate gib typ disk; do
    tw="TWIN_${pre//-/_}"
    (cd "$repo" && N1R_NODE=$n N1R_TWIN=${!tw} N1R_REF=$REF uv run -q --package verity-ops python - <<'PY'
import contextlib, io, os
from verity_ops import deploy
node, twin, ref = os.environ["N1R_NODE"], os.environ["N1R_TWIN"], os.environ["N1R_REF"]
base = deploy.ssh_runner()
out = io.StringIO()
with contextlib.redirect_stdout(out):
    rc = deploy.main(["install", node, "--by", "n1-replacement-preflight", "--ref", ref], run=lambda m, argv, stdin: base(twin if m == node else m, argv, stdin))
lines = out.getvalue().splitlines()
print(f"  {node} (read on {twin}): rc {rc}, {sum('would install' in l for l in lines)} would install, {len(lines)} lines")
PY
    )
  done <<< "$NODES"
  echo "== n1: custody backlog, and recorded data R2 doesn't hold that nothing regenerates"
  ssh_n1=$(cd "$repo" && python3 -c 'import shlex, sys; sys.path.insert(0, "tools/research/src"); from research import remote; print(shlex.join(remote.ssh_argv(sys.argv[1], 22, user="research")))' "${N1R_N1_HOST:-81.85.2.165}")
  eval "SSH=($ssh_n1)"
  "${SSH[@]}" 'sudo -n /usr/local/bin/vy-custody --dry-run 2>&1 | grep -E "^(would|no Job)" | sed "s/^/  custody: /" | tail -20; df -h /workspace | tail -1 | sed "s/^/  /"'
  if [ "${N1R_RETENTION:-0}" = 1 ]; then   # it walks all of /workspace: ~15 min at n1's load
    "${SSH[@]}" "cd /workspace && PYTHONPATH=/workspace/custody/tools/research/src $CUSTODY_PY -m research retention report" \
      | awk '$1 == "recorded" && $7 == "-" && $8 == "no" {print "  keep, not in R2, not regenerable: " $2 " " $3 " " $4 " " $9}'
  else
    echo "  (N1R_RETENTION=1 also lists the recorded dirs R2 doesn't hold that nothing regenerates: ~15 min)"
  fi
}

case $cmd in
  plan) plan ;;
  preflight) preflight ;;
  *) echo "n1_replacement: plan | preflight [now | quota | reattach]" >&2; exit 2 ;;
esac
