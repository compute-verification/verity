#!/bin/bash
# vy-verifier-setup GATE_KEYS TUNNEL_KEYS [--lock]: the verifier's auditor host's accounts and state (vy-auditor-1; INF-179), as
# root, safe to rerun. benchmarks/live_verifier/host.py says what runs there and why; /etc/vy/verifier.json is the host's
# configuration (its challenger and each audited node's capacities, deploy.toml's).
#
#   - `auditor`: a system user with no login, home /var/lib/vy-verifier (0700, so research can't enter it): the runs (runs/<id>/),
#     their scratch, the verifier's registry and the key that reaches the challenger's gate (.ssh/vy-challenger-gate, made here
#     once; its public half is printed for infra to register on vy-challenger-1 for the lane verifier.json names). Its known_hosts
#     starts as research's; vy-challenger-1's host key goes in it before the first run.
#   - research's authorized_keys becomes GATE_KEYS's public keys, each forced to vy-verifier-gate (restrict,command=), and
#     /etc/sudoers.d/60-vy-verifier lets research run that gate as root and nothing else (installed once visudo accepts it).
#     GATE_KEYS holds `LANE TYPE BASE64 [COMMENT]` a line: a lane's own key, whose runs the gate records as that lane's (`by`),
#     its forced command `vy-verifier-gate LANE`; LANE `-` registers a key that reads only (status, fetch). A lane makes its key
#     on its own machine and sends infra only the public half. A copy is kept at /etc/vy/verifier-gate-keys, which a rerun can
#     take.
#   - `prover` and `challenger`: system users that only forward (12-vy-verifier.conf: one command, vy-verifier-tunnel, and
#     their keys in root's /etc/vy/verifier-tunnel-keys.d/ACCOUNT, never their own). TUNNEL_KEYS holds `ACCOUNT ADDRESS TYPE
#     BASE64 [COMMENT]` a line: the audited node's tunnel key as `prover` (its `ssh -L` to a run's port), vy-challenger-1's
#     tunnel key (`auditor@vy-challenger-1 tunnel`) as `challenger` (its sessions' `ssh -R`), each accepted only from ADDRESS
#     (`from=`). A copy is kept at /etc/vy/verifier-tunnel-keys.
#   - /etc/vy/verifier-store.env (root's, 0600) is made empty if absent: infra writes the store's write credential there,
#     which only vy-verifier-publish@ reads.
#   - --lock also moves cloud-init's `research ALL=(ALL) NOPASSWD:ALL` aside, to /var/lib/vy-verifier-setup/, so research holds
#     no root but the gate; vy-admin (vy-admin-setup) holds root. Rollback: move that file back to /etc/sudoers.d/.
set -euo pipefail
die() { echo "vy-verifier-setup: $*" >&2; exit "${2:-1}"; }
usage="usage: vy-verifier-setup GATE_KEYS TUNNEL_KEYS [--lock]"
[ $# -ge 2 ] && [ $# -le 3 ] || die "$usage" 2
keys=$1 tunnels=$2 lock=0
[ $# = 2 ] || { [ "$3" = --lock ] || die "$usage" 2; lock=1; }
[ "$(id -u)" = 0 ] || die "run as root" 3
for f in "$keys" "$tunnels"; do
  [ -f "$f" ] && ! grep -q 'PRIVATE KEY' "$f" || die "$f: a file of public keys, one a line, no options"
done
id vy-admin > /dev/null 2>&1 || die "no vy-admin: run vy-admin-setup first, or this would leave no way to root"
lane=$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["challenger"]["lane"])' /etc/vy/verifier.json) \
  || die "/etc/vy/verifier.json: deploy it first (deploy.toml)"

A=auditor H=/var/lib/vy-verifier
if ! id "$A" > /dev/null 2>&1; then
  useradd --system --user-group --home-dir "$H" --no-create-home --shell /usr/sbin/nologin --comment "verifier auditor" "$A"
fi
install -d -m 0700 -o "$A" -g "$A" "$H" "$H/.ssh"
install -d -m 0755 -o "$A" -g "$A" "$H/runs" "$H/scratch"
if [ ! -e "$H/.ssh/vy-challenger-gate" ]; then
  runuser -u "$A" -- ssh-keygen -q -t ed25519 -N '' -C "auditor@vy-auditor-1 challenger-gate" -f "$H/.ssh/vy-challenger-gate"
fi
[ -e "$H/.ssh/known_hosts" ] || install -m 0644 -o "$A" -g "$A" /home/research/.ssh/known_hosts "$H/.ssh/known_hosts"
echo "challenger gate key (vy-challenger-1's GATE_KEYS, lane $lane): $lane $(cat "$H/.ssh/vy-challenger-gate.pub")"
[ -e /etc/vy/verifier-store.env ] || install -m 0600 -o root -g root /dev/null /etc/vy/verifier-store.env

KEY='(ssh-ed25519|ssh-rsa|ecdsa-sha2-nistp(256|384|521))\ ([A-Za-z0-9+/]+=*)(\ [^[:cntrl:]]*)?'
ak=/home/research/.ssh/authorized_keys
tmp=$(mktemp "$ak.XXXXXX")
n=0 seen= lanes=
while IFS= read -r ln || [ -n "$ln" ]; do
  case "$ln" in ''|'#'*) continue ;; esac
  [[ $ln =~ ^(-|[A-Za-z0-9][A-Za-z0-9._:-]{0,63})\ ($KEY)$ ]] \
    || { rm -f "$tmp"; die "$keys: a line that is not LANE TYPE BASE64 [COMMENT] (LANE - reads only; no options)"; }
  l=${BASH_REMATCH[1]} key=${BASH_REMATCH[2]} blob=${BASH_REMATCH[5]}
  case "$l" in root|vy-admin|research|auditor|prover|challenger) rm -f "$tmp"; die "$keys: $l is a host account, not a lane" ;; esac
  case " $seen " in *" $blob "*) rm -f "$tmp"; die "$keys: one key on two lines (sshd would take the first's lane)" ;; esac
  seen="$seen $blob"
  if [ "$l" = - ]; then
    printf 'restrict,command="/usr/local/sbin/vy-verifier-gate" %s\n' "$key" >> "$tmp"
  else
    printf 'restrict,command="/usr/local/sbin/vy-verifier-gate %s" %s\n' "$l" "$key" >> "$tmp"
    lanes="$lanes $l"
  fi
  n=$((n + 1))
done < "$keys"
[ "$n" -gt 0 ] || { rm -f "$tmp"; die "$keys holds no key"; }

T=/etc/vy/verifier-tunnel-keys.d
install -d -m 0755 -o root -g root "$T"
declare -A tk=([prover]= [challenger]=)
while IFS= read -r ln || [ -n "$ln" ]; do
  case "$ln" in ''|'#'*) continue ;; esac
  [[ $ln =~ ^(prover|challenger)\ ([0-9]{1,3}(\.[0-9]{1,3}){3})\ ($KEY)$ ]] \
    || { rm -f "$tmp"; die "$tunnels: a line that is not ACCOUNT ADDRESS TYPE BASE64 [COMMENT] (ACCOUNT prover or challenger, ADDRESS an IPv4 address)"; }
  acct=${BASH_REMATCH[1]} from=${BASH_REMATCH[2]} key=${BASH_REMATCH[4]} blob=${BASH_REMATCH[7]}
  case " $seen " in *" $blob "*) rm -f "$tmp"; die "$tunnels: a key that is on another line, or a gate key" ;; esac
  seen="$seen $blob"
  # A tunnel needs no terminal, and the prover's key gets none (run.sh's `ssh -T`). The challenger's keeps `pty` only because
  # serve.py's `ssh -R`, pinned on vy-challenger-1, forces one (`-tt`), and ssh ends a session whose forced pty is refused;
  # it goes once vy-challenger-1 runs a serve.py without -tt.
  opts=restrict,port-forwarding
  [ "$acct" = prover ] || opts+=,pty
  tk[$acct]+=$(printf '%s,from="%s" %s' "$opts" "$from" "$key")$'\n'
done < "$tunnels"
for acct in prover challenger; do
  if ! id "$acct" > /dev/null 2>&1; then
    useradd --system --user-group --home-dir /var/lib/vy-verifier-tunnel --no-create-home --shell /bin/sh --password '*' \
      --comment "verifier host's $acct tunnel" "$acct"
  fi
  printf '%s' "${tk[$acct]}" > "$T/$acct.tmp"
  chmod 0644 "$T/$acct.tmp"
  mv -f "$T/$acct.tmp" "$T/$acct"
done
install -d -m 0755 -o root -g root /var/lib/vy-verifier-tunnel
[ "$tunnels" -ef /etc/vy/verifier-tunnel-keys ] || install -D -m 0644 "$tunnels" /etc/vy/verifier-tunnel-keys
echo "tunnel keys: prover $(grep -c . "$T/prover" || true), challenger $(grep -c . "$T/challenger" || true)"

[ "$keys" -ef /etc/vy/verifier-gate-keys ] || install -D -m 0644 "$keys" /etc/vy/verifier-gate-keys
chown research:research "$tmp"
chmod 0600 "$tmp"
mv -f "$tmp" "$ak"
echo "research's authorized_keys: $n keys, each forced to vy-verifier-gate; lanes that run:${lanes:- none}"

s=/etc/sudoers.d/60-vy-verifier
printf '%s\n' "# vy-verifier-setup: research runs the verifier's gate as root, and nothing else (benchmarks/live_verifier/host.py)" \
  'research ALL=(root) NOPASSWD: /usr/local/sbin/vy-verifier-gate --ssh *' > "$s.tmp"
visudo -cqf "$s.tmp" || { rm -f "$s.tmp"; die "visudo refuses $s"; }
chmod 0440 "$s.tmp"
mv -f "$s.tmp" "$s"
if [ "$lock" = 1 ]; then
  install -d -m 0700 /var/lib/vy-verifier-setup
  for f in /etc/sudoers.d/90-cloud-init-users; do
    [ ! -e "$f" ] || mv -f "$f" /var/lib/vy-verifier-setup/
  done
fi
visudo -cq || die "visudo -c refuses the configuration"
echo "sudoers: $s$([ "$lock" = 1 ] && echo "; cloud-init's grant moved to /var/lib/vy-verifier-setup/")"
sshd -t || die "sshd -t refuses the configuration (12-vy-verifier.conf): fix it before reloading ssh"
echo "sshd -t: ok; reload with \`systemctl reload ssh\`"
