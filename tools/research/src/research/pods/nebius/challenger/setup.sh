#!/bin/bash
# vy-challenger-setup GATE_KEYS [--lock]: the auditor host's accounts and state (vy-challenger-1; RT4-INT-17, -18), as root, safe to
# rerun. benchmarks/live_challenger/auditor.py says what runs there and why.
#
#   - `auditor`: a system user with no login, home /var/lib/vy-challenger (0700, so research can't enter it): the challenger key
#     (key/, made by `auditor.py keygen`), the ledger (windows.jsonl, append-only by chattr +a, which only root can lift), the
#     sessions (sessions/<id>/) and the tunnel key (.ssh/vy-challenger-tunnel, made here once; its public half is printed for each
#     peer's authorized_keys, and /etc/vy/peers names it). Its known_hosts starts as research's.
#   - research's authorized_keys becomes GATE_KEYS's public keys, each forced to vy-challenger-gate (restrict,command=), and
#     /etc/sudoers.d/60-vy-challenger lets research run that gate as root and nothing else (installed once visudo accepts it).
#     GATE_KEYS holds `LANE TYPE BASE64 [COMMENT]` a line: a lane's own key, whose opens the gate records as that lane's (`by`,
#     RT4-INT-24), its forced command `vy-challenger-gate LANE`; LANE `-` registers a key that reads only (status, ledger,
#     pubkey, fetch: the shared lane key the control pod's publisher holds). A lane makes its key on its own machine and sends
#     infra only the public half. A copy is kept at /etc/vy/challenger-gate-keys, which a rerun can take.
#   - --lock also moves cloud-init's `research ALL=(ALL) NOPASSWD:ALL` aside, to /var/lib/vy-challenger-setup/, so research holds
#     no root but the gate; vy-admin (vy-admin-setup) holds root. Rollback: move that file back to /etc/sudoers.d/.
set -euo pipefail
die() { echo "vy-challenger-setup: $*" >&2; exit "${2:-1}"; }
[ $# -ge 1 ] && [ $# -le 2 ] || die "usage: vy-challenger-setup GATE_KEYS [--lock]" 2
keys=$1 lock=0
[ $# = 1 ] || { [ "$2" = --lock ] || die "usage: vy-challenger-setup GATE_KEYS [--lock]" 2; lock=1; }
[ "$(id -u)" = 0 ] || die "run as root" 3
[ -f "$keys" ] && ! grep -q 'PRIVATE KEY' "$keys" || die "$keys: a file of public keys, one a line, no options"
id vy-admin > /dev/null 2>&1 || die "no vy-admin: run vy-admin-setup first, or this would leave no way to root"

A=auditor H=/var/lib/vy-challenger
if ! id "$A" > /dev/null 2>&1; then
  useradd --system --user-group --home-dir "$H" --no-create-home --shell /usr/sbin/nologin --comment "challenger auditor" "$A"
fi
install -d -m 0700 -o "$A" -g "$A" "$H" "$H/sessions" "$H/.ssh"
[ -e "$H/windows.jsonl" ] || install -m 0644 -o "$A" -g "$A" /dev/null "$H/windows.jsonl"
chattr +a "$H/windows.jsonl"
if [ ! -e "$H/.ssh/vy-challenger-tunnel" ]; then
  runuser -u "$A" -- ssh-keygen -q -t ed25519 -N '' -C "auditor@vy-challenger-1 tunnel" -f "$H/.ssh/vy-challenger-tunnel"
fi
[ -e "$H/.ssh/known_hosts" ] || install -m 0644 -o "$A" -g "$A" /home/research/.ssh/known_hosts "$H/.ssh/known_hosts"
echo "tunnel key: $(cat "$H/.ssh/vy-challenger-tunnel.pub")"

ak=/home/research/.ssh/authorized_keys
tmp=$(mktemp "$ak.XXXXXX")
n=0 seen= lanes=
while IFS= read -r ln || [ -n "$ln" ]; do
  case "$ln" in ''|'#'*) continue ;; esac
  [[ $ln =~ ^(-|[A-Za-z0-9][A-Za-z0-9._:-]{0,63})\ ((ssh-ed25519|ssh-rsa|ecdsa-sha2-nistp(256|384|521))\ ([A-Za-z0-9+/]+=*)(\ [^[:cntrl:]]*)?)$ ]] \
    || { rm -f "$tmp"; die "$keys: a line that is not LANE TYPE BASE64 [COMMENT] (LANE - reads only; no options)"; }
  lane=${BASH_REMATCH[1]} key=${BASH_REMATCH[2]} blob=${BASH_REMATCH[5]}
  case "$lane" in root|vy-admin|research|auditor) rm -f "$tmp"; die "$keys: $lane is a host account, not a lane" ;; esac
  case " $seen " in *" $blob "*) rm -f "$tmp"; die "$keys: one key on two lines (sshd would take the first's lane)" ;; esac
  seen="$seen $blob"
  if [ "$lane" = - ]; then
    printf 'restrict,command="/usr/local/sbin/vy-challenger-gate" %s\n' "$key" >> "$tmp"
  else
    printf 'restrict,command="/usr/local/sbin/vy-challenger-gate %s" %s\n' "$lane" "$key" >> "$tmp"
    lanes="$lanes $lane"
  fi
  n=$((n + 1))
done < "$keys"
[ "$n" -gt 0 ] || { rm -f "$tmp"; die "$keys holds no key"; }
[ "$keys" -ef /etc/vy/challenger-gate-keys ] || install -D -m 0644 "$keys" /etc/vy/challenger-gate-keys
chown research:research "$tmp"
chmod 0600 "$tmp"
mv -f "$tmp" "$ak"
echo "research's authorized_keys: $n keys, each forced to vy-challenger-gate; lanes that open:${lanes:- none}"

s=/etc/sudoers.d/60-vy-challenger
printf '%s\n' "# vy-challenger-setup: research runs the challenger's gate as root, and nothing else (benchmarks/live_challenger/auditor.py)" \
  'research ALL=(root) NOPASSWD: /usr/local/sbin/vy-challenger-gate --ssh *' > "$s.tmp"
visudo -cqf "$s.tmp" || { rm -f "$s.tmp"; die "visudo refuses $s"; }
chmod 0440 "$s.tmp"
mv -f "$s.tmp" "$s"
if [ "$lock" = 1 ]; then
  install -d -m 0700 /var/lib/vy-challenger-setup
  for f in /etc/sudoers.d/90-cloud-init-users; do
    [ ! -e "$f" ] || mv -f "$f" /var/lib/vy-challenger-setup/
  done
fi
visudo -cq || die "visudo -c refuses the configuration"
echo "sudoers: $s$([ "$lock" = 1 ] && echo "; cloud-init's grant moved to /var/lib/vy-challenger-setup/")"
