#!/bin/bash
# First boot of the Nebius server, as root, from cloud-init's runcmd (launch.sh embeds it); safe to run again.
#
#   - sshd: keys only, no root, only $VY_USER;
#   - the data disk (/dev/disk/by-id/virtio-data) is formatted ONLY when it holds no filesystem, and mounted at /workspace, where
#     the tooling already looks: /workspace/hf (HF_HOME, the weight cache), /workspace/research (the research root),
#     /workspace/ramlock (--mem-gb reservations), /workspace/cp (the vLLM integration's build root, as on a RunPod pod: its
#     pipeline and tests create directories there, and /workspace itself is root's);
#   - /run/gpu-lease for gpu-lease, and the lease: vy-lease.service runs research's lease.sh as $VY_USER, which stops this VM
#     through the Nebius API (vy-nebius-stop, the attached service account's token from the metadata service) once the lease
#     and its grace have passed.  Every boot grants VY_BOOT_LEASE_S so a started VM is not stopped before anyone can extend it.
# Its settings come from /etc/vy/vy.env, written by cloud-init.
set -euo pipefail
. /etc/vy/vy.env
U=${VY_USER:-research}
DEV=/dev/disk/by-id/virtio-${VY_DATA_DEVICE_ID:-data}
LEASE_DIR=/home/$U/.research
log() { echo "vy-setup: $*"; }

cat > /etc/ssh/sshd_config.d/10-vy.conf <<EOF
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitRootLogin no
AllowUsers $U
X11Forwarding no
EOF
systemctl reload ssh || systemctl reload sshd || true

for _ in $(seq 60); do [ -e "$DEV" ] && break; sleep 2; done
[ -e "$DEV" ] || { log "no data disk at $DEV"; exit 1; }
if [ -z "$(blkid -o value -s TYPE "$DEV" || true)" ]; then
  log "formatting $DEV (it holds no filesystem)"
  mkfs.ext4 -q -L vy-data -m 0 -i 262144 -E lazy_itable_init=1,lazy_journal_init=1 "$DEV"
fi
mkdir -p /workspace
grep -q ' /workspace ' /etc/fstab || echo "LABEL=vy-data /workspace ext4 defaults,noatime,nofail,x-systemd.device-timeout=120 0 2" >> /etc/fstab
mountpoint -q /workspace || mount /workspace
install -d -o "$U" -g "$U" /workspace/hf /workspace/hf/hub /workspace/hf/prefetch /workspace/research /workspace/ramlock \
  /workspace/cache /workspace/cache/uv /workspace/cp

# TMPDIR: the root disk is 247 GB, and one Lean audit's scratch tree is ~22 GB; four at once filled it on vy-nebius-1 (6 Oct)
install -d -m 1777 -o root -g root /workspace/tmp
if [ -n "${VY_FS_TAG:-}" ]; then   # a shared filesystem launch.sh attached (NEBIUS_FILESYSTEM_ID), shared by every VM it is attached to
  mkdir -p /workspace/fs
  grep -q ' /workspace/fs ' /etc/fstab || echo "$VY_FS_TAG /workspace/fs virtiofs defaults,nofail 0 0" >> /etc/fstab
  mountpoint -q /workspace/fs || mount /workspace/fs
  chown "$U:$U" /workspace/fs
fi
echo 'd /workspace/tmp 1777 root root 10d' > /etc/tmpfiles.d/workspace-tmp.conf
touch /etc/environment
for kv in HF_HOME=/workspace/hf UV_CACHE_DIR=/workspace/cache/uv LEASE_DIR=$LEASE_DIR CUDA_DEVICE_ORDER=PCI_BUS_ID TMPDIR=/workspace/tmp; do
  grep -q "^${kv%%=*}=" /etc/environment || echo "$kv" >> /etc/environment
done

echo 'd /run/gpu-lease 1777 root root -' > /etc/tmpfiles.d/gpu-lease.conf
systemd-tmpfiles --create /etc/tmpfiles.d/gpu-lease.conf

export DEBIAN_FRONTEND=noninteractive
apt-get update -q
# python3-numpy: `python3 tools/verity/lean/check.py --update` runs PoUW's vector generators, which import numpy, under that
# python3 (node 2 and cpu-1 lacked it, 11 Oct)
apt-get install -y -q git rsync fio jq python3-venv python3-numpy util-linux ripgrep
UV_VERSION=0.12.20        # tools/verity/check/pod_setup.sh's pin, the only uv check's preflight accepts (tests/test_uv_pin.py)
[ "$(uv --version 2> /dev/null | cut -d' ' -f2)" = "$UV_VERSION" ] \
  || curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
if [ ! -x /usr/local/bin/nebius ]; then
  curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | env HOME=/opt/nebius bash
  ln -sf /opt/nebius/.nebius/bin/nebius /usr/local/bin/nebius
fi

install -d -o "$U" -g "$U" -m 700 "$LEASE_DIR"
install -o "$U" -g "$U" -m 755 /usr/local/share/vy/lease.sh "$LEASE_DIR/lease.sh"
if [ ! -s "$LEASE_DIR/lease" ] && [ -n "${VY_LEASE_EPOCH:-}" ]; then
  printf '%s %s %s\n' "$VY_LEASE_EPOCH" "$(date -u -d "@$VY_LEASE_EPOCH" +%Y-%m-%dT%H:%M:%SZ)" launch > "$LEASE_DIR/lease"
  chown "$U:$U" "$LEASE_DIR/lease"
fi
cat > /usr/local/sbin/vy-lease-loop <<'EOF'
#!/bin/sh
LEASE_MACHINE_ID=$(curl -sf -H 'Metadata: true' http://metadata.nebius.internal/v1/instance-data/id) || exit 1
export LEASE_MACHINE_ID
exec /bin/sh "$LEASE_DIR/lease.sh" loop
EOF
chmod 755 /usr/local/sbin/vy-lease-loop
cat > /etc/systemd/system/vy-lease.service <<EOF
[Unit]
Description=research lease: stop this VM through the Nebius API once its lease has expired
After=network-online.target
Wants=network-online.target

[Service]
User=$U
Environment=LEASE_DIR=$LEASE_DIR
Environment=LEASE_TERMINATE_CMD=/usr/local/sbin/vy-nebius-stop
ExecStartPre=/bin/sh $LEASE_DIR/lease.sh extend ${VY_BOOT_LEASE_S:-7200} boot
ExecStart=/usr/local/sbin/vy-lease-loop
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

# The hard deadline (/etc/research/deadline, `EPOCH ISO BY`, root's): lease.sh clamps every extension to it, and this timer, which the
# research user can neither edit nor stop, stops the VM from EPOCH - 300 s whatever the lease files say.
install -d -m 755 /etc/research
if [ -n "${VY_DEADLINE_EPOCH:-}" ] && [ ! -s /etc/research/deadline ]; then
  printf '%s %s %s\n' "$VY_DEADLINE_EPOCH" "$(date -u -d "@$VY_DEADLINE_EPOCH" +%Y-%m-%dT%H:%M:%SZ)" launch > /etc/research/deadline
fi
cat > /usr/local/sbin/vy-deadline-check <<'EOF'
#!/bin/sh
f=/etc/research/deadline
[ -e "$f" ] || exit 0
d=""; read -r d _rest < "$f" 2>/dev/null
case $d in ''|*[!0-9]*) d=0;; esac
[ "$(date +%s)" -ge $((d - 300)) ] || exit 0
id=$(curl -sf -H 'Metadata: true' http://metadata.nebius.internal/v1/instance-data/id) || exit 1
logger -t vy-deadline "deadline $d reached: stopping $id"
exec /usr/local/sbin/vy-nebius-stop "$id"
EOF
chmod 755 /usr/local/sbin/vy-deadline-check
cat > /etc/systemd/system/vy-deadline.service <<'EOF'
[Unit]
Description=stop this VM through the Nebius API once /etc/research/deadline has passed

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/vy-deadline-check
EOF
cat > /etc/systemd/system/vy-deadline.timer <<'EOF'
[Unit]
Description=check the hard deadline every minute

[Timer]
OnBootSec=30s
OnUnitActiveSec=60s
AccuracySec=5s

[Install]
WantedBy=timers.target
EOF

# Fixed clocks node-wide at every boot, so timings repeat: 2,100 MHz graphics held under full bf16 load at ~555 W, where unlocked
# GPUs ride the 600 W cap and drift (2,190 -> 2,167 MHz); the RTX PRO 6000 has one memory clock.  Only root can change them.
cat > /etc/systemd/system/vy-clocks.service <<EOF
[Unit]
Description=lock every GPU's clocks (graphics ${VY_GPU_CLOCK_MHZ:-2100} MHz, memory ${VY_MEM_CLOCK_MHZ:-12481} MHz)

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/bin/nvidia-smi -pm 1
ExecStart=/usr/bin/nvidia-smi -lgc ${VY_GPU_CLOCK_MHZ:-2100},${VY_GPU_CLOCK_MHZ:-2100}
ExecStart=/usr/bin/nvidia-smi -lmc ${VY_MEM_CLOCK_MHZ:-12481},${VY_MEM_CLOCK_MHZ:-12481}
ExecStop=/usr/bin/nvidia-smi -rgc
ExecStop=/usr/bin/nvidia-smi -rmc

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now vy-lease.service
systemctl enable --now vy-deadline.timer
[ "${VY_LOCK_CLOCKS:-1}" = 0 ] || systemctl enable --now vy-clocks.service   # the clocks are the RTX PRO 6000's

nvidia-smi -pm 1 > /dev/null 2>&1 || log "nvidia-smi -pm 1 refused (persistence mode stays off)"
date -u +%Y-%m-%dT%H:%M:%SZ > /var/lib/vy-setup.done
log "done"
