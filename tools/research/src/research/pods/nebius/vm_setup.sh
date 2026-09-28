#!/bin/bash
# First boot of the Nebius server, as root, from cloud-init's runcmd (launch.sh embeds it); safe to run again.
#
#   - sshd: keys only, no root, only $VY_USER;
#   - the data disk (/dev/disk/by-id/virtio-data) is formatted ONLY when it holds no filesystem, and mounted at /workspace, where
#     the tooling already looks: /workspace/hf (HF_HOME, the weight cache), /workspace/research (the research root),
#     /workspace/ramlock (--mem-gb reservations);
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
  /workspace/cache /workspace/cache/uv

touch /etc/environment
for kv in HF_HOME=/workspace/hf UV_CACHE_DIR=/workspace/cache/uv LEASE_DIR=$LEASE_DIR CUDA_DEVICE_ORDER=PCI_BUS_ID; do
  grep -q "^${kv%%=*}=" /etc/environment || echo "$kv" >> /etc/environment
done

echo 'd /run/gpu-lease 1777 root root -' > /etc/tmpfiles.d/gpu-lease.conf
systemd-tmpfiles --create /etc/tmpfiles.d/gpu-lease.conf

export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q git rsync fio jq python3-venv util-linux
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
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
systemctl daemon-reload
systemctl enable --now vy-lease.service

nvidia-smi -pm 1 > /dev/null 2>&1 || log "nvidia-smi -pm 1 refused (persistence mode stays off)"
date -u +%Y-%m-%dT%H:%M:%SZ > /var/lib/vy-setup.done
log "done"
