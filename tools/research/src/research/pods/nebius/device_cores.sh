#!/bin/bash
# sudo vy-device-cores: keep everything but vy-device's scopes off vy-nebius-2's device cores 120-127 (see device.sh), as far as a
# running kernel allows. Idempotent; settings are runtime only, so run it again after a boot. What stays on those cores is the
# per-CPU kernel work only a boot with nohz_full=120-127 isolcpus=managed_irq,domain,120-127 rcu_nocbs=120-127 irqaffinity=0-119
# moves: bound kworkers, timers, vmstat_update (here run every 10 s instead of every second).
# /etc/sysctl.d/90-vy-device-cores.conf keeps this script's sysctls from boot.
set -euo pipefail
systemctl set-property --runtime user.slice AllowedCPUs=0-119
systemctl set-property --runtime system.slice AllowedCPUs=0-119
systemctl set-property --runtime init.scope AllowedCPUs=0-119
for s in provers.slice vy.slice; do systemctl set-property --runtime "$s" AllowedCPUs=0-119,128-191; done
systemctl set-property --runtime devices.slice AllowedCPUs=120-127
systemctl set-property --runtime pous.slice AllowedCPUs=124-127
for s in devices.slice pous.slice/pous-quiet.slice; do
  if [ -w "/sys/fs/cgroup/$s/cgroup.pressure" ]; then echo 0 > "/sys/fs/cgroup/$s/cgroup.pressure"; fi
done
sysctl -q -w vm.stat_interval=10 kernel.numa_balancing=0
# unbound workqueues, IRQs and the kernel threads free to move each keep what they had less 120-127; one with nothing left stays
python3 - <<'EOF'
import glob, os, subprocess
DEV = set(range(120, 128))
def parse(s):
    out = set()
    for part in s.strip().split(","):
        if part:
            a, _, b = part.partition("-")
            out.update(range(int(a), int(b or a) + 1))
    return out
def fmt(cpus):
    return ",".join(str(c) for c in sorted(cpus))
def write(path, value):
    try:
        with open(path, "w") as f:
            f.write(value)
    except OSError:
        pass
wq = "/sys/devices/virtual/workqueue/cpumask"
words = open(wq).read().strip().split(",")
mask = int("".join(words), 16) & ~sum(1 << c for c in DEV)
write(wq, ",".join(f"{(mask >> (32 * i)) & 0xffffffff:08x}" for i in reversed(range(len(words)))))
for d in glob.glob("/proc/irq/[0-9]*"):
    try:
        cur = parse(open(f"{d}/smp_affinity_list").read())
    except OSError:
        continue
    if cur & DEV and cur - DEV:
        write(f"{d}/smp_affinity_list", fmt(cur - DEV))
for st in glob.glob("/proc/[0-9]*/status"):
    try:
        f = dict(l.split(":\t", 1) for l in open(st).read().splitlines() if ":\t" in l)
    except OSError:
        continue
    if f.get("PPid", "").strip() != "2":
        continue
    cur = parse(f["Cpus_allowed_list"])
    if cur & DEV and cur - DEV:
        subprocess.run(["taskset", "-pc", fmt(cur - DEV), f["Pid"].strip()], capture_output=True)
EOF
