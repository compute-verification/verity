#!/bin/sh
# vy-control-verity's services, from the network volume (infra, 1 Oct). Every piece of state lives under /data; the container disk
# holds only these links, which a RunPod reset (any pod edit, even a rename) wipes along with /post_start.sh itself: after a reset,
# run `sh /data/boot/post_start.sh` once. The guard is started here too, so a restart never leaves the budgets unguarded; this is
# the only pod that may run it (budgets.GUARD_HOST).
mkdir -p /root/.config /root/.local/bin /root/.local/share/uv
for pair in /root/.research:/data/state/root-research /root/.runpod:/data/state/runpod /root/.config/verity:/data/state/config-verity \
    /root/.local/bin/uv:/data/uv/bin/uv /root/.local/share/uv/python:/data/uv/python \
    /usr/local/bin/python3.12:/data/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12 \
    /workspace/steward:/data/steward /workspace/guard-src:/data/guard-src /workspace/infra:/data/infra /workspace/console:/data/console /workspace/chats:/data/chats \
    /workspace/research:/data/research /workspace/research-control:/data/research-control /post_start.sh:/data/boot/post_start.sh; do
  l=${pair%%:*}; t=${pair#*:}
  [ -d "$l" ] && [ ! -L "$l" ] && rmdir "$l" 2>/dev/null
  ln -sfn "$t" "$l"
done
# Drive's merges need `git --attr-source` (git >= 2.40); the image's Ubuntu git is 2.34, so git comes from the git-core PPA.
# add-apt-repository can't run on this image (its python3 lacks apt_pkg), so the key and the list are written directly.
PPA_KEY=/usr/share/keyrings/git-core-ppa.gpg
git_ok() { git version | awk '{split($3, v, "."); exit !(v[1] > 2 || (v[1] == 2 && v[2] >= 40))}'; }
packages() {
  [ -s "$PPA_KEY" ] || curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xE1DD270288B4E6030699E45FA1715D88E1DF1F24" \
    | gpg --dearmor > "$PPA_KEY" || return 1
  echo "deb [signed-by=$PPA_KEY] https://ppa.launchpadcontent.net/git-core/ppa/ubuntu jammy main" \
    > /etc/apt/sources.list.d/git-core-ppa.list
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync sqlite3 git postgresql-client
}
# One apt at a time: two at once lose the dpkg lock. The trust store must not hold emSign ECC Root CA - G3 or Comodo AAA, and a
# reset restores the image's, so ca-certificates and NSS are upgraded every boot.
(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --only-upgrade ca-certificates "libnss3*" > /dev/null
 for try in 1 2 3; do
   command -v rsync > /dev/null && command -v sqlite3 > /dev/null && command -v psql > /dev/null && git_ok && break
   echo "$(date -u +%FT%TZ) packages, try $try"; packages; sleep 20
 done
 git_ok && echo "$(date -u +%FT%TZ) $(git version)" || echo "$(date -u +%FT%TZ) GIT-TOO-OLD: $(git version); drive refuses to merge"
 setsid -f /data/infra/drive.sh > /dev/null 2>&1 < /dev/null) >> /data/boot/packages.log 2>&1 &
(cd /workspace/guard-src && PYTHONPATH=tools/research/src RESEARCH_MACHINES_D=/workspace/steward/research-notes/machines.d LC_ALL=C.UTF-8 \
  setsid -f python3 -m research pods guard --supervise --budgets /workspace/steward/research-notes \
  >> /root/.research/pods/guard-budgets.log 2>&1 < /dev/null)
setsid -f /data/steward/run.sh > /dev/null 2>&1 < /dev/null
setsid -f /data/infra/tmp_sweep.sh > /dev/null 2>&1 < /dev/null
setsid -f /data/console/console-loop.sh > /dev/null 2>&1 < /dev/null
setsid -f /data/infra/store_index.sh > /dev/null 2>&1 < /dev/null
setsid -f /data/infra/store_gc.sh > /dev/null 2>&1 < /dev/null
setsid -f /data/infra/fixtures_export.sh > /dev/null 2>&1 < /dev/null
setsid -f /data/infra/agent_watch.sh > /dev/null 2>&1 < /dev/null
setsid -f /data/infra/cloud_watch.sh > /dev/null 2>&1 < /dev/null
setsid -f /data/infra/control-disk/run.sh > /dev/null 2>&1 < /dev/null
setsid -f /data/watchers/run.sh > /dev/null 2>&1 < /dev/null
