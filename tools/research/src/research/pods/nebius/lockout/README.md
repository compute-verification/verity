# Locking the Nebius nodes down

This directory takes root away from `research` on both nodes and on a CPU node (vy-nebius-cpu-N). Its four tools are installed by
deploy.toml: `vy-admin-setup`, `vy-seal-runs`, `vy-sudo` and `vy-lockout` (`vy-lockout plan NODE` prints the node's steps with their
checks). After the lock-out, agents can read run dirs but not change them. Each ended run is sealed: `vy-seal-runs.timer` makes a run
dir root's and read-only once its status is terminal and custody vouches for it (store.custody's own rule, never time alone).
`research` keeps only the root forms its node's allowlist names, each with its evidence in `lockout.py`, and gets no cluster-admin:
`sudoers/50-vy-research` (`ALLOW`) on both nodes, `sudoers/50-vy-research-cpu` (`CPU_ALLOW`) on a CPU node, staged there under the
same name. Root moves to `vy-admin`, which only Dan and infra's lead hold. Every sudo use is logged to the journal, and vy-admin's
sessions also go under `/var/log/sudo-io`.

A CPU node keeps what a run and the store need: a job user's unit, its stop and what it left in the run dir (`research run
--job-user`, through `vy-sudo`), `vy-seal-runs delete` and `evict`, the start of `vy-lean-cache-daily.service`, and `sudo -n true`.
It may run as itself and as nobody else: its image leaves `ubuntu` in sudo, adm and lxd. It has no loop to switch and no cluster,
so steps 4 to 6 below are not in its plan. Its sshd lets only `research` log in (vm_setup.sh's `AllowUsers`), so step 2 there
also reloads ssh once deploy has staged `/etc/ssh/sshd_config.d/11-vy-admin.conf` (`sudo sshd -t && sudo systemctl reload ssh`). Anything else that ran as root there through `research` moves to a root unit or to
vy-admin before the node's fallback expires (`EXCLUDED`); `vy-lockout audit --node NODE` lists what would be refused, before the
lock-out too, and anyone may run it.

Apply order. Every step runs as vy-admin on the node and has its own rollback. Without `--apply`, a step only says what it would do.

1. Deploy: `research deploy install NODE --by WHO --apply`. It stages the tools, `/usr/local/lib/vy-lockout/` and node 1's
   `/etc/vy/env/root.env`, and starts nothing. From this deploy on, `research run --job-user USER` asks `vy-sudo` for its
   unit, and `vy-sudo` starts one only as a job user: an account root has listed in the group `vy-job`. No tool here makes a
   job user. Before the deploy, on the node, run `sudo groupadd -f --system vy-job`, then `sudo usermod -aG vy-job USER` for
   each account made for jobs and nothing else, or for a new one
   `sudo useradd --system --no-create-home --shell /usr/sbin/nologin --groups vy-job USER`. After the lock-out only vy-admin
   can. `getent group vy-job` prints every job user. Rollback: `research deploy attic`.
2. vy-admin: `sudo vy-admin-setup KEYS` with a file of public keys copied to the node. The keys are never taken from a secret
   store and never put in git. Then log in as vy-admin and run `sudo -n true`. Rollback: `userdel -r vy-admin` and remove
   `/etc/sudoers.d/10-vy-admin`.
3. Seal: run `vy-seal-runs seal --dry-run` and read its output, then `systemctl enable --now vy-seal-runs.timer` and
   `vy-lockout roots --apply` (the runs roots become root's and sticky). Rollback: disable the timer and run
   `vy-lockout roots --back --apply`. A sealed dir's former owner and mode are in `/var/lib/vy-seal-runs/sealed.jsonl`.
4. Switch: `vy-lockout switch NODE --unit U --apply`, one loop at a time in `SWITCH` order. It stops the tmux loop and starts
   its unit. Rollback: `--back`, newest first.
5. Root units: `vy-lockout root-units vy-nebius-1` runs the loops that need cluster-admin as root. It refuses while any
   code or directory such a loop runs or writes isn't root's alone. Rollback: `--back`.
6. Kubeconfig: `vy-lockout kubeconfig --apply` gives research a read-only token (`research-view.yaml`) in place of its admin
   certificate. Then run `k3s certificate rotate` in a quiet window. Rollback: `--back`.
7. Harden: `vy-lockout harden --apply` takes ubuntu out of lxd, takes `cap_sys_ptrace` off py-spy and puts research in
   systemd-journal. Rollback: `--back`.
8. Sudoers: once `vy-lockout preflight NODE` passes, `vy-lockout sudoers --node NODE --apply` installs the node's allowlist and
   a 24 h fallback granting ALL, and moves cloud-init's `NOPASSWD: ALL` grant aside. Preflight refuses while the admin login
   doesn't work: it wants an ssh login as vy-admin newer than its keys and its sudoers file, and the step itself run from a
   vy-admin login. `vy-sudo-fallback-expire.timer` removes the fallback at its expiry, and only vy-admin can put it back
   (`vy-lockout fallback on`). From this step on, `research deploy install` runs `sudo python3` and so needs vy-admin's login
   (`research deploy install NODE --login vy-admin`). Rollback: `vy-lockout rollback --apply`.
9. Watch: during the 24 h, `vy-lockout audit` counts what research ran that the allowlist would refuse, and after them what
   sudo did refuse. It prints command shapes, never arguments. Fix each such form by PR before the fallback expires. Once the fallback is gone, research's
   `sudo -n true` still passes, since it is on the allowlist, and any other root command (`sudo -n id`) is refused.

On a sealed run, `research retention gc` and `rm` used to fail: gc reported it as read-only, and rm refused it as
read-only. `research data evict --runs` would have raised on its first unlink. Now each passes the whole run dir to the root
helper once its own checks pass (store/sealed.py): gc and rm run `sudo -n vy-seal-runs delete DIR`, and evict runs
`vy-seal-runs evict DIR --min-bytes N`. The helper checks again, as root, that the dir sits directly under a runs root, is
sealed and still has custody. Evict removes only the files `.custody` lists, at the size it recorded and unchanged since.
A path inside a sealed run is refused.
