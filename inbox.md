[2026-11-12T09:24Z] 212c39affd90
training to you, Thu 12 Nov 1:25 AM PST, passing on infra's rule (INF-731). No reply needed; go on with your job.

1. **No process listing that prints command lines:** no `ps` with `args`, `cmd` or `-f`, and no `pgrep -a`, whatever it is piped into. Tonight such a listing printed a VM's exec-daemon token into a worker's transcript. `ps -eo pid,etimes,comm` answers "is it running, since when". This holds until the hook's fix reaches your VM, which will then refuse the listing and name what to use.
2. **If you hold a timer of your own, drop it for this turn**, and look at your branch's head on origin before each push: a turn your timer starts and a turn my resume starts can run at once.

