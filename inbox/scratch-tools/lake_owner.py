import os, re, subprocess, time
out = subprocess.run(['ps','-eo','pid,ppid,etimes,comm'],capture_output=True,text=True).stdout.splitlines()[1:]
rows = {int(l.split()[0]): (int(l.split()[1]), int(l.split()[2]), l.split(None,3)[3].strip()) for l in out}
def cmd(pid):
    try: return open(f'/proc/{pid}/cmdline','rb').read().decode('utf-8','replace').replace('\0',' ')
    except Exception: return ''
def cwd(pid):
    try: return os.readlink(f'/proc/{pid}/cwd')
    except Exception: return '?'
now = time.strftime('%H:%M:%SZ', time.gmtime())
print('now', now)
for pid,(pp,et,c) in sorted(rows.items()):
    if c != 'lake': continue
    chain=[]; p=pid; run=None; test=None
    while p in rows and p != 1 and len(chain) < 25:
        s = cmd(p) + ' ' + cwd(p)
        m = re.search(r'r2026\d{4}-\d{6}-[0-9a-f]{4}', s)
        if m and not run: run = m.group(0)
        t = re.search(r'(tests?/[\w/]+\.py(::\w+)?|pytest[^|]{0,120})', cmd(p))
        if t and not test: test = t.group(0)[:160]
        chain.append(f"{rows[p][2]}")
        p = rows[p][0]
    print(f'lake {pid} up {et//60} min, cwd ...{cwd(pid)[-64:]}\n   run={run} test={test}\n   chain={" < ".join(chain[:12])}')
