import re,sys,glob
seen=set()
for f in sys.argv[1:]:
    lines=open(f,errors='replace').read().splitlines()
    i=0
    while i<len(lines):
        m=re.match(r'error: (\S+\.lean):(\d+):\d+: (.*)',lines[i])
        if m and (m.group(1),m.group(2)) not in seen:
            seen.add((m.group(1),m.group(2)))
            blk=[lines[i]]
            j=i+1
            while j<len(lines) and j<i+120 and not re.match(r'(error|warning):|[✔⚠✖] \[|lean-build|Some required',lines[j]):
                blk.append(lines[j]); j+=1
            keep=[blk[0]]
            for k,l in enumerate(blk[1:],1):
                if '↦' in l or 'not exposed' in l or 'stuck' in l or 'Unknown' in l:
                    keep.extend(blk[k:k+3] if 'stuck' in l else [l])
            print(f.split('/')[-1], '|', ' / '.join(x.strip() for x in keep)[:700])
        i+=1
