import sys,re,subprocess,collections
tree=sys.argv[1]
d=subprocess.run(['git','-C',tree,'diff','--cached','-U0','-M'],capture_output=True,text=True).stdout
ID=re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
pairs=collections.defaultdict(set); f=None; minus=[]; plus=[]
def flush():
    global minus,plus
    if len(minus)==len(plus):
        for a,b in zip(minus,plus):
            ta,tb=ID.findall(a),ID.findall(b)
            if len(ta)==len(tb):
                for x,y in zip(ta,tb):
                    if x!=y: pairs[(x,y)].add(f)
            else:
                for x in ta:
                    if re.search(r'(?i)draw|drew',x) and x not in tb: pairs[(x,'?')].add(f)
    minus,plus=[],[]
for l in d.splitlines():
    if l.startswith('+++ '): flush(); f=l[6:]; continue
    if l.startswith('--- ') or l.startswith('diff ') or l.startswith('index '): continue
    if l.startswith('@@'): flush(); continue
    if l.startswith('-'): minus.append(l[1:])
    elif l.startswith('+'): plus.append(l[1:])
flush()
print("old\tnew\tfiles")
for (x,y),fs in sorted(pairs.items()):
    print(f"{x}\t{y}\t{len(fs)}\t{' '.join(sorted(fs)[:4])}")
