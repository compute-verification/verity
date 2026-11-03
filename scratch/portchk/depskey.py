import hashlib,json,subprocess,sys
def show(rev,path):
    p=subprocess.run(['git','show',f'{rev}:{path}'],capture_output=True,cwd='/agent/repos/verity')
    return p.stdout if p.returncode==0 else None
def key(rev,pkg):
    m=show(rev,f'{pkg}/lake-manifest.json'); t=show(rev,f'{pkg}/lean-toolchain')
    if m is None or t is None: return None
    doc=json.loads(m); fetched=[d for d in doc.get('packages',[]) if d.get('type')!='path']
    if not fetched: return None
    canon=json.dumps({"packagesDir":doc.get("packagesDir"),"packages":fetched},sort_keys=True,separators=(",",":"))
    return hashlib.sha256(canon.encode()+b"\0"+t).hexdigest()[:12]
revs=sys.argv[1:]
for pkg in ['Security','Security/Properties','Security/Proofs','verity/core/service/flock','verity/core/protocols/pouw/lean/check','verity/core/protocols/network/nci/lean']:
    print(pkg.ljust(40),' '.join(f'{r[:14]}={key(r,pkg)}' for r in revs))
