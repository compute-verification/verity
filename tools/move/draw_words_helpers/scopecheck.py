import sys,ast,subprocess
tree,tip=sys.argv[1],sys.argv[2]
def scopes(src):
    t=ast.parse(src); out=[]
    def walk(node,path):
        names=set(); attrs=set(); refs=set()
        for n in ast.walk(node):
            if isinstance(n,ast.Name): refs.add(n.id)
            if isinstance(n,ast.arg): names.add(n.arg)
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and n is not node: names.add(n.name)
            if isinstance(n,ast.Attribute): attrs.add(n.attr)
            if isinstance(n,(ast.Import,ast.ImportFrom)):
                for a in n.names: names.add((a.asname or a.name).split('.')[0])
            if isinstance(n,ast.keyword) and n.arg: attrs.add('kw:'+n.arg)
        out.append((path,len(names|refs),len(attrs)))
        for c in ast.iter_child_nodes(node):
            for n in ast.walk(c):
                pass
        for n in ast.walk(node):
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and n is not node and getattr(n,'_seen',None) is None:
                n._seen=1; walk(n,path+'/'+n.name)
    walk(t,'')
    return out
st=subprocess.run(['git','-C',tree,'diff','--cached','--name-status','-M'],capture_output=True,text=True).stdout.splitlines()
for l in st:
    p=l.split('\t'); a=p[1]; b=p[-1]
    if not b.endswith('.py'): continue
    o=subprocess.run(['git','-C',tip,'show','HEAD:'+a],capture_output=True,text=True).stdout
    n=open(f'{tree}/{b}').read()
    try: so=scopes(o); sn=scopes(n)
    except SyntaxError as e: print("SYNTAX",b,e); continue
    if len(so)!=len(sn): print("SCOPES differ",b); continue
    for (pa,a1,b1),(pb,a2,b2) in zip(so,sn):
        if a2<a1 or b2<b1: print(f"MERGE {b} {pb} names {a1}->{a2} attrs {b1}->{b2}")
