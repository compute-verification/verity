import re, json, sys, collections
def load(log):
    mods={}; defs={}; G={}; X={}
    for line in open(log, errors='replace'):
        line=line.rstrip("\n")
        if not line.startswith("SP|"): continue
        p=line.split("|")
        if p[1]=="M": mods[int(p[2])]=p[3]
        elif p[1]=="D": defs[int(p[2])]=(mods[int(p[3])], p[4], p[5])
        elif p[1]=="G":
            ms=[int(x) for x in re.findall(r"\d+", p[3])]
            ds=[int(x) for x in re.findall(r"\d+", p[4])]
            G[p[2]]=(ms,ds,p[5])
        elif p[1]=="X": X[p[2]]=p[3]
    return mods,defs,G,X
if __name__=="__main__":
    mods,defs,G,X=load(sys.argv[1])
    print(len(mods),len(defs),len(G),len(X))
    json.dump({"mods":mods,"defs":defs,"G":G,"X":X}, open(sys.argv[2],"w"))
