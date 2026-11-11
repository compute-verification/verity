import json,collections,re,sys
INST_RE = re.compile(r"(^Pouw\.PearlC\.(Prices\.(h100|sm120|sm120Loop)|devH100|devSm120v1|devSm120v2|sm120E4m3|sm120E4m3K32|sm120Step|sm120StepAt)$)"
  r"|(^Pouw\.PearlC\.W1\.(W1Prices\.(sm120|sm120Loop|h100|fp4Sm120)|W1Device\.(sm120At|h100At|h100|sm120|sm120Loop|fp4Sm120|s8Sm120At|s8Sm120|s8H100At|s8H100)|sm120Frags|sm120S8Frags|h100Frags|h100S8Frags)$)"
  r"|(Fp4Prices\.(sm120|sm120Issue|sm120At)$)|(devFp4Sm120\w*$)|(sm120Nvf4$)|(sm120Mxf4$)|(hopperE4m3K32$)|(hopperBf16K16$)|(hopperStep$)"
  r"|(h100(Nominal|Legacy|LoopFree|Measured|PeakTc|Floor)$)|(nominalMac$)"
  r"|(^Accounting\.(h200S8Price|w1Price)$)"
  r"|(^Verity\.Accel\.(H200|Sm120)\.(card|clock|tensorClock)$)"
  r"|(h200Sheet$)|(h200Measured$)|(hostH200$)|(hostSm120$)|(pub(Mm|Sha|Best)(H200|Sm120)$)"
  r"|(^Pous\.StreamedWipe\.(h200|h200Disks)$)|(FpG1.*\.sm120$)")
HEAD_MOD = re.compile(r"^Definitions\.Accounting\.(Headline|HeadlineDc)\.")
def load(p): return json.load(open(p))
def inst_map(r):
    defs=r["defs"]; out={}
    for g,(ms,ds,_) in r["G"].items():
        s=set()
        for d in ds:
            m,c,l=defs[str(d)]
            if INST_RE.search(c): s.add(c)
            elif HEAD_MOD.match(m) and not m.endswith("Encoding"): s.add("@"+m.split("Definitions.Accounting.")[1])
        out[g]=s
    return out
if __name__=="__main__":
    r=load(sys.argv[1]); lock=json.load(open(sys.argv[2]))
    im=inst_map(r)
    by=collections.defaultdict(list)
    for g,s in im.items():
        for c in s: by[c].append(g)
    for c in sorted(by): print(f"{len(by[c]):4d} {c}")
    hit=[g for g,s in im.items() if s]
    print("guarantees reading an instance:",len(hit),"of",len(im))
    own=collections.Counter(lock["guarantees"][g]["owner"] for g in hit); print(own)
    json.dump({g:sorted(s) for g,s in im.items() if s}, open(sys.argv[3],"w"), indent=0)
