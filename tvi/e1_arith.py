"""Arithmetic behind tvi/20261024T2130Z-draft-tvi-e1-statement.md (E1, phase 1). Plain Python, no inputs.

Units: k in bits (1 kB = 8e3 bits), c in FLOPs, P = DeepSeek-V4-Pro's 49B active parameters, h = 17 bits per token.
"""
kB=8e3; MB=8e6
c=1.4e17; beta=0.4; d=7000; d7=7168; h=17; P=49e9
print("== R1 headline rho = c b/(6 d k)")
for k,lab in [(10*kB,'10kB'),(MB,'1MB')]:
    print(lab, c*beta/(6*d*k), "(d=7168:", c*beta/(6*d7*k),")")
print("== eta = c/(k phi), phi=7e9")
for k,lab in [(10*kB,'10kB'),(MB,'1MB')]:
    print(lab, "eta=", c/(k*7e9))
print("tokens of compute in c at 2P+attn ~ phi*h:", c/(7e9*h), " at 2P:", c/(2*P))
print("tokens in 10kB:", 10*kB/h, " fwd share of c (phi*h per tok):", 10*kB*7e9/c, " fwd+bwd (3x):", 3*10*kB*7e9/c)
print("== rho_free = phi beta/(6d) for listed phi")
for phi in [7e9,6.6e9,8.9e9,6.9e10]:
    print(phi, [round(phi*b/(6*dd)) for b in (0.4,) for dd in (d,d7)], " b=.03,.1:", round(phi*0.03/(6*d)), round(phi*0.1/(6*d)))
print("phi needed for 1.1e5 at b=.4,d=7000:", 1.1e5*6*d/0.4, " for 3e4 at b=.1:", 3e4*6*d/0.1)
print("rho at c=1.4e17,10kB, b=.03..0.1:", c*0.03/(6*d*10*kB), c*0.1/(6*d*10*kB))
print("== kappa = phi h/(6P)")
for phi in [7e9,6.6e9,8.9e9,6.9e10]:
    print(phi, phi*h/(6*P))
print("== cheap inputs: k > 4 d h (beta_w=beta)")
print("4dh bits:", 4*d*h, "=", 4*d*h/8e3, "kB;  1MB / that:", MB/(4*d*h), " d=7168:", 4*d7*h/8e3, "kB")
print("== cap condition")
for bx in (0.1,):
    bw=bv=0.4; x=bv*bx
    kstar_note=d**2*bw*x/bv; kstar=4*d**2*bw*x/bv
    print("note k* =", kstar_note/8e3,"kB ; exact (AM-GM) k* =", kstar/8e3, "kB ; d=7168:", 4*d7**2*bw*x/bv/8e3)
print("d^2 beta (all equal):", d**2*0.4/8e6, "MB  d7:", d7**2*0.4/8e6)
def maxRQ(k,din,dout,bw,bx):
    R=k/(2*din*bw)
    if R<=dout: return k*k/(4*din*din*bw*bx)
    return dout*(k-din*dout*bw)/(din*bx)
for (din,dout,bw,bx,bv) in [(d7,d7,.4,.04,.4),(d7,4*d7,.4,.4,.4),(d7,d7,.4,.4,.4),(d7,3072,.4,.04,.4)]:
    ks=[k for k in [x*1e4 for x in range(1,100000)] if maxRQ(k,din,dout,bw,bx)*bv>k]
    print("din,dout,bw,bx,bv",din,dout,bw,bx,bv," first failing k:", (ks[0]/8e3 if ks else None),"kB")
print("== cliff")
for F in (14000,256):
    for kap in (1/3, 7e9*h/(6*P), 2/3):
        print("F",F,"kappa",round(kap,3)," program (2/3)*2*F/h:",round((2/3)*2*F/h), " kappa*X*F/h:",round(kap*2*F/h)," with amortization at S=k (m=2):", round(kap*(2+2*F/h)*2))
def W(S,k,F,mmax=200):
    best=None
    for m in range(1,mmax):
        if S/m>=k: continue
        X=2*(m-1)
        w=(m*h+X*F)/(1-S/(m*k))
        if best is None or w<best[0]: best=(w,m)
    return best
for F in (14000,256):
  for s in (0.5,0.9,0.99,1.01,1.5,2,4,10,24):
    w,m=W(s*MB,MB,F)
    print("F",F,"S/k",s,"m",m,"W/h",round(w/h,1)," rho(eta=1,kappa=1/3)",round(w/h/3,1))
print("full training W_III/h = beta P/(d h):", beta*P/(d7*h), " rho:", beta*P/(d7*h)/3)
print("S* where 8(S/k)F = beta P/d:", beta*P/d7/(8*14000), beta*P/d7/(8*256))
print("== accumulator")
for k,lab in [(10*kB,'10kB'),(MB,'1MB')]:
    print(lab, "c h b/(6k^2) b=1:", c*h/(6*k*k), " tile (2/3) c beta h/k^2:", (2/3)*c*beta*h/k**2, " (1/3)c beta h /k^2:", c*beta*h/(3*k*k))
print("== rho table")
full10=c*beta/(6*d*10*kB); full1=c*beta/(6*d*MB)
for r in (16,256):
    print("LoRA r",r," r/d:", full10*r/d7, full1*r/d7, " 3r/d:", 3*full10*r/d7, 3*full1*r/d7)
print("FP8 beta=.1:", c*0.1/(6*d*10*kB), c*0.1/(6*d*MB), " beta=.03:", c*0.03/(6*d*10*kB))
V=129280; Tt=2*P+6*d7*V
print("head-only per-value 10kB: c*d*beta/(k*Ttok):", c*d7*beta/(10*kB*Tt), " Ttok",Tt)
print("head-only tile attack 1MB:", c*d7*4*d7*0.4*h/(MB**2*Tt), " cap k*=4 d h:",4*d7*h/8e3,"kB")
print("LoRA rank-8 per layer bits (7 mats, 2 bits/param):", 7*2*8*d7*2/8e3,"kB; x61:", 61*7*2*8*d7*2/8e6,"MB")
print("one dense matrix d^2 beta:", d7**2*0.4/8e6,"MB")
print("== join vs production k*=b^2 d/(beta r^2)")
for b,bt in [(4,0.4),(4,0.004),(16,0.4)]:
    print(b,bt, b*b*d7/(bt*100)/8,"bytes")
print("== hypothetical additive-and-priced (no Lean theorem states it): ratio 4 d cw ca/(beta k)")
for cw,ca in [(6.7,16),(0.4,16),(2.8,8),(0.7,4)]:
    for k,lab in [(10*kB,'10kB'),(MB,'1MB')]:
        print(cw,ca,lab, round(4*d7*cw*ca/(beta*k),3))
print("== Lean forms vs production, regime III, ratio to c beta/(6 d k) (d=7168, beta=0.4)")
for name,gamma,cw,ca in [("TwoCosts uniform cw=2.3 (bf16 cheapest 1%)",1,2.3,16),
                         ("Additive uniform cw=2.3",4,2.3,16),
                         ("Priced mean cw=6.7 (branch)",1,6.7,16),
                         ("Priced FP8 mean cw=2.8, ca=8",1,2.8,8),
                         ("Additive uniform FP8 cw=0.45, ca=8",4,0.45,8)]:
    print(name, [round(gamma*d7*cw*ca/(beta*k),3) for k in (10*kB,MB)])
print("k* for d_in = 3072, 512, 1536 at bx=bv/10, bw=bv:", [round(4*di**2*0.4*0.1/8e3,1) for di in (3072,512,1536)], "kB")
