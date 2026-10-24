"""Red-team #3925 window_charge: can a prover lower the charge?
Differential reference_window vs code_v1's true segment length on adversarial windows, plus the four named vectors."""
import importlib.util, json, random, sys
from pathlib import Path

ENC = Path("/tmp/rt/p18charge/benchmarks/network_e2e/encodings")
def load(name):
    spec = importlib.util.spec_from_file_location(f"enc_{name}", ENC / f"{name}.py")
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m
    spec.loader.exec_module(m); return m
CV = load("code_v1")

from verity.core.protocols.certifier import window_charge as WC, arith_encode as AE

FIX = Path("/tmp/rt/p18charge/verity/core/protocols/certifier/tests/fixtures/arith_encode_vectors.json")
BOOK = json.loads(FIX.read_text())["streams"][0]["decisions"]["book"]
cvbook = CV.Book.from_json(BOOK)

def true_charge(arrived, hits, order):
    """code_v1's actual segment length for one window from a fresh state."""
    win = CV.Window(arrived=tuple(arrived), hits=tuple(hits), resumes=(),
                    order=tuple((p, (r,)) for p, r in order), lag=())
    return CV.encode([win], cvbook).lens[0]

def wc_charge(arrived, hits, order):
    N, H, O = len(arrived), len(hits), len(order)
    wb = WC.window_book(BOOK, N, H, O)[0]
    ok, bits, nxt = WC.reference_window(N, H, O, wb, WC.first_carry(BOOK),
                                        [1]*N, list(arrived), [1]*H, list(hits),
                                        [1]*O, [p for p, _ in order], [r for _, r in order])
    return ok, bits, nxt

def mono_order(rng, O, pos0=0):
    """O order entries with strictly increasing positions and arbitrary ranks (as reality would commit)."""
    order, pos = [], pos0
    for _ in range(O):
        g = rng.choice([0, 1, 2, 255, rng.randrange(1 << 16), (1 << 16) + rng.randrange(1 << 10),
                        rng.randrange(1 << 24), (1 << 24) - 1, 1 << 24])
        p = pos + g
        if p >= (1 << 32) - 1:
            break
        r = rng.choice([0, 1, rng.randrange(1 << 10), rng.randrange(1 << 24), 1 << 24])
        order.append((p, r)); pos = p + 1
    return order

print("=== DIFFERENTIAL: reference_window bits  vs  code_v1 true segment length ===")
rng = random.Random(3925)
bad = 0; n = 0
cases = []
# decisions: long runs (saturation), all-ones, alternating, random
for arr in ([0]*20, [1]*20, [0,1]*10, [1]+[0]*19, [0]*19+[1]):
    cases.append((arr, [], []))
# hits at boundaries
cases.append(([1,0,1], [0, 1, 1<<15, (1<<16)-1, 1<<16], []))
# order entries: wide-offset gaps, negative & large rank steps, decreasing ranks
for seed in range(40):
    r = random.Random(seed)
    arr = [r.randrange(2) for _ in range(r.randrange(6))]
    hits = [r.choice([0,1,1<<15,(1<<16)-1,1<<16]) for _ in range(r.randrange(4))]
    order = mono_order(r, r.randrange(5))
    cases.append((arr, hits, order))
for arrived, hits, order in cases:
    ok, bits, _ = wc_charge(arrived, hits, order)
    tc = true_charge(arrived, hits, order)
    n += 1
    if not ok or bits != tc:
        bad += 1
        print(f"  MISMATCH ok={ok} wc={bits} code_v1={tc}  arr={arrived[:6]}.. hits={hits} order={order[:3]}..")
print(f"  {n} windows, {bad} mismatches (undercharge if wc < code_v1)")

print("\n=== MULTI-WINDOW chain: WC.session lens vs code_v1.encode lens ===")
rng = random.Random(7)
wins_cv, wins_wc = [], []
N = H = O = 4
pos = 0
for _ in range(6):
    arr = [rng.randrange(2) for _ in range(rng.randrange(N+1))]
    hits = [rng.choice([0,1,1<<15]) for _ in range(rng.randrange(H+1))]
    order = mono_order(rng, rng.randrange(O+1), pos)
    if order: pos = order[-1][0] + 1
    wins_cv.append(CV.Window(arrived=tuple(arr), hits=tuple(hits), resumes=(),
                             order=tuple((p,(r,)) for p,r in order), lag=()))
    # pad to N,H,O slots (disabled slots change nothing)
    en = [1]*len(arr) + [0]*(N-len(arr)); bits_ = list(arr)+[0]*(N-len(arr))
    hen = [1]*len(hits)+[0]*(H-len(hits)); short = list(hits)+[0]*(H-len(hits))
    oen = [1]*len(order)+[0]*(O-len(order))
    pp = [p for p,_ in order]+[0]*(O-len(order)); rr=[r for _,r in order]+[0]*(O-len(order))
    wins_wc.append((en, bits_, hen, short, oen, pp, rr))
cv_lens = list(CV.encode(wins_cv, cvbook).lens)
wc_lens = [x[0] for x in WC.session(N, H, O, BOOK, wins_wc)]
print(f"  code_v1: {cv_lens}")
print(f"  WC     : {wc_lens}")
print(f"  equal: {cv_lens == wc_lens}")

print("\n=== VECTOR: count disagrees with decisions ===")
# There is NO independent count input; n_arrived = popcount(en) internally.
import inspect
sig = [name for name,_n,_w in WC._window_in(type('S',(),{'N':4,'H':2,'O':2}))]
print(f"  window_charge inputs: {sig}")
print(f"  any explicit count field? {[s for s in sig if s.startswith('n_')]}")
# try: enable 2 slots but can we charge as if 3 decisions? count is derived -> no knob. Show popcount tie:
for en in ([1,1,0,0],[1,0,1,0],[1,1,1,0]):
    _, bits, _ = WC.reference_window(4,0,0, WC.window_book(BOOK,4,0,0)[0], WC.first_carry(BOOK), en, [1,1,1,0])
    print(f"  en={en} popcount={sum(en)} charge={bits}")

print("\n=== VECTOR: resume (no slot exists) ===")
print(f"  n_resumes coded as count of () -> always 0; no resume input field: {'resume' not in ' '.join(sig)}")

print("\n=== VECTOR: reorder order entries (non-monotonic positions refused) ===")
wb = WC.window_book(BOOK,0,0,2)[0]
for p in ([10, 20], [20, 10], [10, 10]):
    ok,_,_ = WC.reference_window(0,0,2, wb, WC.first_carry(BOOK), [],[],[],[],[1,1], p, [0,0])
    print(f"  positions {p}: ok={ok}")

print("\n=== VECTOR: truncate (drop a slot lowers the count; cannot drop a symbol while keeping count) ===")
for en in ([1,1,1],[1,1,0]):
    _, bits, nxt = WC.reference_window(3,0,0, WC.window_book(BOOK,3,0,0)[0], WC.first_carry(BOOK), en, [0,0,0])
    print(f"  en={en} count={sum(en)} charge={bits}")

print("\n=== CIRCUIT differential: evaluate_windows (the prover's circuit) vs code_v1 and reference ===")
# fixed shape so one evaluation; pad disabled slots. Target wide-offset gaps (bin>=18) and rank steps explicitly.
N, H, O = 6, 5, 4
wb = WC.window_book(BOOK, N, H, O)[0]
circ_cases, specs = [], []
rng = random.Random(99)
wide_gaps = [1<<17, (1<<17)+123, 1<<20, (1<<23)+(5<<15)+7, (1<<24)-1, 1<<24]
for i in range(30):
    arr = [rng.randrange(2) for _ in range(rng.randrange(N+1))]
    hits = [rng.choice([0,1,1<<15,(1<<16)-1,1<<16]) for _ in range(rng.randrange(H+1))]
    nord = rng.randrange(O+1)
    order, pos = [], 0
    for _ in range(nord):
        g = rng.choice(wide_gaps + [0,1,2])
        p = pos + g
        r = rng.choice([0,1,rng.randrange(1<<24),1<<24, max(0,pos-1)])  # incl. decreasing ranks
        order.append((p,r)); pos = p+1
    specs.append((arr, hits, order))
    en=[1]*len(arr)+[0]*(N-len(arr)); bits_=list(arr)+[0]*(N-len(arr))
    hen=[1]*len(hits)+[0]*(H-len(hits)); short=list(hits)+[0]*(H-len(hits))
    oen=[1]*len(order)+[0]*(O-len(order))
    pp=[p for p,_ in order]+[0]*(O-len(order)); rr=[r for _,r in order]+[0]*(O-len(order))
    circ_cases.append((wb, WC.first_carry(BOOK), en, bits_, hen, short, oen, pp, rr))
circ = WC.evaluate_windows(N, H, O, circ_cases)
bad = 0
for (arr, hits, order), (ok, bits, _nxt) in zip(specs, circ):
    tc = true_charge(arr, hits, order)
    refok, refbits, _ = wc_charge(arr, hits, order)
    if not ok or bits != tc or (refok and refbits != bits):
        bad += 1
        print(f"  MISMATCH circuit ok={ok} bits={bits} code_v1={tc} ref={refbits}  arr={arr} hits={hits} order={order}")
print(f"  {len(specs)} windows through the circuit, {bad} mismatches (wide-offset gaps & decreasing rank steps included)")
