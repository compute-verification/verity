"""DEVELOPMENT BINDING, NOT FOR REVIEW: scratch Flock pieces for core's BitAnd_v1, E4m3ToF32_v1 and F32ToE2m1Sat_v1, which
Flock has no piece for on this branch (@circuits' #885 lands core's own). `stage.py` puts them in `ir_lower.PIECES` for the
row units only, and every result staged with them is labelled 'core3 scratch'. E4m3ToF32 is a 256-entry table read
(`fp.lookup`, 201 ANDs, checked on every byte); F32ToE2m1Sat is seven threshold comparisons and a NaN select (180 ANDs,
checked on 40k words)."""
from verity.ml.circuits.ops import prims as P
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.boolean.forms import NOT, const_bits, mux, sub
from verity.ml.circuits.silicon import cast

E4M3_TABLE = [cast.e4m3_to_f32_word(b) for b in range(256)]


def e2m1(C, x):
    mag = list(x[:31])
    f = [NOT(sub(C, mag, const_bits(b + (0 if up else 1), 31), 31)[1]) for b, up in cast.E2M1_UP]
    code = [f[0] ^ f[1] ^ f[2] ^ f[3] ^ f[4] ^ f[5] ^ f[6], f[1] ^ f[3] ^ f[5], f[3], x[31]]
    nan = sub(C, const_bits(0x7F800000, 31), mag, 31)[1]
    return mux(C, nan, const_bits(P.E2M1_NAN_CODE, 4), code)


PIECES = {"BitAnd_v1": lambda C, a, b: [C.AND(a[0], b[0])],
          "E4m3ToF32_v1": lambda C, a: fp.lookup(C, list(a), E4M3_TABLE, 32),
          "F32ToE2m1Sat_v1": e2m1}
