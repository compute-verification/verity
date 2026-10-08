/-
PROTOCOL.md §2.5: SHA-256 (FIPS 180-4) and BLAKE3 (compression, hash, keyed hash).
-/
namespace Flock

abbrev Digest := ByteArray

namespace Sha256

def k : Array UInt32 := #[
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2]

def iv : Array UInt32 := #[0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]

@[inline] def rotr (x : UInt32) (n : UInt32) : UInt32 := (x >>> n) ||| (x <<< (32 - n))

/-- One compression of the 64-byte block of `data` at `off`. -/
def compress (h : Array UInt32) (data : ByteArray) (off : Nat) : Array UInt32 := Id.run do
  let mut w : Array UInt32 := Array.mkEmpty 64
  for i in [0:16] do
    let b := fun j => (data.get! (off + 4 * i + j)).toUInt32
    w := w.push ((b 0 <<< 24) ||| (b 1 <<< 16) ||| (b 2 <<< 8) ||| b 3)
  for i in [16:64] do
    let x := w[i - 15]!
    let y := w[i - 2]!
    let s0 := rotr x 7 ^^^ rotr x 18 ^^^ (x >>> 3)
    let s1 := rotr y 17 ^^^ rotr y 19 ^^^ (y >>> 10)
    w := w.push (w[i - 16]! + s0 + w[i - 7]! + s1)
  let mut a := h[0]!
  let mut b := h[1]!
  let mut c := h[2]!
  let mut d := h[3]!
  let mut e := h[4]!
  let mut f := h[5]!
  let mut g := h[6]!
  let mut hh := h[7]!
  for i in [0:64] do
    let s1 := rotr e 6 ^^^ rotr e 11 ^^^ rotr e 25
    let ch := (e &&& f) ^^^ ((~~~e) &&& g)
    let t1 := hh + s1 + ch + k[i]! + w[i]!
    let s0 := rotr a 2 ^^^ rotr a 13 ^^^ rotr a 22
    let maj := (a &&& b) ^^^ (a &&& c) ^^^ (b &&& c)
    let t2 := s0 + maj
    hh := g
    g := f
    f := e
    e := d + t1
    d := c
    c := b
    b := a
    a := t1 + t2
  return #[h[0]! + a, h[1]! + b, h[2]! + c, h[3]! + d, h[4]! + e, h[5]! + f, h[6]! + g, h[7]! + hh]

/-- SHA-256 of `msg`. -/
def hash (msg : ByteArray) : Digest := Id.run do
  let len := msg.size
  let mut data := msg
  data := data.push 0x80
  while data.size % 64 != 56 do
    data := data.push 0
  let bits := len * 8
  for i in [0:8] do
    data := data.push ((bits >>> (8 * (7 - i))) % 256).toUInt8
  let mut h := iv
  for blk in [0:data.size / 64] do
    h := compress h data (64 * blk)
  -- the state's eight words, big-endian: 32 bytes whatever the loops computed (`Sha256.hash_size`)
  return ⟨Array.ofFn (n := 32) fun i => (h[i.val / 4]! >>> (24 - 8 * (i.val % 4)).toUInt32).toUInt8⟩

end Sha256

namespace Sha512

def k : Array UInt64 := #[
  0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc, 0x3956c25bf348b538, 0x59f111f1b605d019,
  0x923f82a4af194f9b, 0xab1c5ed5da6d8118, 0xd807aa98a3030242, 0x12835b0145706fbe, 0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2,
  0x72be5d74f27b896f, 0x80deb1fe3b1696b1, 0x9bdc06a725c71235, 0xc19bf174cf692694, 0xe49b69c19ef14ad2, 0xefbe4786384f25e3,
  0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65, 0x2de92c6f592b0275, 0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5,
  0x983e5152ee66dfab, 0xa831c66d2db43210, 0xb00327c898fb213f, 0xbf597fc7beef0ee4, 0xc6e00bf33da88fc2, 0xd5a79147930aa725,
  0x06ca6351e003826f, 0x142929670a0e6e70, 0x27b70a8546d22ffc, 0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed, 0x53380d139d95b3df,
  0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6, 0x92722c851482353b, 0xa2bfe8a14cf10364, 0xa81a664bbc423001,
  0xc24b8b70d0f89791, 0xc76c51a30654be30, 0xd192e819d6ef5218, 0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8,
  0x19a4c116b8d2d0c8, 0x1e376c085141ab53, 0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8, 0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb,
  0x5b9cca4f7763e373, 0x682e6ff3d6b2b8a3, 0x748f82ee5defb2fc, 0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec,
  0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915, 0xc67178f2e372532b, 0xca273eceea26619c, 0xd186b8c721c0c207,
  0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178, 0x06f067aa72176fba, 0x0a637dc5a2c898a6, 0x113f9804bef90dae, 0x1b710b35131c471b,
  0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc, 0x431d67c49c100d4c, 0x4cc5d4becb3e42b6, 0x597f299cfc657e2a,
  0x5fcb6fab3ad6faec, 0x6c44198c4a475817]

def iv : Array UInt64 := #[0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
  0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179]

@[inline] def rotr (x : UInt64) (n : UInt64) : UInt64 := (x >>> n) ||| (x <<< (64 - n))

/-- The working registers `(a, b, c, d, e, f, g, h)`. -/
abbrev Regs := UInt64 × UInt64 × UInt64 × UInt64 × UInt64 × UInt64 × UInt64 × UInt64

/-- The big-endian word of the 8 bytes of `data` at `o`. -/
@[inline] def word (data : ByteArray) (o : Nat) : UInt64 :=
  (((((((((0 : UInt64) <<< 8 ||| (data.get! (o + 0)).toUInt64) <<< 8 ||| (data.get! (o + 1)).toUInt64) <<< 8 |||
    (data.get! (o + 2)).toUInt64) <<< 8 ||| (data.get! (o + 3)).toUInt64) <<< 8 ||| (data.get! (o + 4)).toUInt64) <<< 8 |||
    (data.get! (o + 5)).toUInt64) <<< 8 ||| (data.get! (o + 6)).toUInt64) <<< 8 ||| (data.get! (o + 7)).toUInt64)

/-- Schedule word `t` of words `t - 16`, `t - 15`, `t - 7` and `t - 2`. -/
@[inline] def sched (a b c d : UInt64) : UInt64 :=
  a + (rotr b 1 ^^^ rotr b 8 ^^^ b >>> 7) + c + (rotr d 19 ^^^ rotr d 61 ^^^ d >>> 6)

/-- Round `i` of the registers `s` with schedule word `w`. -/
@[inline] def round (s : Regs) (i : Nat) (w : UInt64) : Regs :=
  let (a, b, c, d, e, f, g, hh) := s
  let t1 := hh + (rotr e 14 ^^^ rotr e 18 ^^^ rotr e 41) + (e &&& f ^^^ ~~~e &&& g) + k[i]! + w
  let t2 := (rotr a 28 ^^^ rotr a 34 ^^^ rotr a 39) + (a &&& b ^^^ a &&& c ^^^ b &&& c)
  (t1 + t2, a, b, c, d + t1, e, f, g)

/-- One compression of the 128-byte block of `data` at `off`, straight-line so that the schedule's words and the registers
stay unboxed (an `Array UInt64` or a loop's state boxes each word). `word`, `sched` and `round` keep the shape of FIPS
180-4's three loops term for term: the soundness proofs' `compress_unfold` reads them back as those loops. -/
def compress (h : Array UInt64) (data : ByteArray) (off : Nat) : Array UInt64 :=
  let w0 := word data (off + 8 * 0)
  let w1 := word data (off + 8 * 1)
  let w2 := word data (off + 8 * 2)
  let w3 := word data (off + 8 * 3)
  let w4 := word data (off + 8 * 4)
  let w5 := word data (off + 8 * 5)
  let w6 := word data (off + 8 * 6)
  let w7 := word data (off + 8 * 7)
  let w8 := word data (off + 8 * 8)
  let w9 := word data (off + 8 * 9)
  let w10 := word data (off + 8 * 10)
  let w11 := word data (off + 8 * 11)
  let w12 := word data (off + 8 * 12)
  let w13 := word data (off + 8 * 13)
  let w14 := word data (off + 8 * 14)
  let w15 := word data (off + 8 * 15)
  let w16 := sched w0 w1 w9 w14
  let w17 := sched w1 w2 w10 w15
  let w18 := sched w2 w3 w11 w16
  let w19 := sched w3 w4 w12 w17
  let w20 := sched w4 w5 w13 w18
  let w21 := sched w5 w6 w14 w19
  let w22 := sched w6 w7 w15 w20
  let w23 := sched w7 w8 w16 w21
  let w24 := sched w8 w9 w17 w22
  let w25 := sched w9 w10 w18 w23
  let w26 := sched w10 w11 w19 w24
  let w27 := sched w11 w12 w20 w25
  let w28 := sched w12 w13 w21 w26
  let w29 := sched w13 w14 w22 w27
  let w30 := sched w14 w15 w23 w28
  let w31 := sched w15 w16 w24 w29
  let w32 := sched w16 w17 w25 w30
  let w33 := sched w17 w18 w26 w31
  let w34 := sched w18 w19 w27 w32
  let w35 := sched w19 w20 w28 w33
  let w36 := sched w20 w21 w29 w34
  let w37 := sched w21 w22 w30 w35
  let w38 := sched w22 w23 w31 w36
  let w39 := sched w23 w24 w32 w37
  let w40 := sched w24 w25 w33 w38
  let w41 := sched w25 w26 w34 w39
  let w42 := sched w26 w27 w35 w40
  let w43 := sched w27 w28 w36 w41
  let w44 := sched w28 w29 w37 w42
  let w45 := sched w29 w30 w38 w43
  let w46 := sched w30 w31 w39 w44
  let w47 := sched w31 w32 w40 w45
  let w48 := sched w32 w33 w41 w46
  let w49 := sched w33 w34 w42 w47
  let w50 := sched w34 w35 w43 w48
  let w51 := sched w35 w36 w44 w49
  let w52 := sched w36 w37 w45 w50
  let w53 := sched w37 w38 w46 w51
  let w54 := sched w38 w39 w47 w52
  let w55 := sched w39 w40 w48 w53
  let w56 := sched w40 w41 w49 w54
  let w57 := sched w41 w42 w50 w55
  let w58 := sched w42 w43 w51 w56
  let w59 := sched w43 w44 w52 w57
  let w60 := sched w44 w45 w53 w58
  let w61 := sched w45 w46 w54 w59
  let w62 := sched w46 w47 w55 w60
  let w63 := sched w47 w48 w56 w61
  let w64 := sched w48 w49 w57 w62
  let w65 := sched w49 w50 w58 w63
  let w66 := sched w50 w51 w59 w64
  let w67 := sched w51 w52 w60 w65
  let w68 := sched w52 w53 w61 w66
  let w69 := sched w53 w54 w62 w67
  let w70 := sched w54 w55 w63 w68
  let w71 := sched w55 w56 w64 w69
  let w72 := sched w56 w57 w65 w70
  let w73 := sched w57 w58 w66 w71
  let w74 := sched w58 w59 w67 w72
  let w75 := sched w59 w60 w68 w73
  let w76 := sched w60 w61 w69 w74
  let w77 := sched w61 w62 w70 w75
  let w78 := sched w62 w63 w71 w76
  let w79 := sched w63 w64 w72 w77
  let s0 : Regs := (h[0]!, h[1]!, h[2]!, h[3]!, h[4]!, h[5]!, h[6]!, h[7]!)
  let s1 := round s0 0 w0
  let s2 := round s1 1 w1
  let s3 := round s2 2 w2
  let s4 := round s3 3 w3
  let s5 := round s4 4 w4
  let s6 := round s5 5 w5
  let s7 := round s6 6 w6
  let s8 := round s7 7 w7
  let s9 := round s8 8 w8
  let s10 := round s9 9 w9
  let s11 := round s10 10 w10
  let s12 := round s11 11 w11
  let s13 := round s12 12 w12
  let s14 := round s13 13 w13
  let s15 := round s14 14 w14
  let s16 := round s15 15 w15
  let s17 := round s16 16 w16
  let s18 := round s17 17 w17
  let s19 := round s18 18 w18
  let s20 := round s19 19 w19
  let s21 := round s20 20 w20
  let s22 := round s21 21 w21
  let s23 := round s22 22 w22
  let s24 := round s23 23 w23
  let s25 := round s24 24 w24
  let s26 := round s25 25 w25
  let s27 := round s26 26 w26
  let s28 := round s27 27 w27
  let s29 := round s28 28 w28
  let s30 := round s29 29 w29
  let s31 := round s30 30 w30
  let s32 := round s31 31 w31
  let s33 := round s32 32 w32
  let s34 := round s33 33 w33
  let s35 := round s34 34 w34
  let s36 := round s35 35 w35
  let s37 := round s36 36 w36
  let s38 := round s37 37 w37
  let s39 := round s38 38 w38
  let s40 := round s39 39 w39
  let s41 := round s40 40 w40
  let s42 := round s41 41 w41
  let s43 := round s42 42 w42
  let s44 := round s43 43 w43
  let s45 := round s44 44 w44
  let s46 := round s45 45 w45
  let s47 := round s46 46 w46
  let s48 := round s47 47 w47
  let s49 := round s48 48 w48
  let s50 := round s49 49 w49
  let s51 := round s50 50 w50
  let s52 := round s51 51 w51
  let s53 := round s52 52 w52
  let s54 := round s53 53 w53
  let s55 := round s54 54 w54
  let s56 := round s55 55 w55
  let s57 := round s56 56 w56
  let s58 := round s57 57 w57
  let s59 := round s58 58 w58
  let s60 := round s59 59 w59
  let s61 := round s60 60 w60
  let s62 := round s61 61 w61
  let s63 := round s62 62 w62
  let s64 := round s63 63 w63
  let s65 := round s64 64 w64
  let s66 := round s65 65 w65
  let s67 := round s66 66 w66
  let s68 := round s67 67 w67
  let s69 := round s68 68 w68
  let s70 := round s69 69 w69
  let s71 := round s70 70 w70
  let s72 := round s71 71 w71
  let s73 := round s72 72 w72
  let s74 := round s73 73 w73
  let s75 := round s74 74 w74
  let s76 := round s75 75 w75
  let s77 := round s76 76 w76
  let s78 := round s77 77 w77
  let s79 := round s78 78 w78
  let s80 := round s79 79 w79
  #[h[0]! + s80.1, h[1]! + s80.2.1, h[2]! + s80.2.2.1, h[3]! + s80.2.2.2.1, h[4]! + s80.2.2.2.2.1,
    h[5]! + s80.2.2.2.2.2.1, h[6]! + s80.2.2.2.2.2.2.1, h[7]! + s80.2.2.2.2.2.2.2]

/-- SHA-512 of `msg` (FIPS 180-4: 128-byte blocks, a 128-bit length). -/
def hash (msg : ByteArray) : Digest := Id.run do
  let len := msg.size
  let mut data := msg.push 0x80
  for _ in [0:(112 + 128 - data.size % 128) % 128] do data := data.push 0
  let bits := len * 8
  for i in [0:16] do
    data := data.push ((bits >>> (8 * (15 - i))) % 256).toUInt8
  let mut h := iv
  for blk in [0:data.size / 128] do
    h := compress h data (128 * blk)
  -- the state's eight words, big-endian: 64 bytes whatever the loops computed (`Sha512.hash_size`)
  return ⟨Array.ofFn (n := 64) fun i => (h[i.val / 8]! >>> (8 * (7 - i.val % 8)).toUInt64).toUInt8⟩

end Sha512

namespace Blake3

def CHUNK_START : UInt32 := 1
def CHUNK_END : UInt32 := 2
def PARENT : UInt32 := 4
def ROOT : UInt32 := 8
def KEYED_HASH : UInt32 := 16

def iv : Array UInt32 := Sha256.iv
def perm : Array Nat := #[2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]

@[inline] def rotr (x : UInt32) (n : UInt32) : UInt32 := (x >>> n) ||| (x <<< (32 - n))

@[inline] def g (s : Array UInt32) (a b c d : Nat) (mx my : UInt32) : Array UInt32 := Id.run do
  let mut s := s
  s := s.set! a (s[a]! + s[b]! + mx)
  s := s.set! d (rotr (s[d]! ^^^ s[a]!) 16)
  s := s.set! c (s[c]! + s[d]!)
  s := s.set! b (rotr (s[b]! ^^^ s[c]!) 12)
  s := s.set! a (s[a]! + s[b]! + my)
  s := s.set! d (rotr (s[d]! ^^^ s[a]!) 8)
  s := s.set! c (s[c]! + s[d]!)
  s := s.set! b (rotr (s[b]! ^^^ s[c]!) 7)
  return s

/-- The BLAKE3 compression function: the 16-word output state. -/
def compress (cv : Array UInt32) (m : Array UInt32) (counter : UInt64) (blockLen : UInt32) (flags : UInt32) : Array UInt32 := Id.run do
  let mut s : Array UInt32 := #[cv[0]!, cv[1]!, cv[2]!, cv[3]!, cv[4]!, cv[5]!, cv[6]!, cv[7]!,
    iv[0]!, iv[1]!, iv[2]!, iv[3]!, counter.toUInt32, (counter >>> 32).toUInt32, blockLen, flags]
  let mut m := m
  for r in [0:7] do
    s := g s 0 4 8 12 m[0]! m[1]!
    s := g s 1 5 9 13 m[2]! m[3]!
    s := g s 2 6 10 14 m[4]! m[5]!
    s := g s 3 7 11 15 m[6]! m[7]!
    s := g s 0 5 10 15 m[8]! m[9]!
    s := g s 1 6 11 12 m[10]! m[11]!
    s := g s 2 7 8 13 m[12]! m[13]!
    s := g s 3 4 9 14 m[14]! m[15]!
    if r < 6 then
      m := perm.map (fun i => m[i]!)
  let mut out := s
  for i in [0:8] do
    out := out.set! i (s[i]! ^^^ s[i + 8]!)
    out := out.set! (i + 8) (s[i + 8]! ^^^ cv[i]!)
  return out

def wordsOfBytes (b : ByteArray) (off : Nat) (n : Nat) : Array UInt32 := Id.run do
  let mut w : Array UInt32 := Array.mkEmpty n
  for i in [0:n] do
    let at_ := off + 4 * i
    let byte := fun j => if at_ + j < b.size then (b.get! (at_ + j)).toUInt32 else 0
    w := w.push (byte 0 ||| (byte 1 <<< 8) ||| (byte 2 <<< 16) ||| (byte 3 <<< 24))
  return w

def bytesOfWords (w : Array UInt32) (n : Nat) : ByteArray := Id.run do
  let mut out := ByteArray.emptyWithCapacity (4 * n)
  for i in [0:n] do
    let x := w[i]!
    out := out.push x.toUInt8 |>.push (x >>> 8).toUInt8 |>.push (x >>> 16).toUInt8 |>.push (x >>> 24).toUInt8
  return out

/-- Streaming hasher state (the reference implementation's chunk state and chaining-value stack). -/
structure Hasher where
  key : Array UInt32
  flags : UInt32
  cv : Array UInt32
  chunkCounter : UInt64
  block : ByteArray
  blocksCompressed : Nat
  stack : Array (Array UInt32)
  deriving Inhabited

def Hasher.new (key : Array UInt32) (flags : UInt32) : Hasher :=
  ⟨key, flags, key, 0, ByteArray.empty, 0, #[]⟩

def Hasher.plain : Hasher := Hasher.new iv 0
def Hasher.keyed (key : ByteArray) : Hasher := Hasher.new (wordsOfBytes key 0 8) KEYED_HASH

private def startFlag (h : Hasher) : UInt32 := if h.blocksCompressed == 0 then CHUNK_START else 0

private def chunkCv (h : Hasher) : Array UInt32 :=
  (compress h.cv (wordsOfBytes h.block 0 16) h.chunkCounter h.block.size.toUInt32 (h.flags ||| startFlag h ||| CHUNK_END)).extract 0 8

private def parentCv (h : Hasher) (l r : Array UInt32) : Array UInt32 :=
  (compress h.key (l ++ r) 0 64 (h.flags ||| PARENT)).extract 0 8

/-- Absorb `input`. -/
def Hasher.update (h0 : Hasher) (input : ByteArray) : Hasher := Id.run do
  let mut h := h0
  let mut pos := 0
  while pos < input.size do
    if h.blocksCompressed * 64 + h.block.size == 1024 then
      -- the chunk is full and more input follows: close it and merge
      let cvc := chunkCv h
      let total := h.chunkCounter + 1
      let mut newCv := cvc
      let mut t := total
      let mut st := h.stack
      while t % 2 == 0 do
        newCv := parentCv h (st.back!) newCv
        st := st.pop
        t := t / 2
      h := { h with stack := st.push newCv, cv := h.key, chunkCounter := total, block := ByteArray.empty, blocksCompressed := 0 }
    if h.block.size == 64 then
      let out := compress h.cv (wordsOfBytes h.block 0 16) h.chunkCounter 64 (h.flags ||| startFlag h)
      h := { h with cv := out.extract 0 8, blocksCompressed := h.blocksCompressed + 1, block := ByteArray.empty }
    let chunkLeft := 1024 - (h.blocksCompressed * 64 + h.block.size)
    let take := min (min (64 - h.block.size) chunkLeft) (input.size - pos)
    h := { h with block := input.copySlice pos h.block h.block.size take }
    pos := pos + take
  return h

/-- The 32-byte root output. -/
def Hasher.finalize (h : Hasher) : Digest := Id.run do
  -- the current chunk's output, then the stack's parents from the top
  let mut inCv := h.cv
  let mut words := wordsOfBytes h.block 0 16
  let mut counter := h.chunkCounter
  let mut blen := h.block.size.toUInt32
  let mut flags := h.flags ||| startFlag h ||| CHUNK_END
  let mut i := h.stack.size
  while i > 0 do
    i := i - 1
    let cvOut := (compress inCv words counter blen flags).extract 0 8
    inCv := h.key
    words := h.stack[i]! ++ cvOut
    counter := 0
    blen := 64
    flags := h.flags ||| PARENT
  return bytesOfWords (compress inCv words 0 blen (flags ||| ROOT)) 8

def hash (b : ByteArray) : Digest := (Hasher.plain.update b).finalize
def keyedHash (key : ByteArray) (b : ByteArray) : Digest := ((Hasher.keyed key).update b).finalize

end Blake3

end Flock
