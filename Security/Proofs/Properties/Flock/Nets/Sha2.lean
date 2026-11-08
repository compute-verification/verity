module

public import Proofs.Properties.Flock.Nets.Seg

/-!
# `sha512x3`'s compression 2, chunk by chunk

Compression 2 of `sha512x3` (`compression3`), from its inputs `cv2` and `msg2`: its certificate `sha2`, each of its
fifteen chunks checked in the kernel, and the run they make (`seg_sha2`).
-/

@[expose] public section

open Flock Flock.HmNets FlockSoundness.Discharge.Hm

namespace FlockDeploy.Concrete.Nets

/-- `sha512x3`'s compression 2's chaining input. -/
def sha2Cv : List Word := words (vars 3072 512) 8

/-- `sha512x3`'s compression 2's message. -/
def sha2Msg : List Word := words (vars 3584 1024) 16

/-- Where `sha512x3`'s compression 2's run puts each committed word. -/
def sha2 : Cert :=
  ⟨167412, [167637, 167926, 168215, 168504, 168793, 169082, 169371, 169660, 169949, 170238, 170527, 170816, 171105,
  171394, 171683, 171972, 172261, 172550, 172839, 173128, 173417, 173706, 173995, 174284, 174573, 174862, 175151,
  175440, 175729, 176018, 176307, 176596, 176885, 177174, 177463, 177752, 178041, 178330, 178619, 178908, 179197,
  179486, 179775, 180064, 180353, 180642, 180931, 181220, 181509, 181798, 182087, 182376, 182665, 182954, 183243,
  183532, 183821, 184110, 184399, 184688, 184977, 185266, 185555, 185844], [186623, 187403, 188183, 188961, 189738,
  190518, 191298, 192075, 192854, 193633, 194411, 195190, 195970, 196750, 197530, 198308, 199087, 199867, 200647,
  201427, 202207, 202987, 203765, 204545, 205325, 206101, 206881, 207659, 208438, 209218, 209998, 210774, 211552,
  212331, 213111, 213891, 214670, 215447, 216226, 217006, 217784, 218564, 219344, 220120, 220897, 221673, 222452,
  223229, 224006, 224786, 225566, 226343, 227123, 227903, 228683, 229463, 230241, 231015, 231794, 232572, 233349,
  234129, 234909, 235689, 236467, 237247, 238026, 238803, 239582, 240361, 241140, 241920, 242698, 243478, 244256,
  245034, 245813, 246592, 247370, 248150], [186484, 187264, 188044, 188822, 189599, 190379, 191159, 191936, 192715,
  193494, 194272, 195051, 195831, 196611, 197391, 198169, 198948, 199728, 200508, 201288, 202068, 202848, 203626,
  204406, 205186, 205962, 206742, 207520, 208299, 209079, 209859, 210635, 211413, 212192, 212972, 213752, 214531,
  215308, 216087, 216867, 217645, 218425, 219205, 219981, 220758, 221534, 222313, 223090, 223867, 224647, 225427,
  226204, 226984, 227764, 228544, 229324, 230102, 230876, 231655, 232433, 233210, 233990, 234770, 235550, 236328,
  237108, 237887, 238664, 239443, 240222, 241001, 241781, 242559, 243339, 244117, 244895, 245674, 246453, 247231,
  248011], 248814⟩

-- Each chunk runs the builder in the kernel, past the default heartbeat budget.
set_option maxHeartbeats 2000000

theorem sha2_s0 : schedOk (sha2Msg) sha2 0 = true := by decide +kernel

theorem sha2_s1 : schedOk (sha2Msg) sha2 1 = true := by decide +kernel

theorem sha2_s2 : schedOk (sha2Msg) sha2 2 = true := by decide +kernel

theorem sha2_s3 : schedOk (sha2Msg) sha2 3 = true := by decide +kernel

theorem sha2_r0 : roundOk (sha2Cv) (sha2Msg) sha2 0 = true := by decide +kernel

theorem sha2_r1 : roundOk (sha2Cv) (sha2Msg) sha2 1 = true := by decide +kernel

theorem sha2_r2 : roundOk (sha2Cv) (sha2Msg) sha2 2 = true := by decide +kernel

theorem sha2_r3 : roundOk (sha2Cv) (sha2Msg) sha2 3 = true := by decide +kernel

theorem sha2_r4 : roundOk (sha2Cv) (sha2Msg) sha2 4 = true := by decide +kernel

theorem sha2_r5 : roundOk (sha2Cv) (sha2Msg) sha2 5 = true := by decide +kernel

theorem sha2_r6 : roundOk (sha2Cv) (sha2Msg) sha2 6 = true := by decide +kernel

theorem sha2_r7 : roundOk (sha2Cv) (sha2Msg) sha2 7 = true := by decide +kernel

theorem sha2_r8 : roundOk (sha2Cv) (sha2Msg) sha2 8 = true := by decide +kernel

theorem sha2_r9 : roundOk (sha2Cv) (sha2Msg) sha2 9 = true := by decide +kernel

theorem sha2_f : ffOk (sha2Cv) sha2 = true := by decide +kernel

theorem seg_sha2 : Seg 128 (compress (sha2Cv) (sha2Msg)) 167412 (ffOut (sha2Cv) sha2) 248814 :=
  seg_compress _ _ sha2 (forall_lt4 sha2_s0 sha2_s1 sha2_s2 sha2_s3)
    (forall_lt10 sha2_r0 sha2_r1 sha2_r2 sha2_r3 sha2_r4 sha2_r5 sha2_r6 sha2_r7 sha2_r8 sha2_r9) sha2_f

theorem sha2_out : ∀ w ∈ ffOut (sha2Cv) sha2, ∀ f ∈ w, f.s.length ≤ 16 := ffOut_width sha2_f

end FlockDeploy.Concrete.Nets
