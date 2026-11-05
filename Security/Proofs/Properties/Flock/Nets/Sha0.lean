module

public import Proofs.Properties.Flock.Nets.Seg

/-!
# `sha512x3`'s compression 0, chunk by chunk

Compression 0 of `sha512x3` (`compression3`), from its inputs `cv0` and `msg0`: its certificate `sha0`, each of its
fifteen chunks checked in the kernel, and the run they make (`seg_sha0`).
-/

@[expose] public section

open Flock Flock.HmNets FlockSoundness.Discharge.Hm

namespace FlockDeploy.Concrete.Nets

/-- `sha512x3`'s compression 0's chaining input. -/
def sha0Cv : List Word := words (vars 0 512) 8

/-- `sha512x3`'s compression 0's message. -/
def sha0Msg : List Word := words (vars 512 1024) 16

/-- Where `sha512x3`'s compression 0's run puts each committed word. -/
def sha0 : Cert :=
  ⟨4608, [4833, 5122, 5411, 5700, 5989, 6278, 6567, 6856, 7145, 7434, 7723, 8012, 8301, 8590, 8879, 9168, 9457, 9746,
  10035, 10324, 10613, 10902, 11191, 11480, 11769, 12058, 12347, 12636, 12925, 13214, 13503, 13792, 14081, 14370,
  14659, 14948, 15237, 15526, 15815, 16104, 16393, 16682, 16971, 17260, 17549, 17838, 18127, 18416, 18705, 18994,
  19283, 19572, 19861, 20150, 20439, 20728, 21017, 21306, 21595, 21884, 22173, 22462, 22751, 23040], [23819, 24599,
  25379, 26157, 26934, 27714, 28494, 29271, 30050, 30829, 31607, 32386, 33166, 33946, 34726, 35504, 36283, 37063,
  37843, 38623, 39403, 40183, 40961, 41741, 42521, 43297, 44077, 44855, 45634, 46414, 47194, 47970, 48748, 49527,
  50307, 51087, 51866, 52643, 53422, 54202, 54980, 55760, 56540, 57316, 58093, 58869, 59648, 60425, 61202, 61982,
  62762, 63539, 64319, 65099, 65879, 66659, 67437, 68211, 68990, 69768, 70545, 71325, 72105, 72885, 73663, 74443,
  75222, 75999, 76778, 77557, 78336, 79116, 79894, 80674, 81452, 82230, 83009, 83788, 84566, 85346], [23680, 24460,
  25240, 26018, 26795, 27575, 28355, 29132, 29911, 30690, 31468, 32247, 33027, 33807, 34587, 35365, 36144, 36924,
  37704, 38484, 39264, 40044, 40822, 41602, 42382, 43158, 43938, 44716, 45495, 46275, 47055, 47831, 48609, 49388,
  50168, 50948, 51727, 52504, 53283, 54063, 54841, 55621, 56401, 57177, 57954, 58730, 59509, 60286, 61063, 61843,
  62623, 63400, 64180, 64960, 65740, 66520, 67298, 68072, 68851, 69629, 70406, 71186, 71966, 72746, 73524, 74304,
  75083, 75860, 76639, 77418, 78197, 78977, 79755, 80535, 81313, 82091, 82870, 83649, 84427, 85207], 86010⟩

-- Each chunk runs the builder in the kernel, past the default heartbeat budget.
set_option maxHeartbeats 2000000

theorem sha0_s0 : schedOk (sha0Msg) sha0 0 = true := by decide +kernel

theorem sha0_s1 : schedOk (sha0Msg) sha0 1 = true := by decide +kernel

theorem sha0_s2 : schedOk (sha0Msg) sha0 2 = true := by decide +kernel

theorem sha0_s3 : schedOk (sha0Msg) sha0 3 = true := by decide +kernel

theorem sha0_r0 : roundOk (sha0Cv) (sha0Msg) sha0 0 = true := by decide +kernel

theorem sha0_r1 : roundOk (sha0Cv) (sha0Msg) sha0 1 = true := by decide +kernel

theorem sha0_r2 : roundOk (sha0Cv) (sha0Msg) sha0 2 = true := by decide +kernel

theorem sha0_r3 : roundOk (sha0Cv) (sha0Msg) sha0 3 = true := by decide +kernel

theorem sha0_r4 : roundOk (sha0Cv) (sha0Msg) sha0 4 = true := by decide +kernel

theorem sha0_r5 : roundOk (sha0Cv) (sha0Msg) sha0 5 = true := by decide +kernel

theorem sha0_r6 : roundOk (sha0Cv) (sha0Msg) sha0 6 = true := by decide +kernel

theorem sha0_r7 : roundOk (sha0Cv) (sha0Msg) sha0 7 = true := by decide +kernel

theorem sha0_r8 : roundOk (sha0Cv) (sha0Msg) sha0 8 = true := by decide +kernel

theorem sha0_r9 : roundOk (sha0Cv) (sha0Msg) sha0 9 = true := by decide +kernel

theorem sha0_f : ffOk (sha0Cv) sha0 = true := by decide +kernel

theorem seg_sha0 : Seg 128 (compress (sha0Cv) (sha0Msg)) 4608 (ffOut (sha0Cv) sha0) 86010 :=
  seg_compress _ _ sha0 (forall_lt4 sha0_s0 sha0_s1 sha0_s2 sha0_s3)
    (forall_lt10 sha0_r0 sha0_r1 sha0_r2 sha0_r3 sha0_r4 sha0_r5 sha0_r6 sha0_r7 sha0_r8 sha0_r9) sha0_f

theorem sha0_out : ∀ w ∈ ffOut (sha0Cv) sha0, ∀ f ∈ w, f.s.length ≤ 16 := ffOut_width sha0_f

end FlockDeploy.Concrete.Nets
