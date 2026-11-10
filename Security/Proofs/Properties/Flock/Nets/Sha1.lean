module

public import Proofs.Properties.Flock.Nets.Seg

/-!
# `sha512x3`'s compression 1, chunk by chunk

Compression 1 of `sha512x3` (`compression3`), from its inputs `cv1` and `msg1`: its certificate `sha1`, each of its
fifteen chunks checked in the kernel, and the run they make (`seg_sha1`).
-/

@[expose] public section

open Flock Flock.HmNets FlockSoundness.Discharge.Hm

namespace FlockDeploy.Concrete.Nets

/-- `sha512x3`'s compression 1's chaining input. -/
def sha1Cv : List Word := words (vars 1536 512) 8

/-- `sha512x3`'s compression 1's message. -/
def sha1Msg : List Word := words (vars 2048 1024) 16

/-- Where `sha512x3`'s compression 1's run puts each committed word. -/
def sha1 : Cert :=
  ⟨86010, [86235, 86524, 86813, 87102, 87391, 87680, 87969, 88258, 88547, 88836, 89125, 89414, 89703, 89992, 90281,
  90570, 90859, 91148, 91437, 91726, 92015, 92304, 92593, 92882, 93171, 93460, 93749, 94038, 94327, 94616, 94905,
  95194, 95483, 95772, 96061, 96350, 96639, 96928, 97217, 97506, 97795, 98084, 98373, 98662, 98951, 99240, 99529,
  99818, 100107, 100396, 100685, 100974, 101263, 101552, 101841, 102130, 102419, 102708, 102997, 103286, 103575,
  103864, 104153, 104442], [105221, 106001, 106781, 107559, 108336, 109116, 109896, 110673, 111452, 112231, 113009,
  113788, 114568, 115348, 116128, 116906, 117685, 118465, 119245, 120025, 120805, 121585, 122363, 123143, 123923,
  124699, 125479, 126257, 127036, 127816, 128596, 129372, 130150, 130929, 131709, 132489, 133268, 134045, 134824,
  135604, 136382, 137162, 137942, 138718, 139495, 140271, 141050, 141827, 142604, 143384, 144164, 144941, 145721,
  146501, 147281, 148061, 148839, 149613, 150392, 151170, 151947, 152727, 153507, 154287, 155065, 155845, 156624,
  157401, 158180, 158959, 159738, 160518, 161296, 162076, 162854, 163632, 164411, 165190, 165968, 166748], [105082,
  105862, 106642, 107420, 108197, 108977, 109757, 110534, 111313, 112092, 112870, 113649, 114429, 115209, 115989,
  116767, 117546, 118326, 119106, 119886, 120666, 121446, 122224, 123004, 123784, 124560, 125340, 126118, 126897,
  127677, 128457, 129233, 130011, 130790, 131570, 132350, 133129, 133906, 134685, 135465, 136243, 137023, 137803,
  138579, 139356, 140132, 140911, 141688, 142465, 143245, 144025, 144802, 145582, 146362, 147142, 147922, 148700,
  149474, 150253, 151031, 151808, 152588, 153368, 154148, 154926, 155706, 156485, 157262, 158041, 158820, 159599,
  160379, 161157, 161937, 162715, 163493, 164272, 165051, 165829, 166609], 167412⟩

-- Each chunk runs the builder in the kernel, past the default heartbeat budget.
set_option maxHeartbeats 2000000

theorem sha1_s0 : schedOk (sha1Msg) sha1 0 = true := by decide +kernel

theorem sha1_s1 : schedOk (sha1Msg) sha1 1 = true := by decide +kernel

theorem sha1_s2 : schedOk (sha1Msg) sha1 2 = true := by decide +kernel

theorem sha1_s3 : schedOk (sha1Msg) sha1 3 = true := by decide +kernel

theorem sha1_r0 : roundOk (sha1Cv) (sha1Msg) sha1 0 = true := by decide +kernel

theorem sha1_r1 : roundOk (sha1Cv) (sha1Msg) sha1 1 = true := by decide +kernel

theorem sha1_r2 : roundOk (sha1Cv) (sha1Msg) sha1 2 = true := by decide +kernel

theorem sha1_r3 : roundOk (sha1Cv) (sha1Msg) sha1 3 = true := by decide +kernel

theorem sha1_r4 : roundOk (sha1Cv) (sha1Msg) sha1 4 = true := by decide +kernel

theorem sha1_r5 : roundOk (sha1Cv) (sha1Msg) sha1 5 = true := by decide +kernel

theorem sha1_r6 : roundOk (sha1Cv) (sha1Msg) sha1 6 = true := by decide +kernel

theorem sha1_r7 : roundOk (sha1Cv) (sha1Msg) sha1 7 = true := by decide +kernel

theorem sha1_r8 : roundOk (sha1Cv) (sha1Msg) sha1 8 = true := by decide +kernel

theorem sha1_r9 : roundOk (sha1Cv) (sha1Msg) sha1 9 = true := by decide +kernel

theorem sha1_f : ffOk (sha1Cv) sha1 = true := by decide +kernel

theorem seg_sha1 : Seg 128 (compress (sha1Cv) (sha1Msg)) 86010 (ffOut (sha1Cv) sha1) 167412 :=
  seg_compress _ _ sha1 (forall_lt4 sha1_s0 sha1_s1 sha1_s2 sha1_s3)
    (forall_lt10 sha1_r0 sha1_r1 sha1_r2 sha1_r3 sha1_r4 sha1_r5 sha1_r6 sha1_r7 sha1_r8 sha1_r9) sha1_f

theorem sha1_out : ∀ w ∈ ffOut (sha1Cv) sha1, ∀ f ∈ w, f.s.length ≤ 16 := ffOut_width sha1_f

end FlockDeploy.Concrete.Nets
