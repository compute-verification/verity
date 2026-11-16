module

public import Proofs.Properties.Flock.Nets.Seg
public import Proofs.Properties.Flock.Nets.SaltMid

/-!
# `hm96`'s first salt block, chunk by chunk

The compression of the salt's first 16 words from the salt prefix's midstate (`hmTail`'s second): its certificate `hm2`,
each of its fifteen chunks checked in the kernel, and the run they make (`seg_hm2`).
-/

@[expose] public section

open Flock Flock.HmNets FlockSoundness.Discharge.Hm

namespace FlockDeploy.Concrete.Nets

/-- `hm96`'s first salt block's chaining input. -/
def hm2Cv : List Word := saltMidLit.map constW

/-- `hm96`'s first salt block's message. -/
def hm2Msg : List Word := (saltWords (vars 1536 1536)).take 16

/-- Where `hm96`'s first salt block's run puts each committed word. -/
def hm2 : Cert :=
  ⟨84474, [84699, 84988, 85277, 85566, 85855, 86144, 86433, 86722, 87011, 87300, 87589, 87878, 88167, 88456, 88745,
  89034, 89323, 89612, 89901, 90190, 90479, 90768, 91057, 91346, 91635, 91924, 92213, 92502, 92791, 93080, 93369,
  93658, 93947, 94236, 94525, 94814, 95103, 95392, 95681, 95970, 96259, 96548, 96837, 97126, 97415, 97704, 97993,
  98282, 98571, 98860, 99149, 99438, 99727, 100016, 100305, 100594, 100883, 101172, 101461, 101750, 102039, 102328,
  102617, 102906], [103253, 103900, 104676, 105451, 106228, 107008, 107788, 108565, 109344, 110123, 110901, 111680,
  112460, 113240, 114020, 114798, 115577, 116357, 117137, 117917, 118697, 119477, 120255, 121035, 121815, 122591,
  123371, 124149, 124928, 125708, 126488, 127264, 128042, 128821, 129601, 130381, 131160, 131937, 132716, 133496,
  134274, 135054, 135834, 136610, 137387, 138163, 138942, 139719, 140496, 141276, 142056, 142833, 143613, 144393,
  145173, 145953, 146731, 147505, 148284, 149062, 149839, 150619, 151399, 152179, 152957, 153737, 154516, 155293,
  156072, 156851, 157630, 158410, 159188, 159968, 160746, 161524, 162303, 163082, 163860, 164640], [1592, 103761,
  104537, 105312, 106089, 106869, 107649, 108426, 109205, 109984, 110762, 111541, 112321, 113101, 113881, 114659,
  115438, 116218, 116998, 117778, 118558, 119338, 120116, 120896, 121676, 122452, 123232, 124010, 124789, 125569,
  126349, 127125, 127903, 128682, 129462, 130242, 131021, 131798, 132577, 133357, 134135, 134915, 135695, 136471,
  137248, 138024, 138803, 139580, 140357, 141137, 141917, 142694, 143474, 144254, 145034, 145814, 146592, 147366,
  148145, 148923, 149700, 150480, 151260, 152040, 152818, 153598, 154377, 155154, 155933, 156712, 157491, 158271,
  159049, 159829, 160607, 161385, 162164, 162943, 163721, 164501], 165291⟩

-- Each chunk runs the builder in the kernel, past the default heartbeat budget.
set_option maxHeartbeats 2000000

theorem hm2_s0 : schedOk (hm2Msg) hm2 0 = true := by decide +kernel

theorem hm2_s1 : schedOk (hm2Msg) hm2 1 = true := by decide +kernel

theorem hm2_s2 : schedOk (hm2Msg) hm2 2 = true := by decide +kernel

theorem hm2_s3 : schedOk (hm2Msg) hm2 3 = true := by decide +kernel

theorem hm2_r0 : roundOk (hm2Cv) (hm2Msg) hm2 0 = true := by decide +kernel

theorem hm2_r1 : roundOk (hm2Cv) (hm2Msg) hm2 1 = true := by decide +kernel

theorem hm2_r2 : roundOk (hm2Cv) (hm2Msg) hm2 2 = true := by decide +kernel

theorem hm2_r3 : roundOk (hm2Cv) (hm2Msg) hm2 3 = true := by decide +kernel

theorem hm2_r4 : roundOk (hm2Cv) (hm2Msg) hm2 4 = true := by decide +kernel

theorem hm2_r5 : roundOk (hm2Cv) (hm2Msg) hm2 5 = true := by decide +kernel

theorem hm2_r6 : roundOk (hm2Cv) (hm2Msg) hm2 6 = true := by decide +kernel

theorem hm2_r7 : roundOk (hm2Cv) (hm2Msg) hm2 7 = true := by decide +kernel

theorem hm2_r8 : roundOk (hm2Cv) (hm2Msg) hm2 8 = true := by decide +kernel

theorem hm2_r9 : roundOk (hm2Cv) (hm2Msg) hm2 9 = true := by decide +kernel

theorem hm2_f : ffOk (hm2Cv) hm2 = true := by decide +kernel

theorem seg_hm2 : Seg 128 (compress (hm2Cv) (hm2Msg)) 84474 (ffOut (hm2Cv) hm2) 165291 :=
  seg_compress _ _ hm2 (forall_lt4 hm2_s0 hm2_s1 hm2_s2 hm2_s3)
    (forall_lt10 hm2_r0 hm2_r1 hm2_r2 hm2_r3 hm2_r4 hm2_r5 hm2_r6 hm2_r7 hm2_r8 hm2_r9) hm2_f

theorem hm2_out : ∀ w ∈ ffOut (hm2Cv) hm2, ∀ f ∈ w, f.s.length ≤ 16 := ffOut_width hm2_f

end FlockDeploy.Concrete.Nets
