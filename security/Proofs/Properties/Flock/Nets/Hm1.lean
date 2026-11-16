module

public import Proofs.Properties.Flock.Nets.Seg

/-!
# `hm96`'s compression `F(cv, pad)`, chunk by chunk

The compression of `hm96`'s inputs `cv` and `pad` (`hmTail`'s first): its certificate `hm1`, each of its fifteen chunks
checked in the kernel, and the run they make (`seg_hm1`).
-/

@[expose] public section

open Flock Flock.HmNets FlockSoundness.Discharge.Hm

namespace FlockDeploy.Concrete.Nets

/-- `hm96`'s compression `F(cv, pad)`'s chaining input. -/
def hm1Cv : List Word := words (vars 0 512) 8

/-- `hm96`'s compression `F(cv, pad)`'s message. -/
def hm1Msg : List Word := words (vars 512 1024) 16

/-- Where `hm96`'s compression `F(cv, pad)`'s run puts each committed word. -/
def hm1 : Cert :=
  ⟨3072, [3297, 3586, 3875, 4164, 4453, 4742, 5031, 5320, 5609, 5898, 6187, 6476, 6765, 7054, 7343, 7632, 7921, 8210,
  8499, 8788, 9077, 9366, 9655, 9944, 10233, 10522, 10811, 11100, 11389, 11678, 11967, 12256, 12545, 12834, 13123,
  13412, 13701, 13990, 14279, 14568, 14857, 15146, 15435, 15724, 16013, 16302, 16591, 16880, 17169, 17458, 17747,
  18036, 18325, 18614, 18903, 19192, 19481, 19770, 20059, 20348, 20637, 20926, 21215, 21504], [22283, 23063, 23843,
  24621, 25398, 26178, 26958, 27735, 28514, 29293, 30071, 30850, 31630, 32410, 33190, 33968, 34747, 35527, 36307,
  37087, 37867, 38647, 39425, 40205, 40985, 41761, 42541, 43319, 44098, 44878, 45658, 46434, 47212, 47991, 48771,
  49551, 50330, 51107, 51886, 52666, 53444, 54224, 55004, 55780, 56557, 57333, 58112, 58889, 59666, 60446, 61226,
  62003, 62783, 63563, 64343, 65123, 65901, 66675, 67454, 68232, 69009, 69789, 70569, 71349, 72127, 72907, 73686,
  74463, 75242, 76021, 76800, 77580, 78358, 79138, 79916, 80694, 81473, 82252, 83030, 83810], [22144, 22924, 23704,
  24482, 25259, 26039, 26819, 27596, 28375, 29154, 29932, 30711, 31491, 32271, 33051, 33829, 34608, 35388, 36168,
  36948, 37728, 38508, 39286, 40066, 40846, 41622, 42402, 43180, 43959, 44739, 45519, 46295, 47073, 47852, 48632,
  49412, 50191, 50968, 51747, 52527, 53305, 54085, 54865, 55641, 56418, 57194, 57973, 58750, 59527, 60307, 61087,
  61864, 62644, 63424, 64204, 64984, 65762, 66536, 67315, 68093, 68870, 69650, 70430, 71210, 71988, 72768, 73547,
  74324, 75103, 75882, 76661, 77441, 78219, 78999, 79777, 80555, 81334, 82113, 82891, 83671], 84474⟩

-- Each chunk runs the builder in the kernel, past the default heartbeat budget.
set_option maxHeartbeats 2000000

theorem hm1_s0 : schedOk (hm1Msg) hm1 0 = true := by decide +kernel

theorem hm1_s1 : schedOk (hm1Msg) hm1 1 = true := by decide +kernel

theorem hm1_s2 : schedOk (hm1Msg) hm1 2 = true := by decide +kernel

theorem hm1_s3 : schedOk (hm1Msg) hm1 3 = true := by decide +kernel

theorem hm1_r0 : roundOk (hm1Cv) (hm1Msg) hm1 0 = true := by decide +kernel

theorem hm1_r1 : roundOk (hm1Cv) (hm1Msg) hm1 1 = true := by decide +kernel

theorem hm1_r2 : roundOk (hm1Cv) (hm1Msg) hm1 2 = true := by decide +kernel

theorem hm1_r3 : roundOk (hm1Cv) (hm1Msg) hm1 3 = true := by decide +kernel

theorem hm1_r4 : roundOk (hm1Cv) (hm1Msg) hm1 4 = true := by decide +kernel

theorem hm1_r5 : roundOk (hm1Cv) (hm1Msg) hm1 5 = true := by decide +kernel

theorem hm1_r6 : roundOk (hm1Cv) (hm1Msg) hm1 6 = true := by decide +kernel

theorem hm1_r7 : roundOk (hm1Cv) (hm1Msg) hm1 7 = true := by decide +kernel

theorem hm1_r8 : roundOk (hm1Cv) (hm1Msg) hm1 8 = true := by decide +kernel

theorem hm1_r9 : roundOk (hm1Cv) (hm1Msg) hm1 9 = true := by decide +kernel

theorem hm1_f : ffOk (hm1Cv) hm1 = true := by decide +kernel

theorem seg_hm1 : Seg 128 (compress (hm1Cv) (hm1Msg)) 3072 (ffOut (hm1Cv) hm1) 84474 :=
  seg_compress _ _ hm1 (forall_lt4 hm1_s0 hm1_s1 hm1_s2 hm1_s3)
    (forall_lt10 hm1_r0 hm1_r1 hm1_r2 hm1_r3 hm1_r4 hm1_r5 hm1_r6 hm1_r7 hm1_r8 hm1_r9) hm1_f

theorem hm1_out : ∀ w ∈ ffOut (hm1Cv) hm1, ∀ f ∈ w, f.s.length ≤ 16 := ffOut_width hm1_f

end FlockDeploy.Concrete.Nets
