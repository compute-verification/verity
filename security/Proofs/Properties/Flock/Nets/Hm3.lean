module

public import Proofs.Properties.Flock.Nets.Seg
public import Proofs.Properties.Flock.Nets.Hm2

/-!
# `hm96`'s second salt block, chunk by chunk

The compression of the salt's last 8 words and the padding from the first block's output (`hmTail`'s third): its
certificate `hm3`, each of its fifteen chunks checked in the kernel, and the run they make (`seg_hm3`), whose chaining
input is the first block's output as the kernel runs it (`ffOutK`).
-/

@[expose] public section

open Flock Flock.HmNets FlockSoundness.Discharge.Hm

namespace FlockDeploy.Concrete.Nets

/-- `hm96`'s second salt block's chaining input. -/
noncomputable def hm3Cv : List Word := ffOutK hm2Cv hm2

/-- `hm96`'s second salt block's message. -/
def hm3Msg : List Word := (saltWords (vars 1536 1536)).drop 16 ++ saltTail.map constW

/-- Where `hm96`'s second salt block's run puts each committed word. -/
def hm3 : Cert :=
  ⟨165291, [165366, 165576, 165790, 166004, 166218, 166432, 166710, 166932, 167071, 167210, 167349, 167488, 167627,
  167766, 167978, 168256, 168545, 168834, 169123, 169412, 169701, 169990, 170279, 170568, 170857, 171146, 171435,
  171724, 172013, 172302, 172591, 172880, 173169, 173458, 173747, 174036, 174325, 174614, 174903, 175192, 175481,
  175770, 176059, 176348, 176637, 176926, 177215, 177504, 177793, 178082, 178371, 178660, 178949, 179238, 179527,
  179816, 180105, 180394, 180683, 180972, 181261, 181550, 181839, 182128], [182907, 183687, 184467, 185245, 186022,
  186802, 187582, 188359, 189063, 189767, 190470, 191174, 191879, 192584, 193289, 194056, 194835, 195615, 196395,
  197175, 197955, 198735, 199513, 200293, 201073, 201849, 202629, 203407, 204186, 204966, 205746, 206522, 207300,
  208079, 208859, 209639, 210418, 211195, 211974, 212754, 213532, 214312, 215092, 215868, 216645, 217421, 218200,
  218977, 219754, 220534, 221314, 222091, 222871, 223651, 224431, 225211, 225989, 226763, 227542, 228320, 229097,
  229877, 230657, 231437, 232215, 232995, 233774, 234551, 235330, 236109, 236888, 237668, 238446, 239226, 240004,
  240782, 241561, 242340, 243118, 243898], [182768, 183548, 184328, 185106, 185883, 186663, 187443, 188220, 188924,
  189628, 190331, 191035, 191740, 192445, 193150, 193917, 194696, 195476, 196256, 197036, 197816, 198596, 199374,
  200154, 200934, 201710, 202490, 203268, 204047, 204827, 205607, 206383, 207161, 207940, 208720, 209500, 210279,
  211056, 211835, 212615, 213393, 214173, 214953, 215729, 216506, 217282, 218061, 218838, 219615, 220395, 221175,
  221952, 222732, 223512, 224292, 225072, 225850, 226624, 227403, 228181, 228958, 229738, 230518, 231298, 232076,
  232856, 233635, 234412, 235191, 235970, 236749, 237529, 238307, 239087, 239865, 240643, 241422, 242201, 242979,
  243759], 244562⟩

-- Each chunk runs the builder in the kernel, past the default heartbeat budget.
set_option maxHeartbeats 2000000

theorem hm3_s0 : schedOk (hm3Msg) hm3 0 = true := by decide +kernel

theorem hm3_s1 : schedOk (hm3Msg) hm3 1 = true := by decide +kernel

theorem hm3_s2 : schedOk (hm3Msg) hm3 2 = true := by decide +kernel

theorem hm3_s3 : schedOk (hm3Msg) hm3 3 = true := by decide +kernel

theorem hm3_r0 : roundOk (hm3Cv) (hm3Msg) hm3 0 = true := by decide +kernel

theorem hm3_r1 : roundOk (hm3Cv) (hm3Msg) hm3 1 = true := by decide +kernel

theorem hm3_r2 : roundOk (hm3Cv) (hm3Msg) hm3 2 = true := by decide +kernel

theorem hm3_r3 : roundOk (hm3Cv) (hm3Msg) hm3 3 = true := by decide +kernel

theorem hm3_r4 : roundOk (hm3Cv) (hm3Msg) hm3 4 = true := by decide +kernel

theorem hm3_r5 : roundOk (hm3Cv) (hm3Msg) hm3 5 = true := by decide +kernel

theorem hm3_r6 : roundOk (hm3Cv) (hm3Msg) hm3 6 = true := by decide +kernel

theorem hm3_r7 : roundOk (hm3Cv) (hm3Msg) hm3 7 = true := by decide +kernel

theorem hm3_r8 : roundOk (hm3Cv) (hm3Msg) hm3 8 = true := by decide +kernel

theorem hm3_r9 : roundOk (hm3Cv) (hm3Msg) hm3 9 = true := by decide +kernel

theorem hm3_f : ffOk (hm3Cv) hm3 = true := by decide +kernel

theorem seg_hm3 : Seg 128 (compress (hm3Cv) (hm3Msg)) 165291 (ffOut (hm3Cv) hm3) 244562 :=
  seg_compress _ _ hm3 (forall_lt4 hm3_s0 hm3_s1 hm3_s2 hm3_s3)
    (forall_lt10 hm3_r0 hm3_r1 hm3_r2 hm3_r3 hm3_r4 hm3_r5 hm3_r6 hm3_r7 hm3_r8 hm3_r9) hm3_f

theorem hm3_out : ∀ w ∈ ffOut (hm3Cv) hm3, ∀ f ∈ w, f.s.length ≤ 16 := ffOut_width hm3_f

end FlockDeploy.Concrete.Nets
