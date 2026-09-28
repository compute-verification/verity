import Difftest
import Identity

/-!
# The identity difftest: `Identity`'s functions behind `Difftest.serve`

`identity-difftest` serves `table` to the core difftest harness (`verity/core/difftest`), whose Python half
(`tests/test_identity_difftest.py`) holds each function to `verity.ml.commitments.identity` and `hashlib`:

* `sha512`: `Sha512.State` absorbing the payload in the given pieces, then finalized (`hashlib.sha512`);
* `tagged`: `taggedSha512` (`tagged_sha512`), the empty tag refused;
* `identity`: `identityDigestSha512` (`identity_digest_sha512`) of a JSON manifest.
-/

namespace Identity.Difftest

open Lean (Json)

def str (j : Json) (k : String) : Flock.V String := Flock.J.str j k "input"
def bytes (j : Json) (k : String) : Flock.V ByteArray := do Flock.unhex (← str j k)
def hex (b : ByteArray) : Json := .str (Flock.hexOf b)

def sha512 (i : Json) : Flock.V Json := do
  let payload ← bytes i "payload"
  let pieces ← (← Flock.J.arr i "pieces" "input").mapM (Flock.J.asNat · "piece")
  let (st, off) := pieces.foldl (init := (Sha512.init, 0)) fun (st, off) n =>
    (st.update (payload.extract off (off + n)), off + n)
  if off != payload.size then throw "the pieces are not the payload"
  return Json.mkObj [("sha512", hex st.finalize)]

def tagged (i : Json) : Flock.V Json := do
  return Json.mkObj [("digest", hex (← taggedSha512 (← str i "tag") (← bytes i "payload")))]

def identity (i : Json) : Flock.V Json := do
  let manifest ← match i.getObjVal? "manifest" with
    | .ok m => pure m
    | .error _ => throw "manifest missing"
  return Json.mkObj [("digest", hex (← identityDigestSha512 (← str i "tag") manifest))]

def table := _root_.Difftest.table [("sha512", sha512), ("tagged", tagged), ("identity", identity)]

end Identity.Difftest
