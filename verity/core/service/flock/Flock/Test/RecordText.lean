/-
S0 (PROTOCOL.md §5.1): `RecordText.printT` prints what `Json.compress` prints, on the writers' shared vectors
(`verity/ml/flock/live/src/record_text_vectors.json`: escapes, key order, literals, integers, floats and a verdict), and
`RecordText.check` accepts exactly the canonical text: reordered keys, whitespace, a number not in `JsonNumber.toString`'s
form, a duplicate key and a text that isn't JSON are each refused.
-/
import Flock.RecordText

namespace Flock.Test.RecordText

open Flock Lean

/-- `printT` and `Json.compress` agree on what `s` parses to. -/
def same (s : String) : Bool :=
  match Json.parse s with
  | .ok j => RecordText.printT j == j.compress
  | .error _ => false

/-- `printT` of what `s` parses to. -/
def printed (s : String) : Option String := (Json.parse s).toOption.map RecordText.printT

def accepts (s : String) : Bool := (RecordText.check s).toBool

/-- S0 refuses `s` with its own reason (a text that isn't JSON is refused with the parser's). -/
def refused (s : String) : Bool :=
  match RecordText.check s with
  | .error e => e == RecordText.refusal
  | .ok () => false

-- escapes
#guard same "{\"s\": \"a\\\"b\\\\c\\nd\\re\\tf\\u0001g\\u001fh/i\x7fjék l😀m\", \"e\": \"\", \"kéy\": 1}"
#guard printed "{\"s\": \"a\\\"b\\\\c\\nd\\re\\tf\\u0001g\\u001fh/i\x7fjék l😀m\", \"e\": \"\", \"kéy\": 1}" == some "{\"e\":\"\",\"kéy\":1,\"s\":\"a\\\"b\\\\c\\nd\\re\\u0009f\\u0001g\\u001fh/i\x7fjék l😀m\"}"
#guard accepts "{\"e\":\"\",\"kéy\":1,\"s\":\"a\\\"b\\\\c\\nd\\re\\u0009f\\u0001g\\u001fh/i\x7fjék l😀m\"}"
-- order
#guard same "{\"b\": 1, \"a\": 2, \"B\": 3, \"aa\": 4, \"a_b\": 5, \"é\": 6, \"z\": {\"y\": 1, \"x\": [3, {\"q\": null, \"p\": true}]}}"
#guard printed "{\"b\": 1, \"a\": 2, \"B\": 3, \"aa\": 4, \"a_b\": 5, \"é\": 6, \"z\": {\"y\": 1, \"x\": [3, {\"q\": null, \"p\": true}]}}" == some "{\"B\":3,\"a\":2,\"a_b\":5,\"aa\":4,\"b\":1,\"z\":{\"x\":[3,{\"p\":true,\"q\":null}],\"y\":1},\"é\":6}"
#guard accepts "{\"B\":3,\"a\":2,\"a_b\":5,\"aa\":4,\"b\":1,\"z\":{\"x\":[3,{\"p\":true,\"q\":null}],\"y\":1},\"é\":6}"
-- literals
#guard same "{\"n\": null, \"t\": true, \"f\": false, \"arr\": [], \"obj\": {}, \"nested\": [[], [[]], {}]}"
#guard printed "{\"n\": null, \"t\": true, \"f\": false, \"arr\": [], \"obj\": {}, \"nested\": [[], [[]], {}]}" == some "{\"arr\":[],\"f\":false,\"n\":null,\"nested\":[[],[[]],{}],\"obj\":{},\"t\":true}"
#guard accepts "{\"arr\":[],\"f\":false,\"n\":null,\"nested\":[[],[[]],{}],\"obj\":{},\"t\":true}"
-- integers
#guard same "{\"z\": 0, \"neg\": -3, \"big\": 12345678901234567890, \"u64\": 18446744073709551615, \"i64\": -9223372036854775808}"
#guard printed "{\"z\": 0, \"neg\": -3, \"big\": 12345678901234567890, \"u64\": 18446744073709551615, \"i64\": -9223372036854775808}" == some "{\"big\":12345678901234567890,\"i64\":-9223372036854775808,\"neg\":-3,\"u64\":18446744073709551615,\"z\":0}"
#guard accepts "{\"big\":12345678901234567890,\"i64\":-9223372036854775808,\"neg\":-3,\"u64\":18446744073709551615,\"z\":0}"
-- verdict
#guard same "{\"soundness\": {\"per_proof_log2\": -132.02, \"terms\": {\"coin_commitment_hiding\": -191.0, \"irs_field\": -169.441}}, \"accepted\": true, \"secs\": 0.53}"
#guard printed "{\"soundness\": {\"per_proof_log2\": -132.02, \"terms\": {\"coin_commitment_hiding\": -191.0, \"irs_field\": -169.441}}, \"accepted\": true, \"secs\": 0.53}" == some "{\"accepted\":true,\"secs\":0.53,\"soundness\":{\"per_proof_log2\":-132.02,\"terms\":{\"coin_commitment_hiding\":-191,\"irs_field\":-169.441}}}"
#guard accepts "{\"accepted\":true,\"secs\":0.53,\"soundness\":{\"per_proof_log2\":-132.02,\"terms\":{\"coin_commitment_hiding\":-191,\"irs_field\":-169.441}}}"
-- floats
#guard same "{\"f\": [0.0, -0.0, 1.0, -191.0, 0.53, 1e-05, 1e-12, 1.5e+20, 123456.789, -132.02, -169.441, 2.5e-07, 1e+300, 5e-324, 1.7976931348623157e+308, 0.1, 1e+21, 1e+22, 3.14159, 100.0, 1e-09, 1e-10, 1.25e-10]}"

-- refusals
#guard refused "{\"a\": 1}"
#guard refused "{\"b\":1,\"a\":2}"
#guard refused "{\"a\":1.0}"
#guard refused "{\"a\":1e-5}"
#guard refused "{\"a\":1,\"a\":1}"
#guard refused "{\"a\":\"\\t\"}"
#guard refused "{\"a\":1} "
#guard !accepts "{\"a\":" && !refused "{\"a\":"
#guard accepts "{\"a\":\"\\u0009\",\"b\":0.00001}"

end Flock.Test.RecordText
