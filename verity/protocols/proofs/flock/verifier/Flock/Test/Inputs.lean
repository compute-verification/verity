module

public import Circuits.Descriptor
meta import Base.Hash
meta import Base.JsonParse
meta import Circuits.Descriptor

/-!
`Program.decode` on §5.1's `inputs` and §5.4's supplied inputs' marks: `NotAnd_v1`'s accept and its digests, each refusal
of its `inputs`, a registered and a supplied input beside its run input, and each refusal of a supplied input's mark. Each
verdict is `codec.decode_program`'s on the same bytes.
-/

namespace Flock.Program.InputsTest

/-- `NotAnd_v1(x: Value<1>) = Not(x)` as `codec.encode_program` writes it, `codec.decode_program`'s accept; its digests are
`codec.definition_digests`' and `codec.program_digest`'s on these bytes. -/
private def NOT_DESC : String :=
  r#"{"boolean":{},"callees":["Input1_v1","Not_v1"],"definitions":{"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[NotAnd_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"axes":[],"fn":0,"form":"batch","n":1,"out":0},{"args":[{"refs":{"affine":["n",0,0,1]},"type":1}],"fn":1,"form":"primitive","out":1}],"params":[],"ret":{"affine":["n",1,0,1]},"ret_type":["Value",1]},"form":"composite","id":"Root[NotAnd_v1]_v1"}},"inputs":[["x",1]],"program_fn":"NotAnd_v1","root":"Root[NotAnd_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Array",1,["Value",1]],["Value",1]]}"#

/-- `NOT_DESC` with a registered input node after the run input node, the parameters as `inputs` names them. -/
private def mixedDesc (inputs : String) : String :=
  ((((NOT_DESC.replace r#"["Input1_v1","Not_v1"]"# r#"["Input1_v1","Not_v1","RegisteredInput1_v1"]"#).replace
    r#""Root[NotAnd_v1]_v1":{"# r#""RegisteredInput1_v1":{"form":"primitive","id":"RegisteredInput1_v1","params":[],"ret":["Value",1]},"Root[NotAnd_v1]_v1":{"#).replace
    r#""form":"batch","n":1,"out":0},"# r#""form":"batch","n":1,"out":0},{"args":[],"axes":[],"fn":2,"form":"batch","n":1,"out":0},"#).replace
    r#""ret":{"affine":["n",1,0,1]}"# r#""ret":{"affine":["n",2,0,1]}"#).replace r#""inputs":[["x",1]]"# inputs

/-- `decode`'s verdict on `s`: the program digest of an accept, else the refusal's code. -/
private def verdict (s : String) : String := match decode s.toUTF8 with
  | .ok P => hexOf P.sha512
  | .error e => (e.splitOn ":").head!

#guard verdict NOT_DESC == "8793c3fbb20b66b520af8987be9dc885a37593a13e866a6795c0619a6800aa07aa66e141722a09da83e0a3c30cda5339dd430122db2de67ff3271fa01bba536e"
#guard match (JsonParse.parse NOT_DESC).toOption.map definitionDigests with
  | some (.ok d) => d.toList.mergeSort (fun a b => a.1 ≤ b.1) == [
      ("Input1_v1", "01a1a9db691df5e9a51666f246f71239e9681d94750b07fa28e9ae24228055d112e38b257c0728673e6635057cad11565109279e2b71116867e38ae5d8fa18b7"),
      ("Not_v1", "e53494d3815c1d18a5eb3a18aae5781f0dfea16eec4cb1263e08f7b875c358a06b6007f7bf6de72f2b65cf840da36e3c9c84f4752f3bb32c3373654987230214"),
      ("Root[NotAnd_v1]_v1", "b5a3def87804c6f07a020e710c0b1349c1719870efedebc9028a4a98329b57ac49e1dcc7a8ccce3a64fb2583a35e4b1a99d812264c8f7935ba3be2a35aa68bd0")]
  | _ => false
#guard verdict (NOT_DESC.replace r#""boolean":{},"# r#""boolean":{},"word":{},"#) == "E_KEYS"
#guard verdict (NOT_DESC.replace r#""schema":"verity-ir/descriptor/v2","# r#""schema":"verity-ir/descriptor/v1","#) == "E_SCHEMA"
#guard verdict (NOT_DESC.replace r#""inputs":[["x",1]],"# "") == "E_INPUTS"
#guard verdict (NOT_DESC.replace r#""inputs":[["x",1]]"# r#""inputs":[["x","1"]]"#) == "E_INPUTS"
#guard verdict (NOT_DESC.replace r#""inputs":[["x",1]]"# r#""inputs":[["",1]]"#) == "E_INPUTS_NAME"
#guard verdict (NOT_DESC.replace r#""inputs":[["x",1]]"# r#""inputs":[["x",1],["x",0]]"#) == "E_INPUTS_NAME"
#guard verdict (NOT_DESC.replace r#""inputs":[["x",1]]"# r#""inputs":[["x",2]]"#) == "E_INPUTS_COUNT"
#guard verdict (mixedDesc r#""inputs":[["x",2]]"#) == "E_INPUTS_MIXED"
#guard (verdict (mixedDesc r#""inputs":[["x",1],["y",1]]"#)).length == 128
/-- `NOT_DESC` with a 64-leaf supplied input node after the run input node (`log.base/v1`'s 8 bytes, read `le`), the
parameters as `inputs` names and marks them. -/
private def suppliedDesc (inputs : String) : String :=
  ((((NOT_DESC.replace r#"["Input1_v1","Not_v1"]"# r#"["Input1_v1","Not_v1","SuppliedInput1_v1"]"#).replace
    r#""Root[NotAnd_v1]_v1":{"# r#""Root[NotAnd_v1]_v1":{"#).replace
    r#""form":"batch","n":1,"out":0},"# r#""form":"batch","n":1,"out":0},{"args":[],"axes":[],"fn":2,"form":"batch","n":64,"out":0},"#).replace
    r#""ret":{"affine":["n",1,0,1]}"# r#""ret":{"affine":["n",2,0,1]}"#).replace r#""inputs":[["x",1]]"# inputs
  |>.replace r#""Root[NotAnd_v1]_v1":{"bindings""# r#""SuppliedInput1_v1":{"form":"primitive","id":"SuppliedInput1_v1","params":[],"ret":["Value",1]},"Root[NotAnd_v1]_v1":{"bindings""#
  |>.replace r#""n":64,"out":0"# r#""n":64,"out":2"# |>.replace r#"["Value",1]]}"# r#"["Value",1],["Array",64,["Value",1]]]}"#

private def logMark (m : String) : String := s!"\"inputs\":[[\"x\",1],[\"base\",1,{m}]]"
private def LOG_MARK : String := r#"{"args":{},"bits":64,"layout":"le","supplied":"log.base/v1"}"#

/-- `decode`'s whole refusal text on `s`, or the empty string on an accept. -/
private def refusal (s : String) : String := match decode s.toUTF8 with
  | .ok _ => ""
  | .error e => e

#guard (verdict (suppliedDesc (logMark LOG_MARK))).length == 128
#guard verdict (suppliedDesc (logMark LOG_MARK)) != verdict (suppliedDesc (logMark (LOG_MARK.replace "log.base" "pous.setup.salt"
  |>.replace "64," "192,")))
#guard refusal (suppliedDesc r#""inputs":[["x",1],["base",1]]"#) ==
  "E_INPUTS_MIXED: input 'base': supplied input nodes and no mark naming their function"
#guard refusal (suppliedDesc r#""inputs":[["x",2,{"args":{},"bits":65,"layout":"le","supplied":"log.base/v1"}]]"#) ==
  "E_INPUTS_MIXED: parameter 'x''s input nodes 0..1 mix run and supplied inputs"
#guard refusal (NOT_DESC.replace r#""inputs":[["x",1]]"# (s!"\"inputs\":[[\"x\",1,{LOG_MARK}]]")) ==
  "E_INPUTS_MIXED: input 'x': marked supplied, and its nodes are ['run']"
#guard refusal (suppliedDesc (logMark (LOG_MARK.replace r#""bits":64"# r#""bits":0"#))) ==
  "E_SUPPLIED_MARK: input 'base': a mark is {supplied, args, layout, bits}, a part's with of = [input, lo, hi], not " ++
  "[{'args': {}, 'bits': 0, 'layout': 'le', 'supplied': 'log.base/v1'}]"
#guard verdict (suppliedDesc (logMark (LOG_MARK.replace r#""layout":"le""# r#""layout":"be""#))) == "E_SUPPLIED_MARK"
#guard verdict (suppliedDesc (logMark (LOG_MARK.replace r#""args":{}"# r#""args":{"k":""}"#))) == "E_SUPPLIED_MARK"
#guard verdict (suppliedDesc (logMark (LOG_MARK.replace r#""args":{}"# r#""args":{},"of":["b",1,1]"#))) == "E_SUPPLIED_MARK"
#guard verdict (suppliedDesc (logMark (LOG_MARK.replace r#""args":{}"# r#""args":{},"seed":1"#))) == "E_SUPPLIED_MARK"
#guard refusal (suppliedDesc (logMark (LOG_MARK.replace "log.base/v1" "log.base/v2"))) ==
  "E_SUPPLIED_FN: input 'base': 'log.base/v2' is no supplied function"
#guard refusal (suppliedDesc (logMark (LOG_MARK.replace r#""args":{}"# r#""args":{"round":3}"#))) ==
  "E_SUPPLIED_ARGS: input 'base': log.base/v1 takes the args [], not ['round']"
#guard refusal (suppliedDesc (logMark r#"{"args":{"value":70000},"bits":16,"layout":"le","supplied":"const.u16/v1"}"#)) ==
  "E_SUPPLIED_ARGS: input 'base': const.u16/v1 has no value at {'value': 70000}"
#guard refusal (suppliedDesc (logMark (LOG_MARK.replace r#""bits":64"# r#""bits":32"#))) ==
  "E_SUPPLIED_WIDTH: input 'base': log.base/v1's value is 64 bits, read by le as whole leaves of one width; the mark says " ++
  "32 and its nodes hold 64 leaves of widths [1]"
#guard verdict (suppliedDesc (logMark (LOG_MARK.replace r#""layout":"le""# r#""layout":"digest""#))) == "E_SUPPLIED_WIDTH"
#guard (verdict (suppliedDesc (logMark r#"{"args":{},"bits":64,"layout":"le","of":["salt",0,64],"supplied":"window.salt/v1"}"#))).length == 128
#guard verdict (suppliedDesc (logMark r#"{"args":{},"bits":64,"layout":"le","of":["salt",500,564],"supplied":"window.salt/v1"}"#)) == "E_SUPPLIED_WIDTH"
#guard pyStr "it's" == "\"it's\"" && pyStr "a\\b" == "'a\\\\b'" && pyStr "\x01" == "'\\x01'"
#guard verdict (NOT_DESC.replace r#""definitions":{"# r#""definitions":{"Const1[0x1]_v1":{"form":"primitive","id":"Const1[0x1]_v1","params":[],"ret":["Value",1]},"#) == "E_UNREACHED"

end Flock.Program.InputsTest
