/-
A templated statement's template section, right after META: `HmOut.tmplOf` reads its header `TEMPLATE n` and the
template's `n` lines off the file's head alone, so bytes past the section that aren't UTF-8 still give the template, and a
file without the section, with a malformed header or with fewer lines than its header names gives none.
`HmRow.tmplLines`, what `HmRow.parse` passes over before the sections, is the header and those lines under the templated
tags, and nothing under any other's or in a file without the section. `HmOut.parseTmpl` refuses a META `definition`, a
unit class's, before it reads the template. `HmOut.tmplOfView` reads the same template, and the same errors, off an
ASCII file's line view, and the template's parse through the view (`View.tmplLine`, `View.tmplNums`) prints
`Flock.Test.Tmpl`'s template as its parse of the lines does, and refuses a row that isn't numbers with the same error.
-/
module

public import Flock.HmRow
public import Flock.Test.Tmpl
public import Flock.TmplLines
meta import Flock.HmRow
meta import Flock.Test.Tmpl
meta import Flock.TmplLines

@[expose] public section

namespace Flock.Test.TmplSection

open Flock

def file (sec : String) (rest : ByteArray := "CIRCUIT x 0\n".toUTF8) : ByteArray :=
  "flock-circuit/tmpl u\nMETA {\"layout\":\"iv\"}\n".toUTF8 ++ sec.toUTF8 ++ rest

def linesOf (b : ByteArray) : Array String := (((String.fromUTF8? b).getD "").splitOn "\n").toArray

def two : ByteArray := file "TEMPLATE 2\na b\nc d\n"

#guard (HmOut.tmplOf two).toOption == some #["a b", "c d"]
#guard (HmOut.tmplOf (file "TEMPLATE 0\n")).toOption == some #[]
#guard (HmOut.tmplOf (file "TEMPLATE 1\na b\n" ⟨#[0xff, 0xfe]⟩)).toOption == some #["a b"]
#guard (HmOut.tmplOf (file "TEMPLATE 3\na\nb\n" .empty)).toOption == none
#guard (HmOut.tmplOf (file "TEMPLATE two\na\n")).toOption == none
#guard (HmOut.tmplOf (file "TEMPLATE 1 2\na\n")).toOption == none
#guard (HmOut.tmplOf (file "")).toOption == none

def viewOf (b : ByteArray) : Option (Except String (Array String)) := (ByteLines.View.of? b).map HmOut.tmplOfView

#guard (viewOf two).map Except.toOption == some (some #["a b", "c d"])
#guard (viewOf (file "TEMPLATE 0\n")).map Except.toOption == some (some #[])
#guard (viewOf (file "TEMPLATE 1\na b\n" ⟨#[0xff, 0xfe]⟩)).isNone
#guard viewOf (file "TEMPLATE 3\na\nb\n" .empty) matches some (.error "circuit: the template section is truncated")
#guard HmOut.tmplOf (file "TEMPLATE 3\na\nb\n" .empty) matches .error "circuit: the template section is truncated"
#guard (viewOf (file "TEMPLATE two\na\n")).map Except.toOption == some none
#guard (viewOf (file "TEMPLATE 1 2\na\n")).map Except.toOption == some none
#guard (viewOf (file "")).map Except.toOption == some none

/-- `lines` as a templated file's template section. -/
def tmplFile (lines : Array String) : ByteArray := file s!"TEMPLATE {lines.size}\n{"\n".intercalate lines.toList}\n"

/-- The template's parse through the file's view, printed flat (`Tmpl.flatLines`), beside its parse of `tmplOf`'s lines. -/
def viewParsed (b : ByteArray) : Option (Except String (Array String) × Except String (Array String)) :=
  (ByteLines.View.of? b).bind fun v => (HmOut.tmplCountView v).toOption.map fun k =>
    (_root_.Flock.Tmpl.parseAt k (v.tmplLine k) (v.tmplNums k) >>= _root_.Flock.Tmpl.flatLines,
      HmOut.tmplOf b >>= _root_.Flock.Tmpl.parse >>= _root_.Flock.Tmpl.flatLines)

#guard (viewParsed (tmplFile Test.Tmpl.tmplLines)).map (fun (x, y) => (x.toOption, y.toOption)) ==
  some (some Test.Tmpl.flatText, some Test.Tmpl.flatText)
#guard match viewParsed (tmplFile #[Test.Tmpl.header, "DEF 1 2 1 0", "0 1 x 1 1"]) with
  | some (.error e, .error e') => e == e' && e.endsWith "definition 0: row 0 is not numbers"
  | _ => false

#guard HmRow.tmplLines Tags.circuitTmpl (linesOf two) == 3
#guard (linesOf two)[2 + HmRow.tmplLines Tags.circuitTmpl (linesOf two)]! == "CIRCUIT x 0"
#guard HmRow.tmplLines Tags.circuit (linesOf two) == 0
#guard HmRow.tmplLines Tags.circuitTmpl (linesOf (file "")) == 0

def classMeta : ByteArray := "flock-circuit/tmpl u\nMETA {\"definition\":\"d\",\"layout\":\"iv\"}\nTEMPLATE 0\n".toUTF8

#guard HmOut.parseTmpl Tags.circuitTmpl classMeta matches
  .error "META definition: a unit class has no templated statement"

end Flock.Test.TmplSection
