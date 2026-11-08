/-
A templated statement's template section, right after META: `HmOut.tmplOf` reads its header `TEMPLATE n` and the
template's `n` lines off the file's head alone, so bytes past the section that aren't UTF-8 still give the template, and a
file without the section, with a malformed header or with fewer lines than its header names gives none.
`HmRow.tmplLines`, what `HmRow.parse` passes over before the sections, is the header and those lines under the templated
tags, and nothing under any other's or in a file without the section.
-/
module

public import Flock.HmRow
meta import Flock.HmRow

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

#guard HmRow.tmplLines Tags.circuitTmpl (linesOf two) == 3
#guard (linesOf two)[2 + HmRow.tmplLines Tags.circuitTmpl (linesOf two)]! == "CIRCUIT x 0"
#guard HmRow.tmplLines Tags.circuit (linesOf two) == 0
#guard HmRow.tmplLines Tags.circuitTmpl (linesOf (file "")) == 0

end Flock.Test.TmplSection
