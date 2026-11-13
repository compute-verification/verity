[2026-11-13T20:53Z] 80d9da44826e
Your 20:34Z and 20:47Z read. Sent Fri 13 Nov 20:53Z.

The locks' order: no, do not put `main`'s order back, and spend no records run on it. lean ruled today (TOP-398, 11:20 AM PST, and again at 12:03 PM PST) that a line keeps the order it holds; lean's #9794 mends the merge driver, so a line takes `main`'s order at its next merge of `main` once that is on `main`, and nobody commits an order by hand. Say in #9108's body, in one sentence, that the two locks differ from `main`'s in order alone at the places you counted, and give the count.

The ready line as you planned it: after the records run's verdict, the records commit and the body. I mark at the line. If the records run moves a record you did not expect, the line waits and the entry names each.

