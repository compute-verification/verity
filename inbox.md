[2026-11-15T22:31Z] 420569518638
circuits lead, Sun 22:31Z: ci's reviewer gave #10397 a land verdict at 0b43131e86d9 (review-75), with one bookkeeping note: the body has no circuit-check line. None applies (no Definition, template or Boolean lowering changes; the units it picks have sidecars on main), so SAY that in the body, as #9885 did. Do the same in #10388 and #10404's bodies where it holds, and where a Definition or template does change, put circuit-check's report lines in.

