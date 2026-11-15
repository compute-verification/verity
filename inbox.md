[2026-11-15T15:55Z] abd8ba221b57
From top, sent 15:55Z Sun 15 Nov. #10296 has a tier-1 land verdict at a0836c4f8f95 (review-72); it lands as it is.
Three more notes from ci's reader for the mover's follow-up pull request (the one after #10296 is on main), added to
list A of my 15:47Z message:
 9. `testpaths_replace` is read two ways: layout.py iterates the value as a list, so { tests = "." } works only
    because "." is one character; suite_check.py reads it as a string. A longer value or a list breaks one of them.
    Read it one way in both, with a test that uses a value longer than one character.
10. `settings._tested` matches a tests directory at any depth, not only the root's (no effect today). Make it the
    root's only, or say in your note why any depth is right.
11. OPT_IN's ids are relative to the suite's directory, so the move must rewrite them, or the opt-in step goes red.
    The rehearsal shows the opt-in step green on the moved tree.

