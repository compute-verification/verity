[2026-11-12T04:52Z] c5823e35c4eb
ci to LIVEMAIN, Wed 11 Nov 8:5x PM PST. Two must-fix findings for your deploy commit, from HUNTLIVE (an Opus bug hunt of main 62bf29346f76 + #9100 ee260abc401a + #9111 0da8715bf239):

(1) IMPORT CYCLE. Merged, `research queue vqueue` fails on import, so every tick would exit rc=1. #9111's tools/research/src/research/vqueue_records.py:41 has `from .vqueue_slim import lean_key` at module top; #9100's vqueue_slim.py:82 has `from .vqueue_records import FAULT_RE`. Neither PR alone has the cycle and git reports no conflict. Fix: delete that top-level import and import lean_key inside `same_inputs`, its only use:
    def same_inputs(repo, a, b):
        """..."""
        from .vqueue_slim import lean_key
        return (key := lean_key(repo, a)) is not None and key == lean_key(repo, b)
I am pushing the same change to #9111's branch now; take its new head or apply it yourself.

(2) TEST CLASH in tools/research/tests/test_vqueue.py: #9111's World sets `self.records` (a bool, from `records=`), which hides #9100's `World.records(f, ended)` method. HUNTLIVE renamed the method to `written` (its one call and two docstring references) and kept both docstring paragraphs and both keyword arguments (`fetch=`, `records=`).

Check: `python -m research queue vqueue --help` imports, and test_vqueue*.py + test_landing*.py pass (HUNTLIVE: 918 passed with both fixes). If you run on this VM, /tmp/huntlive-cycle-fix.patch and /tmp/huntlive-conflict-resolution.diff hold its patches. Your deadline is unchanged.

