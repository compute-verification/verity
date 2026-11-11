[2026-11-11T20:31Z] e46fac31a65c
FORM to training, TRN-103, 20:31Z Wed 11 Nov. THE RECORDS AT c1ac6b770a PASS, ALL THREE; and Thursday's two things a day early, since nothing in them is waiting any more.

THE RECORDS AT THE HEAD THAT STAYS (RAN, r20261111-193246-4d1d at c1ac6b770a, `lean_changed.py --records security/Proofs`, "done rc=0", OUTCOME "pass", nothing regenerated):
- security/Proofs: "PASS  196892 declarations in 3927 modules; axioms propext, Classical.choice, Quot.sound"; "check.py exit 0 in 689 s".
- security: "PASS  26980 declarations in 716 modules"; "check.py exit 0 in 451 s".
- security/Properties: "PASS  3250 declarations in 95 modules"; "check.py exit 0 in 929 s".
Each "without the kernel replay or the tests: a lean-fast/v1 result". The same counts as WIDTH's build at 16c91c972e.

THE SUITES: r20261111-180142-a092 still has 57 of 69, none failed; it has been on verity-vllm for over two hours. So that the eleven behind it do not hang on its limit, LAUNCHED 20:30Z: r20261111-203011-327b at c1ac6b770a, the delta (`--changed 16c91c972e`) and then those eleven by name. I push and edit the body when all twelve are read and pass, as you ruled; if one fails, its line comes first and nothing is pushed.

THURSDAY'S LINE (owed Thu 8 AM PST). Compiled: `form_owns`, `form_rates`, `form_laidOut`, `form_counts`, `form_complies`. Left: none.

THURSDAY'S PARAGRAPH (owed Thu 10 AM PST):
- THE DAY FOR `Counts`: today, Wed 11 Nov. `form_counts : Counts form` compiles (READ: COUNTS' r20261111-172620-d093) and is in both builds of the candidate (RAN: 4d1d; READ: 540f). It hangs on nothing open: `block_fresh` (COUNTS), `fresh_rows` (WIRED), `block_gates` (WIDTH), each a theorem with no hypothesis left over.
- W, a gate is as wide as its leaf. The statement the assembly reads is `leafWide` (`RecordLeafWide.lean`, 332 lines). Under it are seven modules, 2,351 lines (RAN: wc): `LeafWidths` (the statements, 47), `LeafWidthsType` (451) and `LeafWidthsRange` (405) and `LeafWidthsPart` (393), which are W1b, the decoder's closed forms; `LeafWidthsBody` (219); `LeafWidthsResolve` (782), W2, the width kept along `resolve`; `LeafWidthsDecoded` (54), which closes it for a decoded Program. Who proved the thirteen: groups 1 to 4 COUNTS and 5 and 6 WIDTH, as you assigned them; I did not re-derive each file's author from git, where the merges hide it.
- W1b was the large part, as sized: three modules, 1,249 lines. `LeafHolds` did not shorten it: only the landing bound reads the weaker statement (`RecordLanding.lean`: `leafHolds_of_leafWide`, `landing_le_of_holds`).
- `RootFirst` is closed by `root_first` (`DecodeKeys.lean`, WIDTH's): no hypothesis of that name is left.
- M7, every bit of a wired row is its writer's: `filled`, with `aligned`, in `RecordAligned.lean`, 528 lines (WIDTH's). `filled` takes no `hO`.
- NO LEMMA TOOK `UnitsComposite`. RAN: grep over every `.lean` of `security` and `verity` at c1ac6b770a, no hit. The check asks instead that a batch's every argument has `dims`.
- WIRED's THREE UNCHECKED THINGS, where each went (READ): the ranking from `Spot.Wires.check` is `check_ranked` (`Spot/WiresSound.lean`), which `served_of_read` calls as `ranked`; `resolve` gives a gate below `gates P` in `ResolveWithin.lean` (`gate_within`, `operands_within`, `ret_within`); the zero-width leaf I did not trace to a lemma by name. INFERRED: it is inside `leafWide`'s proof, since `form_laidOut` compiles with no hypothesis; say if you want the line.

