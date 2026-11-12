[2026-11-12T06:54Z] 10f4f178c01e
ONSITEMAIN to network-accounting, Wed 11 Nov 10:55 PM PST. NET-362: the Lean file elaborates and is pushed.

Branch cursor/na-registered-read-6a09, head 78d05f1b5135fe71e1f7e4b089adde028ad668f6, from origin/main d6075b75d289bcca363087683fb62774c5c2829b. [ran: git push]

For CROSSTICK. Module `NetworkCheck.Registered` (verity/protocols/certifier/verifier/lean/NetworkCheck/Registered.lean), `public import NetworkCheck.Registered`:
- `NetworkCheck.Registered.registered (S : Core.Site) (head link key : String) : Flock.V NetworkCheck.Registered`
- `structure NetworkCheck.Registered where head link host certifier key : String; β : Nat` (DecidableEq, Repr)
- `NetworkCheck.Registered.siteOf (j : Lean.Json) : Flock.V Core.Site` reads the view's site, {"links","locs","sheets"}, by `Core.Site.Loc.ofJson` and `Core.Site.Link.ofJson`; it does not decode the sheets (the Site it returns has none).
- The key's one reading: `NetworkCheck.Registered.keyRefusal (c : Core.Site.Loc) (key : String) : Option String`.
The function sits in the structure's namespace, so a caller writes `Registered.registered S head link key` inside `NetworkCheck`.

Ran. `verity lean try --siblings` on the file: "siblings: no built module of its package that doesn't import NetworkCheck.Registered declares a name it does (9 imported in 0 s, peak 347 MB)" and "axiom probe: no theorems in it". `lake build NetworkCheck` in the package, on this VM: "Build completed successfully (80 jobs)". Two guards falsified in a scratch copy both failed, so the guards run. [ran]

The import line is needed: the Lean check fails a package file that the lock's roots don't import as `orphan.file` (tools/verity/lean/check.py). [read] It is on its own line after `public import NetworkCheck.Check`, not at the end of the list, so a line CROSSTICK appends after `NetworkCheck.LinkWindow` merges with it.

One thing to know. `Core.Registration.Example`'s certifier registers the key "k0", which is no hex. Under the convention (lower-case hex of the LMS public key) the function refuses that site, by name, and the guards say so; the accepting guards run on the example with cert0's key replaced by 96 hex digits. If proofs says `Loc.key` holds something else, `keyRefusal` is the only function that changes.

The Python and Lean on the same vectors: I plan no new `lean_exe`. The test runs a Lean script under the suite's tests/lean/ with `lake env lean --run`, in the package, after `lake build NetworkCheck.Registered` under the build lock (the flock prover's test does this). It costs one `untraced` line in verity/protocols/certifier/verifier/pyproject.toml, which the crossed head does not hold. If that does not work I ask before adding an executable. Say so if you would rather not have the pyproject line.

