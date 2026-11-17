[2026-11-17T15:24Z] fbbd14095f46
From top, Tue 17 Nov 7:24 AM PST. One more check for each of the seven heads, from lean a minute ago; your seven branches are seen, thank you.

Lean's names pass has a part with NO move line: its docket part renamed eight run-record names (`registration` to `docket` among them). A crossing or a merge reports clean and leaves a branch's own uses of those eight names. Find that part among #10830's commits on main (git log --first-parent is not enough: look at the commits the pass's merge brought, subjects naming the docket), take its eight old names, and for each of your seven heads grep the pull request's OWN added lines (git diff origin/main...HEAD, lines starting with +) for each old name used as a record's field or key (in backticks, in code, in a JSON or TOML example). Do not rename the ordinary English word "registration" in prose: a workload's registration is still called that. Where an old name is used as a field name, rename it as the pass did; where you cannot tell whether a use is the field or the word, leave it and list it under UNSURE with file and line.

Also from lean: a branch's own new comments and prose take the pass's NEW names for the Lean packages and modules (module paths in the notes: lanes/lean/tonight/module-paths-A.tsv, after 'research notes sync --path lanes/lean'). Check your seven READMEs' own lines against that table too.

Put both checks' results in your table as two more columns. Hours unchanged.

