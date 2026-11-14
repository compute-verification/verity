Lane proofs
Issue: none: the retired-terms debt of one file that #9513 brought to `main`, sized in the lane's notes; no issue was filed for it

**`ChargeOnce.lean`'s three theorems go into a namespace of their own, `FlockSoundness.Audit.ChargeOnce`, and the file's entry in the retired-terms list goes from 5 to 2.** 2 files, 9 lines in and 9 out against `main` `@BASE12@`. No statement's text changes. **No lock record moves**: no lock names the three theorems, their module or either namespace.

## What changes

- **`security/Proofs/Flock/Claims/ChargeOnce.lean`, the namespace.** `namespace FlockSoundness.Audit.Periods` becomes `namespace FlockSoundness.Audit.ChargeOnce`, and `open Claims` becomes `open Claims FlockSoundness.Audit.Periods`. The three theorems' full names change with it: `Ex_le_of_forall`, `Ex_fin_cons` and `ex_prod_add_le`.
- **The same file, its module docstring.** It named the audit theorems' file twice by its file name. It now says "the audit theorems' module imported above" and "that module's", and says that this module's declarations are in a namespace of their own and read that module's `Adapted` and `Predictable`.
- **`tests/lint/allowlists/repo_retired_terms.json`.** The file's entry goes from `"count": 5` to `"count": 2`. Only that line changes.

## Why

#9513 brought the file to `main` with five sites of a retired word: the name of the audit theorems' module and namespace, which the Glossary retires for "monitoring period". Three of the five were the file's own choice: it declared its theorems inside that namespace, from a module outside that directory, and its docstring named the file. This change takes those off.

## What stays: 2 sites, both names `main` declares

- **Line 4, the import of the audit theorems' module.** The module's name is the retired word.
- **Line 33, the line that opens that module's namespace.** The statement of `ex_prod_add_le` reads its `Adapted` and `Predictable`.

Both go when that module and its namespace are renamed. That is a move of its own and not this change: other Lean files import the module, and two verifiers' Python name one of its theorems by its full name.

## Why nothing else moves

- **No statement's text changes** [`git diff`]: the diff of the Lean file is the docstring, the `namespace` line, the `open` line and the `end` line.
- **Nothing else names the three theorems by a full name** [`git grep` at this head]: outside the file, only `security/Proofs/Flock/Soundness/Audit/README.md` names one, `ex_prod_add_le`, by its short name. No lock, list or Python file names them, the module or the new namespace.
- **The new namespace is new** [`git grep` at `main` `@BASE12@`]: nothing else in the tree spells `ChargeOnce` as a name.

## Checked

@CHECKED@
