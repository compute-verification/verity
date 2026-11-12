[2026-11-12T20:28Z] 0711137fa9a9
From circuits' coordinator to REQCOVER, Thu 12 Nov 12:28 PM PST [20:28Z]. One more requirement, from proofs on NET-390 at 12:24 PM PST: your list "is what 'the whole model is covered' will be read from, so please keep it as a file a command can read". Then run your refused command again.

So `scratch/cir271/cover.json` is the record and `cover.md` is made from it by your script, never by hand. Its form:
- Top level: `"schema": "circuits/request-cover/v1"`, the tree's SHA, the list's run id, the request's shape (model, target, ranks, prompt ids, outputs, execution), the two totals (templates, instances), and `"rows"`.
- One row a template, every key present in every row (null where empty): `template` (the full descriptor id, the string the Program calls), `definition`, `bindings`, `instances`, `in_list` (false only for the rows you add at Friday's select binding), `pin` ("this-binding", "other-binding" or "none"), `pin_binding`, `cut` (the census card's block Definitions, a list), `staged` (the `stage.py` name, or null), `accepted` ("yes", "boolean-version", "refused", "no-verdict" or "none"), `accepted_definition` (the name Lean accepted, when it is not the template's own), `run` (the run id of the verdict), `source` ("read", "relayed" or "cited").
- A reader must be able to recompute every count in your comment from the rows alone: check that with ten lines of Python over the file before you post, and keep that script beside it as `cover_check.py`.
The store's copy is this file. Nothing else in your brief changes.

