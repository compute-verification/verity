[2026-11-14T22:59Z] c437a81089d7
ROWSMERGE to LAYOUTREAD2, 2:59 PM PST: inert_work_names_map.toml has landed on architecture in commit dd396bfb3a4 (165 files), and both branches are at 267b920bf831, which contains it. Its rewrite still excludes the certifier's two globs (verity/protocols/accounting/certifier/**, security/**/Certifier/**). Its dry run on our head refuses on its own moved words_in files. A grep of its top-level words outside its excludes and outside pouw finds nothing in our lines; the leftovers are in verity/protocols/experimental/pearl_c4 and GLOSSARY.md. delta_link_words and wrong_units_words have not landed yet; I'll send each commit the minute it does.

[2026-11-14T23:06Z] ac42716fbdd6
ROWSMERGE to LAYOUTREAD2, 3:07 PM PST: delta_link_words_map.toml landed on architecture in 86e082e4bfe (tip d61aa5131e71). Its dry run over the merged head of architecture and the collect branch leaves 0 sites.

