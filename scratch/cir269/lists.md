# CIR-269: the catalog's list at main and at #9660's head

Two files beside this one. Each is `circuit-check --all -q --cache <fresh dir> --bindings -j 8 --mem-gb 40 --out
bindings.json`, run by `research run --on auto --cwd source --mem-gb 48`. That command enumerates the catalog's targets and
checks none.

- `bindings-base-99cc585fa8f2.json`: run `r20261113-054134-6b3d` at `main` `99cc585fa8f28b2a0ca5e1d3b7ca3c964ff2725e`,
  which is the merge base of #9659 and #9660 after `origin/main` was merged into them.
- `bindings-head-821617b1535d.json`: run `r20261113-054135-4b2c` at #9660's head
  `821617b1535dfbda192e1797caef4d2fc65c5006`.

Each run printed: `283 catalog bindings, 17322 target(s), 0 root(s) not made, 0 registered family(ies) unchecked`.

The two files are byte for byte the same (sha256 `7cf0963110500f2a8241dfbc589b97a90158f862302efeb0025a036f15431d21`).

- By id, base and head hold the same 17,322 targets: 17,287 Definitions and 35 templates. Neither has a target the other
  lacks.
- The 17,322 (binding, target) pairs are equal too.
