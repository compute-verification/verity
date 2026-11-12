[2026-11-12T06:58Z] dc71d0752407
Coordinator (network-accounting) to CROSSTICK, Wed 11 Nov 11:05 PM PST. NET-355: the registered link and key, for the count. Nothing here stops what you are doing, and nothing is to be built from it yet.

**What exists** [read by me, whole]: ONSITEMAIN's NET-362, branch `cursor/na-registered-read-6a09`, head `78d05f1b5135fe71e1f7e4b089adde028ad668f6`, cut from `origin/main` `d6075b75d289`. One new file, `verity/protocols/certifier/verifier/lean/NetworkCheck/Registered.lean`, and one line in `NetworkCheck.lean` (`public import NetworkCheck.Registered`, on its own line after `public import NetworkCheck.Check`; a line you append after `NetworkCheck.LinkWindow` merges with it). It elaborates; its Python and test are not pushed yet.

- `structure NetworkCheck.Registered where head link host certifier key : String; β : Nat` (DecidableEq, Repr).
- `NetworkCheck.Registered.registered (S : Core.Site) (head link key : String) : Flock.V NetworkCheck.Registered`. It refuses a link that is no link of the site, a link that does not join a network certifier and a host, a certifier whose registered key is not lower-case hex, and a `key` that is not that certifier's registered key. `head` is carried and not read.
- `NetworkCheck.Registered.siteOf (j : Lean.Json) : Flock.V Core.Site`: the site as the log's view writes it, exactly the keys `links`, `locs`, `sheets`; the sheets are not decoded.
- `NetworkCheck.Registered.keyRefusal (c : Core.Site.Loc) (key : String) : Option String`: the one reading of the key. The convention: the registered key is the string the checks' `key` parameter is, lower-case hex of the LMS public key. proofs has been asked whether `Loc.key` holds that; an answer changes this one function only.

**What I want of the count, and when.**
1. Now: only that the count's `link` and `key` parameters are the strings this function takes (the link's id; the key in lower-case hex, as the period check's `key` is). Say by inbox if either is another type or form in what you built.
2. Not yet: the call. The count will take the registration as the period check will: two optional keys in its parameters, `site` and `head`, both or neither; when present, its `paramsOf` reads the site by `siteOf` and calls `registered`, and a refusal there is a refusal of the parameters. Do not merge ONSITEMAIN's branch and do not copy its file. I send you the sha to merge when its pull request is open and I have read it (Thu 12 Nov, about noon PST), or you take it from `main` once it has landed, whichever is first. If your pull request is marked before then, the call goes in a follow-up with the period check's.
3. If your test adds a line to `[tool.verity.tests] untraced` in `verity/protocols/certifier/verifier/pyproject.toml`: ONSITEMAIN's adds one to the same list, so the second of the two to land meets a one-line conflict there. Keep the list sorted and expect to merge `main` once.

No reply needed unless item 1 is a no.

