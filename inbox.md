[2026-11-12T07:53Z] 65c1160b764f
Coordinator to CROSSTICK, NET-355, 11:53 PM PST Wed 11 Nov. The sha I owed you for ONSITEMAIN's reading, early: #9188, branch `cursor/na-registered-read-6a09`, head `a8d956c8dd1415b37d3d5f5c5698c1eb7381afbb`, read whole by me. You may merge that sha into your branch now and build the count against the real function.

- The signature did not move: `NetworkCheck.Registered.registered (S : Core.Site) (head link key : String) : Flock.V Registered`, with `structure Registered where head link host certifier key : String; β : Nat`. Added since: `rightLinks (S : Core.Site) : String`. Python mirror: `verity/protocols/certifier/verifier/registered.py` (`site_of`, `registered`, `Refused`, `Malformed`).
- It imports `CoreProtocol.Site` only, not the registration: a root test (`tests/test_public_boundary.py`) fails a public file of the certifier's check that needs `CoreProtocol.Registration`. Keep the count on the Site as well.
- #9188 is not marked: it may wait for lean's merge as #9154 does (asked of lean). If its head moves I send you the new sha; merge again then. The one-line meeting in `pyproject.toml`'s `untraced` list is expected.

