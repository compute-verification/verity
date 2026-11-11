[2026-11-11T09:30Z] 0d36a77d1fb4
bounds, TRN-67, Wed 11 Nov 09:30Z. #8490 IS MARKED AND ADMITTED.
Result line (09:28Z, checkout of origin/main 0a722c6ec984, plain): "research queue: #8490 is ready at 8848940a322f (pr:8490@8848940a322f0dcb214d3dba4699ff3e59c1e1f6)". Preflight: "merged onto main at 0a722c6ec984: nothing refused (163 s; Lean: built 2 module(s), run r20261111-092605-5fb4, 0 failed; collected 4 suite(s))". No records warning: it changes no lock.
Queue's line (09:29Z): "#8490 8848940a ready training admitted: the verified queue builds it into a batch onto main once it has a land verdict, or on its mark alone when it is low-risk".
Rests on (read, at 8848940a322f): Lean check from nothing r20261111-072559-cf50 exit 0, AUDIT: PASS, 3 packages (714 + 3,817 + 94 modules), 0 failed, regenerated {}, every module replayed, none reused; unread ratchet exit 0; suites r20261111-072624-1cd4 exit 0: 4 suites, 4 passed, 0 cached, 0 failed. The body names them (sent 09:25Z, before the mark).
THE OTHER FOUR at 09:29Z: all still "ready, admitted" at their marked heads, and each now has a land verdict at its head in the store: #8810 review-16 07:29Z, #8811 review-15 07:33Z, #8593 review-16 07:40Z, #8595 review-15 07:40Z. #8490 has none at 8848940a322f yet (its old land, review-4 at ca2b9ade86fc, is at the old head).
main is 0a722c6ec984 now; all five heads merge onto it without conflict (git merge-tree, 09:23Z); it changed nothing under security/, its lock equals the crossing base's, and no file on it outside the stack's own names a guarantee or imports a module the stack removes (git grep over the 31 guarantees and 33 modules).
Next: the mark's line into #8490's body, the hand-off note, one message with the five marks, and I end.

