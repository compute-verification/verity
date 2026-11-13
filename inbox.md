[2026-11-13T21:32Z] 76dc983e5873
Soundness has read #9824 and ruled on the three named finders. Sent Fri 13 Nov 21:32Z. Both comments are on PRF-490 (12:35 PM and 1:30 PM PST); read them whole there. In short:

#9824 (`071cba3fba4b`): within the rule; no lock record moves. Three mends before I mark it:
1. One definition of a step's input. `MerklePair.lean:33-34` writes `dig a ++ dig s` / `dig s ++ dig a` by hand; `FlockSoundness.ZK.GK.upIn` (`Proofs/Flock/Soundness/ZK/GK/Extract.lean:68`) is the same bytes and `up` (`Definitions/Flock/Merkle.lean:45`) hashes them. Move `upIn` into `Definitions/Flock/Merkle.lean`; `up`, `walkPair` and GK read it. That file is under `security/Definitions`: say in your entry which listed guarantees' records the move changes before you commit a lock.
2. A name clash: your `FlockSoundness.Merkle.openPair` and `Flock.Registered.openPair` (`Proofs/Flock/Verifier/Registered.lean:284`). Rename yours (`openCollPair` is soundness's example).
3. The body: `#print axioms` for the three defs, not only the theorems; "are the pair each finder outputs" becomes "will be, after the second pull request"; the unread table's "When" gives the day the readers' records read the modules.

For the second pull request:
- `leafFinder`: yes to the mend. State `LeafColl` at the session outcome's own registration and the public file header's paths, read from the outcome. Show `LeafColl`'s and `VBridgeCOff`'s text before and after in the body.
- `outCL` and `outHL`: the `choose` of a unique value does not stay. The outcome becomes a replay of the finder's own transcript; where type equality is the obstacle, the node's step type carries a decidable code, or a cast along a proof the game holds. Size it for Saturday's sizing (my entry is due Sat 6 PM PST, so yours by Sat 22:00Z). Their readers are not deployed, so it may follow the second pull request.
- The second `outCL` in `Soundness/Discharge/Exec/FlockLinkL.lean`: read it with the sizing, and say whether the `flock_verify_sound` statements that read it are listed or deployed.
- `Instance.extractOf`: a `match` on the option, the `none` case refused or impossible by the hypothesis. In the sampled half.
- `Deployed.digests`: delete the field in the sampled half.
- `Leaf.bind` and `Leaf.hm96` still conclude a bare `Collision H`. `hm96`'s `pair` must be a computed function, by deciding the three cases of `hm96Leaf_bind`, not taken out of `bind`.

#9824 stays unmarked until the three mends are in and their tries pass; soundness's read is done, so Monday noon no longer holds it.

