# `ValueBinding`: each wire's value is bound to its registered commit string

`Audit.Partition.ValueBinding` (`FlockSoundness/Audit/FlockLink.lean`), taken as `vb` by the link theorem
(`flock_batched_linkSoundE`) and the composed audit (`flock_batched_count_linked`, `_drawn_linked` and the `_placed`
forms).

- **What it says.** A satisfying message of a drawn unit's table opens the commit string of each of the unit's wires,
  registered at `R`, to a value whose bit on the wire is the decoded one (`opening_spec`). Two openings of one string
  with different values give a collision of the leaf hash (`collide_spec`).
- **Why it is data.** It is a structure of functions with specifications, not a `Prop`, because the link finder calls
  `opening` and `collide`.
- **To be discharged by** the `hm96-sha512` leaf layout (M0) in the model. The openings are then Halevi–Micali
  openings, and `collide` is their binding reduction. The canonical `hm96` values and `registered_weights` need the same
  layout (`../README.md` §1.3).
