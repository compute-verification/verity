module

public import Definitions.Claims.Encodings
public import Proofs.Flock.Encodings.Binding
public import Proofs.Flock.Encodings.Relink

/-!
# The registered encodings: their binding statements, proved

The four binding statements of `Definitions.Claims.Encodings` are proved with no assumption, each a collision reduction:
the sum by position binds when its parts do (`byPosBinds`), the encoding binds within the unique decoding radius at a
binding column check (`codeBinds`), the link binds the record to the encoding (`linkBinds`), all in
`Encodings/Binding.lean`; and the sampling relation under the sum carries over the link to the record, or the checks
collide (`relinkR`, `Encodings/Relink.lean`, through `M_δ`'s downward closure at every design, `design_M_down`). Their
witnesses are `Claims.Witnesses.EncodingsGeneric`'s. The composed claims (`CompilerKSLinked`, `CompilerKSRegLinked`)
are stated, not yet proved.
-/

@[expose] public section
