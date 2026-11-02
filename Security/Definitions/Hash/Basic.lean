import Definitions.Core.Hash

/-!
# Messages, digests, collisions, and the byte encodings: Core's

The hash vocabulary (`Hash.Msg`, `Hash.Collision`, `Hash.CollisionIn`, the byte encodings) is Core's
(`Definitions/Core/Hash.lean`), so that Core's interaction model (`Definitions/Core/Interaction.lean`) reads it within
Core's layer. The hash area's modules read it through this one.
-/
