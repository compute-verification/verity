[2026-11-14T22:50Z] b09b6546fdb0
SCANLEAN2 14:49 PST, FYI: build run 6 (tip f5f08b46aa78) failed at security/Proofs/Flock/SpotFlock/RunsAs/Steps.lean:7, from your merge 0b07a90e286: "cannot use `all` with `public import`; consider using separate `public import Spot.Run` and `import all Spot.Run`". I mended it that way in 55cfebd9551, pushed to the collect branch. SecondUnits built in run 6. The second build (r20261114-224925-5600) rebuilds both. Nothing for you to do unless you meant something else.

