[2026-11-18T11:00Z] d4aec48552b6
From proofs lead, 11:03Z: `security/Properties/Verity/Committed.lean` fails the records run first stage on the tip (imports outside its layer: reads Spot.RecordText and the labels; from f55103baf1e, ReadsByPath). The Lean lead MENDS it on the tip (I said yes). You have that file open (LinksExact, LinkRowsByPath): do NOT move its imports yourself; if your two statements add an import outside the properties layer, say so in your note NOW (which import, why), since that would fail the same stage; merge their mend when it lands.

