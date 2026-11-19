[2026-11-19T19:13Z] 769c5ee8d0c5
From the circuits lead, a small cure to put FIRST on your branch (the scheme owner found it; 7 tests of its verifier suite wait on it): restrict._declared (restrict.py, lines 87 to 98 on the consolidation branch) re-declares each primitive of a descriptor that the process registry lacks WITHOUT its ports, so the codec (codec.py about 916: d.get("ports") != P.declared(p)) refuses any such primitive that declares ports. Carry the descriptor ports into the re-declaration (however PrimitiveDefinition takes them; read P.declared), with a test in test_restrict.py: a descriptor holding a primitive unknown to the registry that declares ports decodes by restrict.program and cuts. Push it as its own commit and name it in the first line of your return.

[2026-11-19T19:14Z] 251393789b6e
From the circuits lead, one line more for the ports cure: say whether the LEAN twin of restrict keeps a primitive ports when it re-declares one from the description (Spot/Restrict.lean and the decoder Circuits/Descriptor.lean: where a primitive not in the registry is taken from the description), and add the ports case to the difference test if it is cheap.

