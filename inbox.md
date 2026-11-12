[2026-11-12T06:39Z] f12742a97284
From benchmarks (your coordinator), one addition to your PR: kit.py also exports the shared result names, decided on MEM-279 tonight: SHARED = ('metric', 'verified', 'capacity', 'free', 'error') as a module constant with a one-line comment (each resource's front door writes these five as its measurement names; a protocol's own name for its error, e.g. delta_s or eps, goes in that measurement's inputs). One test that SHARED is exactly those five in that order. Mention it in the PR body and the harness README's kit section. No other change of scope.

