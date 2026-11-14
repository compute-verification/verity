[2026-11-14T19:36Z] 671da58188b4
LEAD: your test_no_verifier_imports.py is red on the tip in its stale-entry half: compute-accounting cut the adapter imports in protocols/pouw/__init__.py down to the serving package names, so KNOWN lists entries nothing imports. Shrink KNOWN to the tip (that is the test working), in the same push as the dead-modules mend.

