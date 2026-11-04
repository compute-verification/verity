"""A Pearl-C served run's document (`verity-pouw/pearl-c-served-passes/v1`, ``served.json``): its passes and its coins, which
``served_passes.py serve`` and serve-openai's Pearl+ arm (`pipeline.serve_pearl`) write and ``served_passes.py retain``
reads to write the window record W (`protocols.pouw.pearl_c.window`).  No stage of the integration reads it back, so it
has no type here; this module owns its schema id."""

SCHEMA = "verity-pouw/pearl-c-served-passes/v1"
