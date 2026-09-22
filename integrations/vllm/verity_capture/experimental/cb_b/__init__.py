"""CB-B (R9): direct structural execution of the actual pinned vLLM model Python on the `meta` device, intercepting
dispatcher calls / custom ops / Triton launches with a lightweight observer, lowered to registered Definitions.

Experimental; never a construction input for accepted Programs.  See `out/gen/r9/cb-b/REPORT.md`.
"""
