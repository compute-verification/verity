"""verity_numerical.bench: the campaign's result conventions, shared by every proof backend.

``contract`` says what a campaign number must say about itself (fingerprint fields, measurement names, the six timing
buckets) and validates a ``research`` ``result.json`` against it.  ``research`` itself never learns any of this: it
records and ships an opaque workload; the meaning of the result is decided here, on the workload side.
"""
