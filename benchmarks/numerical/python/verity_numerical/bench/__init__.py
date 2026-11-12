"""verity_numerical.bench: what the proof backends' measurements share: the result contract, the census, input sets, a measured
cell and the per-unit-type cost inventory.

``contract`` says what a measured number must say about itself (fingerprint fields, measurement names, the six timing
buckets) and validates a ``research`` ``result.json`` against it.  ``research`` itself never learns any of this: it
records and ships an opaque workload; the meaning of the result is decided here, on the workload side.
"""
