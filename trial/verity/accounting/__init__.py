"""Accounting: what happened, per resource (compute, memory, network). `capacity` is what the site could have supplied,
from a physical description of it; `observed` is how much the verifier's hardware at the site saw used, by whatever used
it; `explained` is how much the stream claims and verification backs. `coverage` composes them into the account, whose
headroom is where anything else could have run. Each statement over the stream names its digest, and each holds except
with the probability it states; none says whether what happened was allowed, which is `verity.compliance`'s."""
