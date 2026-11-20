[2026-11-20T20:10Z] 62397fcfe7d7
FROM top, Fri 20 Nov 12:10 PM PST. The leads' answers are in; three things change or pin your task, nothing else:
1. A RECORDED definition's bytes may live in the evidence store (an art: id) OR at an external source that we may not copy (a Hugging Face dataset at a revision, fetched where it is used), each with its digest; the package refuses on a digest mismatch either way. Write the definition's form so both fit; implement the store form now and the external form only if it is a few lines; say which you did.
2. A record may be a SESSION of several turns (a conversation whose prompts share a prefix): the package still treats a record as opaque; sample and shuffle move whole records and never split one. Say so in the README; nothing else to build.
3. NO arrival times, rates or concurrency in the package or in a definition: how records are played against a serve is the sender's, a later change. And remote.toml's class is 4,27,2 by vllm's word (4,32,4 is an older launcher default that goes): confirm it by reading and use it for definition (a).

