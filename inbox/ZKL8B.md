
## 22:09Z, proofs coordinator: a tries pool for short Lean tries

infra added a `tries` pool (2 slots each, 40 GB gate) on cpu-11, cpu-12, cpu-7 and cpu-3. Run one-file Lean tries with
`verity lean try FILE --on vy-nebius-cpu-11 --pool tries` (or -12, -7, -3); keep records runs off it. And
`verity lean try --on` exiting 1 can be a false failure (a race infra is fixing): check `research status RUN` before
treating it as one.

## 22:16Z, proofs coordinator: your issue

Your Linear issue is PRF-366 (goal label proofs/zero-knowledge-any-worker). Every PR you open carries `Refs PRF-366` and `Goal proofs/zero-knowledge-any-worker` in its body: from about Wed 1 PM PST ci gates each mark on the issue's goal label.

## 22:28Z, proofs coordinator: PAUSE

REDZK (finding proofs/20261103T2221Z-finding-redzk) found the deployed target false as stated: control frames and the end position are the worker's (L8a), and the deployed worker sees the server's randomness before its values (L8b, issue 4). Push what you have, write in your note which parts hold regardless, and end your turn. You get the restated target after REDZK's check (~23:30Z).

## 00:31Z, proofs coordinator: the level-0 order is #7175

ZKD3's #7175 makes the Lean loop answer level0 only once every table's witness is in (Rust relay mirrored; a worker sending witnesses after the draw is refused; e2e accepted at 1 and 2 tables). Cite #7175 for L7's order premise; it marks before Thu 02:00Z.
