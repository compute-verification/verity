Part of the certifier's [PROTOCOL.md](../PROTOCOL.md).

## The compact finish (`active.compact_finish_violations`, `certifier_serve.certified_io`)

`COMPACT_LAST` and `COMPACT_STOP` are flags of well-formed compact records, and the certifier commits whatever ids and
finish the server writes: one of three states (none, `length`, `stop`) at each response's end, where an id event marks
the end out of band. A stop carries the EOS id the model sampled, so where a response stops can be checked as an id
event's can.

- **The certified I/O.** The rehearsal's certified I/O (`certifier_serve.certified_io`) refuses a mark before a
  response's last record, a stop on an id that isn't one of the model's EOS ids, an EOS id in any record but the stop,
  and `COMPACT_LAST` alone short of `max_tokens`. Like an EOS id event, a stop on an EOS id that the model didn't sample
  there passes it.
- **The I/O tie.** So the check that refuses a wrong length or finish is the I/O tie's, required of compact framing,
  against the Commit's sampled ids: ordinal o's cells are `compact_records_of` o's sampled ids, one for one, so each
  carries its sampled id, the response is as long as the Commit's, it stops only where the Commit sampled EOS, and its
  `COMPACT_LAST` alone is at `max_tokens` exactly (`compact_finish_violations`). The I/O tie's egress tie holds each compact
  cell's token field to this rule as well as to its id.
- **A client's abort.** A response its client aborted ends on an unmarked record. Both checks accept that end only
  where the Commit's finish for the request is an abort (`aborted`), after 1 ≤ k < `max_tokens` outputs, none of them
  EOS, and refuse a marked end where the Commit's finish is an abort. The rehearsal's own `check` reads no Commit, so it
  refuses every unmarked end. The link is one connection with no per-request cancel, so the certifier observes no
  abort, and an abort on a certified run comes from the prover's side (the shim's cut at the horizon, or an error).
  What bounds it is the advice charging each `output_len` in full, not an observation.
- **Why a stop carries its id.** A stop that was one reserved field (all ones) with no id, accepted at any length up
  to `max_tokens`, would let a server cut a `length` response short at a stop the model never sampled; the all-ones
  field is a stop on id 2^30 − 1, which isn't EOS. For the same reason the tie refuses a `COMPACT_LAST` short of
  `max_tokens`, even when the records are the Commit's ids.
