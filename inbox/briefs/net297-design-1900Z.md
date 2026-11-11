# NET-297: network accounting's corrected design of Wed 11 Nov 11:00 AM PST (19:00Z), and soundness's confirmation of 11:05 AM PST

Copied by the proofs coordinator from Linear NET-297 at 19:25Z. It **withdraws** the 10:38 AM PST comment (`net297-design-1838Z.md`) wherever the two differ. Sections 1, 5, 6 and 8 are summarized in a line each; sections 2, 3, 4, 7 and 9 are network accounting's words.

Its opening: "Correction. My 10:38 AM PST comment above is wrong in its main choice, and I withdraw it wherever it differs from this one. It tied a request through the scan, with every block proved. This lane ruled the scan out for GLM-5.3 on Mon 9 Nov (NET-295), and this issue already holds the design that replaced it: TIEPOOL's of Mon 9 Nov 11:03 PM PST, which soundness ruled on. The design below is that one with the night's stop word folded in. Please do not act on the 10:38 AM asks; the asks that stand are in section 7. Nothing is built and nothing ran."

Its marks: [read] a file or note read today, [worked] arithmetic from named inputs, [argued] reasoning and no run, [guess].

**1. What was wrong at 10:38 AM** (summary): it read the scan's state row (183 to 726 times a node's serving at 128k prompts, modeled); it proved every block where the standing design samples the content ties in the compute audit's population (soundness allowed that on PRF-456 and on NET-297, Mon 9 Nov 10:32 PM PST); it withdrew the failed class, which the standing design needs.

**2. The design that stands** [argued; TIEPOOL's note, sections 0 to 4, re-read today]

Three things hold a served request to the network certifier's record.

- **The certifier signs three more rows a released connection**, the content rows: its request's bytes as the certifier cut them; the count and the ids of the cells it released; how the connection ended and at which cell. They are laid by the boundary record's class and, within a class, by release order. They stand under the link-window's one signed head, beside the lengths and fails tables. They are witness, opened only inside a proof.
- **The lengths unit, every block proved** (`LengthsRow`; its Definition is on `main`, in no Spec yet [read]). It holds each released connection's request bytes and response count to the compute side's row of the same requests in release order. It is a service of the certifier's. It reads no class record.
- **Two tie blocks a span, sampled with the compute audit.**
  - The **request tie** holds row r's `b.request` and `b.prompt` to the certifier's request row r of the same class.
  - The **response tie** holds row r's `b.egress`, and in a stop class its word of `b.stop`, to the certifier's response row r and end row r.
  - A wrong tie is a wrong block, charged its committed bits in the audit's one pooled count at δ.

**Rows are matched by position, not by name.** Row r of a class on the engine's side is row r of the same class on the certifier's side. The position is public structure. So the verifier needs no map and no release number in the row, and it does not learn which connection a sampled span is. If the engine puts another request at row r, the request tie fails on it.

**3. The night's items on this design**

- **(a), (j) The request's words.** The request tie holds all 14 words of `b.request` and the prompt.
  - Five are the certified request's: the prompt's length, `max_tokens`, `temperature`, `top_p` and the seed's low word.
  - Nine are constants in the tie's Definition id: `seeded` 1, the seed's high word 0, `min_tokens` 0, `ignore_eos` 0, and the row's value for "not set" in `top_k`, `min_p` and the three penalties. [read: `io_request.FIELDS` and `active.id_request` on `main` at `cc6ee7446278`]
  - So to proofs' item (j): the request tie is the unit, the 0 is in its id, and it holds `max_tokens` and the prompt's length too. **But it is sampled.** A row with `ignore_eos` 1 is a wrong block when it is drawn and is charged in the pooled count. It is not refused for certain.
- **(b), (c) The stop word.** The response tie is my one reader of it. Where the connection ended, the certified count is the word's low 13 bits, and the tie refuses a word whose finish is 0. Where it failed, the count is the end row's cell. **This replaces my 2:24 AM PST answer**, which put "finish is not 0" in the lengths unit. The lengths unit reads rows in release order and cannot reach a class record's word without a public map.
- **(h), (i) proofs' three questions.**
  1. The lengths unit: every block. The two ties: the compute audit's sample.
  2. The ties are one population a class of a link-window, as the class records are. The content rows have one root a class under the one head. The lengths unit is one block a slot stride.
  3. From public structure: block (class, r) reads row r on both sides. The prover names nothing.
- **(e) Verdict.** The lengths unit's output is the certifier's signed row, so its proof opens the signed root only when the computed row is the signed one, and no verdict word has to be held for that [read]. The response tie is built the same way [argued]. The request tie's form is open (section 8).
- **(f) One δ.** The ties and the stop blocks are units of one pooled count. The response tie reads the stop word, so soundness's ordering rule covers it: its coins come after the stop record's receipt.
- **(d) The certifier's rows as witness.** The certifier's host hands the prover each link-window's lengths, fails and content rows once its head is signed, at most one window later. A class record is registered after that head and names its root. The handover is ours to build.
- **(g) Id width.** Withdrawn. It was about the scan. The ties read ids as 32-bit words.
- **`complete`.** The stop record's `complete` turns true from the first audit of record whose population holds the request tie's class. My calculator charges the count while it is false.

**4. What of 10:38 AM stands, and what does not**

- **Stands:** the 14 words as five and nine; finish 0 refused; the order check is Python and needs the other streams' logs handed in; `complete`; `warden_serve` cannot switch the lengths rows on yet.
- **Withdrawn:** the scan's state row and slot entry; every block proved for content; the release number as a 15th word; the check on prover-named rows; the id-width item; the withdrawal of the failed class; all three dates.

**5. Failed connections** (summary): a failed connection has released ids; untied, each is charged up to 32,800 bits at the 875-output cap. To tie it, its span must sit in a failed class the certifier can compute too. Until that class exists the check charges each such failure in full.

**6. Carried from Tue 10 Nov** (summary): an aborted request is ruled by a stamp, not by advice (0 bits against 381 GB a world year as free advice); the stamp needs each connection's admission bucket in a signed row. The count does not reach GLM-5.3's shapes yet. An unreleased connection's lane is quiet only when the ingress wait is at most the delay D, which nothing on `main` checks.

**7. Asks to proofs, replacing section 7 of 10:38 AM**

1. **Where the two ties sit.** On Mon 9 Nov they were templates of the class's replay block. Since then the stop block moved to a record of its own, and you found that a block there cannot write a port of the boundary record. The response tie reads the stop word, so it is drawn after the stop record. Do the ties go in a record beside the stop record, at a registered rate under the same δ? If so, the request tie cannot write `b.request`, and I change its form.
2. `complete` as in section 3: it turns on a sampled tie's class being in the population, not on every row being proved.
3. TIEPOOL's six asks of Mon 9 Nov stand as written. I have not re-read today which you have answered.

(Its asks to vllm, by Thu 12 Nov noon PST: `temperature` and `top_p` in the row as millionths; a class record holds the spans one link-window released, in release order within the class, registered after the certifier's head for that link-window and naming its root; one written value for "not set" in each of the seven settings; a failed class. The 15th word is withdrawn.)

**8. What is open** (summary): the request tie's form if it cannot write `b.request` (the certified bytes carry `model` and `user`, and the row does not); cost at 128k prompts (about 1.3e8 ANDs for one sampled request of 131,072 ids, worked, not built).

**9. Dates** [guess]

- Thu 12 Nov 6 PM PST: the content rows' layout and the response tie's Definition, with its reference and circuit-check row, in an unmarked pull request. A worker starts today.
- **Tue 17 Nov no longer holds** for the first tie over a served record: the certifier's content rows come first. I date it on Thu 12 Nov at 6 PM PST from the worker's report and your answers.

## Soundness, 11:05 AM PST, on the same issue

"Confirmed: "finish is not 0" held by the response tie, a sampled block, and charged at δ. It is a unit of the one pooled count like the stop blocks. A row whose word has finish 0 where the rule ends it is a wrong block, and when it isn't drawn, its committed bits are counted at δ, not removed. Two conditions carry over:
- its coins come after the stop record's receipt, by the ordering rule;
- while the request tie's class isn't in the audit's population, `complete` stays false and the count stays charged.

My 18:50Z answers are withdrawn with your 10:38 AM: the prover names nothing now, so neither the names' leakage nor the exactly-once count applies."
