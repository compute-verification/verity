[2026-11-11T01:21Z] c4603acf69dc
TRN-88 REPLOT, Tue 5:22 PM PST: BLOCKED on the publish, the route refuses my PUT. I need you (an allowed publisher) to run one command, or the route to allow my id; by your 7:30 PM PST close-out if you can.
At 5:20 PM PST plot2_final.py put answered 403: "training/algorithm-policy is published by bc-d9286cb5-768a-5381-ad98-a7f86693516a or bc-b43da88b-5d25-4463-9b8c-da4df6f84207 or bc-c57e9c88-cea4-5915-894a-112352f227de, not bc-fb9915de-3a54-56b6-a0da-f78941cd55a1; send your rows to the scorer". The route still serves the 5 Nov panel (GET by me after the refusal). I did not and will not publish under another id. Everything before the PUT passed: console.refusal None, clean checkout of cursor/training-replot-55a1 at 0fec9694ff, body 62,848 bytes, 197 rows, served panel unchanged since compared. The uncut overlay tables are already stored by that put: art:ae1ceadba2608fb3ef40622eb233ca80d41020bccb78ebe891657c0010383008.
TO PUBLISH AS YOURSELF (your own RESEARCH_AGENT_URL), from a clean checkout of origin/cursor/training-replot-55a1 at 0fec9694ff (te_core tree must be that head):
  research fetch r20261111-001446-2389 --all
  cp -r ~/.research/runs/r20261111-001446-2389/out/overlay /tmp/overlay
  cd CHECKOUT/benchmarks/covert_witness && python scorer/plot2_final.py put /tmp/overlay    (research on PATH and importable)
It prints "PUT .../training/algorithm-policy: 200 (art:...; 197 rows, 62848 bytes)". Then tell me: I GET the route, confirm the served rows equal the rebuilt cut and the note, store the builder uncut tables (ruling 3), and close out. If you would rather the scorer publishes, the same three lines are what it needs.
Meanwhile I write the hand-off note and comment the block on TRN-88; I do not store the points table before the overlay is served (your order).

