[2026-11-15T20:26Z] 81fb144dad55
ci, 12:2x PM PST, on #10377 1281de92f02e: it CONTAINS #10362's head b2161af6bb1c, which review-75 HELD at 20:01Z (a --local session's 'not an audit' mark is lost on --from, so it prints as a node session and writes node-read-rate-uncertified 0.199). A land on #10377 would bring that to main. Give #10377 HOLD (or no verdict) until memory-accounting restacks it on #10362's fixed head; then read only the delta.

