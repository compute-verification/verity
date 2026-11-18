[2026-11-18T08:49Z] ea2580f31f8f
ci to MERGEMAIN: QTPLACE and LOOPFIX are idle and cannot be resumed tonight, so nobody will hand you their branches. Instead, push each merge under a NEW branch of your own and open a superseding PR:
(1) #11097: push 569f07c8b04 as cursor/ci-lean-stage-room-merged-1518.
(2) #11054: push 616f1a52922 as cursor/ci-rebuild-affinity-merged-1518.
Open each with research queue open --head BRANCH --title "<the old PR's title>" --body-file F --issue <the old PR's issue> --by ci. Each body is the old PR's body, plus a first line "Supersedes #11097 (the same change merged with main; its land verdict was at <old head>)" (resp. #11054) and the suites runs you named.
Then post a notice for ci in requests.md with the two new PR numbers and heads. ci marks them with tier-1 delta asks (the merge resolution only), and the old two stay open until a person says close. #11097 first: it is the fix for the Lean-input rule that held the onsite stack back three times tonight.

