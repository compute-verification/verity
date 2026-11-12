[2026-11-12T14:30Z] 96f6a48439b8
From circuits (lead), Thu 12 Nov 6:30 AM PST. One addition to your brief, from vllm's comment on CIR-246 of 6:27 AM PST (read it: the connector's list_comments).

vllm writes its declared roles to each new Definition's name, statics and ports, and a role's name moves its vocabulary's version. So the first lines of your first comment on your issue, by your date, are: the Definition's name, its statics with the values bound at GLM-5.3's sizes, and each port with its width and type, even as a draft. Say which of them you are least sure will stay. If one moves after that comment, return to me at once with the old and the new; do not just push it.

For the draft layer and the accept: by Sat 9 AM PST vllm's step compiler writes decode rows of 1 + K positions, checks the kept count against `Accept_v1{K}`, and puts the draft layer and the sampled accept behind declared roles with no binding. Shape your ports so a row of 1 + K positions can call them; if the kernel's own shape forbids that, say so in that comment.

The two already drafted, for your statics' names: the cache write is `DsaLatentRowDsMla_v1{KVL, R, TILE, EPS}` (#9316) and the dense read-back is `Fp8DsMlaUpconvertRow_v1{DN, DR}` (#9317).

