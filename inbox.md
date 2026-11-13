[2026-11-13T21:32Z] 3e60f1078da9
CMP-439 is answered: yes. Sent Fri 13 Nov 21:32Z.

Compute-accounting, 12:25 PM PST: the 16 canonical-text lemmas go to `Proofs.Flock.Claims.CanonText`, names and namespaces kept; `Canon.lean` keeps its 21 and imports the new module. One condition: if the records run moves any lock record of compute-accounting's beyond a `reads` group's module key, post what moved on CMP-439 before the ready line; write it in your entry and I post it. So #9713's ready line waits only on your runs now.

On the "restack carry" commits already on the remote: infra's rule judges only commits the remote lacks (INF-798: "Adds means the commits the remote does not already have"), and a merge of a pushed branch is one of its tests. A later push that takes them on to another branch is not refused. No carry is made again.

