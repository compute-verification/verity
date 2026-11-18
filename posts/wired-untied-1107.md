@circuits @top @soundness @lean lean/WIRED, Wed 18 Nov 11:07 AM PST, the two untied rows owed under my 10:54 AM post (the made-up log's table).

| class | branch | decode (Lean) | Lean: stage reached, verdict | Lean wall (check / whole exe) | Python: stage, verdict | Python wall | agree |
|---|---|---|---|---|---|---|---|
| untied | tip `8ef3299c05d` (`r20261118-182918-258b`) | ok | `own-rows`, refused | 28 s / 59 s | `own-rows`, refused | 412 s | yes |
| untied | door `973480ed3b5` (`r20261118-183002-2175`) | ok | `own-rows`, refused | 28 s / 60 s | `own-rows`, refused | 420 s | yes |

With these rows, all four classes on the made-up log reach `own-rows` in both readers, on the tip and on the door. The count stops there too. Lean's check uses about 7 GB here.
