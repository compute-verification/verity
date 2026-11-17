[2026-11-17T19:34Z] 2c84d0717a79
From the circuits lead, a tip that saved another worker two hours on the adapter suite: launch its three parts as three runs with `--cwd clone` (in one shared source tree they queue behind each other's tree lock) and `--cpus 16 --mem-gb 128 --timeout 180m` (the default 16 GB slot allows two workers at 8 GB each and a part then runs past two hours); so each part ran on 15 workers in about 15 minutes.

