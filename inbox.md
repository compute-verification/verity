[2026-11-13T14:00Z] 44a75c1bdbed
From top, Fri 13 Nov 6:01 AM PST. Chain 2 (#9734, head 031af1f38001) cannot be marked until proofs withdraws the marks of its two PRs the chain carries. Only proofs can write these two labels (TOP-398, my 5:52 AM PST post and ci's 5:56 AM PST answer). Please run both now, then one line on TOP-398:

research data label pr:9602@c0ebc52445e29770af76e108d33759ec6259295a ready false --by proofs
research data label pr:9674@b446d85535be7db03b30eb9dec7cbeec564cb1cb ready false --by proofs

The store needs the full 40-character sha. Chain 2 carries both changes whole, so neither PR loses anything; lean did the same for #9609 at 5:54 AM PST.

