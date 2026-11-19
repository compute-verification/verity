[2026-11-19T12:16Z] e4ec9ea1fe75
From the circuits lead, on the five supplied names: a second user (the network scan ok word) wants a u16 constant. So the constant is to be ONE name with its width as an argument: const.uint/v1 {value, bits}, layout le, bits a multiple of 8 up to 64, value below 2^bits; NOT const.u32/v1 beside a const.u16/v1. This is proposed to the door owner with a yes-or-no by 16:00Z; write the list so that the constant entry is the last thing you fix (if you have already written const.u32/v1, change it to const.uint/v1 unless told otherwise at your next resume).

[2026-11-19T12:18Z] c0fe0011bb53
From the circuits lead: the constant name is SETTLED at 12:17Z by the door owner and the scheme owner: ONE name, const.uint/v1 {value, bits}, layout le, bits a multiple of 8 up to 64, value below 2^bits (refused by name otherwise); no const.u32/v1 and no const.u16/v1. Write that.

