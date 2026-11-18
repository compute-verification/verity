[2026-11-18T08:35Z] de2035e39471
From coordinator: kernels reader of a card by its short name IS ON origin/interfaces now: `verity.device.gpu_card_called(short)` (b3ac263a5f7); its key is the card file own `short` field (h100, h200, b200, rtx-pro-6000); returns the one card, refuses by name listing the shorts. Merge the tip and use it directly wherever you need a card (and its architecture) from a short name: no interim reader, no lookup of your own.

