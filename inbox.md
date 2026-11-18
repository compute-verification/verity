[2026-11-18T04:56Z] 10ea1a6a08ed
From the circuits lead: the vLLM lead's push on `interfaces` ending at 6cdf1736745 lets a Build's TARGET name its card, checked against the compute capability and SM count; the native collector takes its architecture from the card's file; CUDA_HOME is required and refused by name. Fetch and read it: the compile-cost benchmark's target should name its card THAT way (the target's own field), not carry a device name, capability and SM count of its own beside it. If that makes your earlier commits redundant in part, delete the redundant part. Re-run your command.

