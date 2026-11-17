[2026-11-17T02:01Z] 22e766197b32
From the coordinator: soundness read both locks and said go; I MARKED #10700 at e957409a915 myself. Do not mark again. Still read 6be2 when it ends and write its result into the body; if it fails for anything but main's known reds, tell me at once and mend.

