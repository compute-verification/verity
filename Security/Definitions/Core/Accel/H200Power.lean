import Definitions.Core.Accel.Power

/-!
# The H200 SXM's energy floors (trusted)

Each row is `verity/hardware/cards/h200-sxm-141gb.toml`'s, under `energy`, with the readings it rests on there (two or
three a row); `verity/hardware/tests/test_lean.py` holds this table to that file number by number.

The card is `H200.card`'s, and each row is against its in-kernel SM clock in Hz: 1830 MHz at most while its tensor pipe
runs and 1980 MHz for ALU or hash work alone. The readings are the card's own power over a steady window at a locked
clock, of one kernel on every SM: resident and idle (`static`), `s8` `wgmma` on zero operands (`mac`) and on uniformly
random ones (`macDecl`), and SHA-512 compressions (`comp`). A row's clock is the highest in-kernel clock of the readings
in its band, rounded up. A static row is the least idle reading at or above its band; a class row is the least, over its
band's readings and the rows above it, of the reading's power less the static row at its clock, over its rate. Each is
rounded down. They were read at 33 to 69 °C, on `s8` only for the tensor unit, and only at the clocks the rows name.
-/

open scoped NNRat

namespace Verity.Accel.H200

/-- **The H200 SXM's energy floors.** -/
def power : Power where
  static := [
    (9039 * 10 ^ 5, 8078 / 100),
    (12000 * 10 ^ 5, 8726 / 100),
    (15000 * 10 ^ 5, 10272 / 100),
    (17550 * 10 ^ 5, 11743 / 100),
    (18300 * 10 ^ 5, 12017 / 100)]
  mac := [
    (9039 * 10 ^ 5, 8019 / 10 ^ 17),
    (18300 * 10 ^ 5, 15495 / 10 ^ 17)]
  macDecl := [
    (9039 * 10 ^ 5, 30803 / 10 ^ 17),
    (17240 * 10 ^ 5, 62979 / 10 ^ 17)]
  comp := [
    (9039 * 10 ^ 5, 250056 / 10 ^ 13),
    (12000 * 10 ^ 5, 295788 / 10 ^ 13),
    (15000 * 10 ^ 5, 378814 / 10 ^ 13),
    (17550 * 10 ^ 5, 501343 / 10 ^ 13),
    (18300 * 10 ^ 5, 520811 / 10 ^ 13),
    (19800 * 10 ^ 5, 520811 / 10 ^ 13)]

end Verity.Accel.H200
