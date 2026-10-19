import Proofs.Accounting.Headline.Sm120
import Proofs.Accounting.Headline.Sm120Measured
import Models.Accounting.Guarantees.FpG1

/-!
# sm_120's headlines at FP8-g1 and W4A8-g1's windows

Each is the record's headline at g1's windows (`Models.Accounting.Guarantees.FpG1`): the record's statement quantifies
over every value type and proof layer, and its conclusion reads neither the relations nor the values, so the
hypothesis that the relations check g1's grids is not used.
-/

namespace Accounting.SecurityProofs.TS

theorem G1Sm120SheetHeadline : Accounting.Guarantees.TS.G1Sm120SheetHeadline := by
  intro Ω _ Pl G Uf V ι nπ pf ops δbind δzk _
  exact Accounting.SecurityProofs.TS.Sm120SheetHeadline Ω Pl G Uf pf δbind δzk

theorem G1Sm120MeasuredHeadline : Accounting.Guarantees.TS.G1Sm120MeasuredHeadline := by
  intro Ω _ Pl G Uf V ι nπ pf ops δbind δzk _
  exact Accounting.SecurityProofs.TS.Sm120MeasuredHeadline Ω Pl G Uf pf δbind δzk

end Accounting.SecurityProofs.TS
