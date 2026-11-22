"""Each application's list attaches in its own order (`verity.protocols.accounting.scheme.order_refusals`)."""
from __future__ import annotations

from pathlib import Path

import pytest

from verity import config as K
from verity.channel import Refused
from verity.protocols.run import schemes
from verity.protocols.accounting import scheme as S
from verity_cli import run as R

ROOT = Path(R.ROOT)
REG = R.registry()
APPLICATIONS = ("onsite_full", "remote")


def _raw(name: str) -> dict:
    return K.parse((ROOT / "applications" / f"{name}.toml").read_text())


def _declared(names: list[str]) -> list[S.Scheme]:
    """Each name's declaration; ``pouw``, which no scheme declares yet, as useful work with its component's requires."""
    work = S.Scheme("pouw", "verity/pouw/stand-in", "compute", (S.Block("Pouw_v1", lambda shape: 1),), (), (), (),
                    depends=REG.components["pouw"].requires)
    return [schemes.get(n) if n in schemes.DECLARATIONS else work for n in names]


@pytest.mark.parametrize("name", APPLICATIONS)
def test_an_applications_list_attaches_in_its_own_order(name: str) -> None:
    assert S.order_refusals(_declared(_raw(name)["protocols"])) == []


def test_onsite_lists_storage_before_useful_work() -> None:
    names = _raw("onsite_full")["protocols"]
    assert names.index("pous") < names.index("pearl-plus")
    names.remove("pous")
    names.insert(names.index("pearl-plus") + 1, "pous")
    assert S.order_refusals(_declared(names)) == [
        "outer: pous is storage (it needs the workload's store) and is listed after pearl-plus, useful work "
        "(resource compute); where both wrap one site the earlier listed is the outer, so protocols lists pous before "
        "pearl-plus"]


def test_a_dependence_listed_late_is_refused_by_the_name_and_words_config_check_gives() -> None:
    raw = _raw("onsite_full")
    names = raw["protocols"]
    names.remove("bandwidth")
    names.insert(names.index("pous"), "bandwidth")
    with pytest.raises(Refused) as checked:
        K.check({**raw, "protocols": names}, REG.components, REG.workloads)
    assert checked.value.name == "order"
    assert S.order_refusals(_declared(names)) == [str(checked.value)]
