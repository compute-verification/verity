"""A `registered_values` derivation's registered inputs: the `values` record, whose fields are allocated as the Definitions read
them, and the tables the rules declare (`Rule.registered_tables`).  They follow the user inputs and precede `weights`, which the
Build's consumers read as the last parameter."""

from __future__ import annotations

from verity.ml.circuits.refs import Affine, Coll
from verity.ml.circuits.types import Record, Type, Value

from verity_vllm.program.registry.tables import table_type


class Registration:
    def __init__(self, values_param: int, tables: dict[str, int]) -> None:
        self.values_param, self.tables = values_param, dict(tables)
        self.fields: list[dict] = []   # {name, width, bits, offset}, in first-read order

    @classmethod
    def declare(cls, params: list[tuple[str, Type]], ruleset, ep, mode: str) -> Registration:
        """Append `values` (filled by `finish`) and every declared table to `params`."""
        values_param, tables = len(params), {}
        params.append(("values", Record()))
        for r in ruleset.rules:
            for tn in (r.registered_tables(ep, mode) if r.modes is None or mode in r.modes else ()):
                if tn not in tables:
                    tables[tn] = len(params)
                    params.append((tn, table_type(tn)))
        return cls(values_param, tables)

    @property
    def names(self) -> tuple[str, ...]:
        return ("values", *self.tables)

    def value(self, name: str, width: int, bits: int, distinct: bool = False) -> Coll:
        """The field `name`, allocated at its first read; every read of one name is the same `width`-bit word.  `distinct`: a value
        of its own (a scalar of a generic kind) -- the field `name`, `name_1`, ... that holds these bits, in first-read order."""
        if distinct:
            same = [f for f in self.fields if f["name"] == name or f["name"].startswith(name + "_") and f["name"][len(name) + 1:].isdigit()]
            f = next((f for f in same if (f["width"], f["bits"]) == (width, bits)), None)
            name = f["name"] if f is not None else name if not same else f"{name}_{len(same)}"
        f = next((f for f in self.fields if f["name"] == name), None)
        if f is None:
            f = {"name": name, "width": width, "bits": bits, "offset": len(self.fields)}
            self.fields.append(f)
        elif (f["width"], f["bits"]) != (width, bits):
            raise ValueError(f"value {name!r} is {width}-bit 0x{bits:x} here and {f['width']}-bit 0x{f['bits']:x} at its first read")
        return Coll(Value(width), Affine("p", self.values_param, f["offset"], 1))

    def table(self, name: str) -> Coll:
        if name not in self.tables:
            raise ValueError(f"table {name!r} read but no rule declared it (`Rule.registered_tables`)")
        t = table_type(name)
        return Coll(t, Affine("p", self.tables[name], 0, t.leaves))

    def finish(self, params: list[tuple[str, Type]]) -> None:
        if not self.fields:
            raise ValueError("registered_values: no Definition read a program value; the `values` parameter would be empty")
        params[self.values_param] = ("values", Record(*[(f["name"], Value(f["width"])) for f in self.fields]))

    def layout(self) -> dict:
        """The registered inputs as the record states them (the fold's `FoldResult.supplied` layout): parameter index, name and every
        field with its leaf offset and bits (hex), and the tables."""
        return {"registration": {"param": self.values_param, "name": "values",
                                 "fields": [{"name": f["name"], "width": f["width"], "bits": f"0x{f['bits']:0{f['width'] // 4}x}",
                                             "offset": f["offset"]} for f in self.fields]},
                "run": [], **({"tables": [{"param": i, "name": n} for n, i in self.tables.items()]} if self.tables else {})}
