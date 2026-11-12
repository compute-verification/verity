"""Strict TOML decoding into frozen dataclasses."""

from __future__ import annotations

import types
import tomllib
from collections.abc import Mapping
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

T = TypeVar("T")


class ConfigError(ValueError):
    """A configuration file does not have the declared shape."""


def _name(path: str, message: str) -> ConfigError:
    return ConfigError(f"{path}: {message}")


def _decode(annotation: Any, raw: Any, path: str) -> Any:
    if annotation is Any:
        if isinstance(raw, dict):
            if any(not isinstance(key, str) for key in raw):
                raise _name(path, "object keys must be strings")
            return {key: _decode(Any, child, f"{path}.{key}") for key, child in raw.items()}
        if isinstance(raw, list):
            return tuple(_decode(Any, child, f"{path}[{i}]") for i, child in enumerate(raw))
        if raw is None or isinstance(raw, (bool, int, float, str)):
            return raw
        raise _name(path, f"unsupported value {type(raw).__name__}")

    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (types.UnionType, Union):
        if raw is None and type(None) in args:
            return None
        errors = []
        for option in (arg for arg in args if arg is not type(None)):
            try:
                return _decode(option, raw, path)
            except ConfigError as error:
                errors.append(str(error))
        raise _name(path, "does not match any allowed type" + (f" ({'; '.join(errors)})" if errors else ""))
    if origin in (tuple, list):
        if not isinstance(raw, list):
            raise _name(path, "must be an array")
        item = args[0] if args else Any
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode(item, child, f"{path}[{i}]") for i, child in enumerate(raw))
        if args and len(raw) != len(args):
            raise _name(path, f"must contain {len(args)} entries")
        return tuple(_decode(a, child, f"{path}[{i}]") for i, (a, child) in enumerate(zip(args, raw, strict=True)))
    if origin in (dict, Mapping) or origin is not None and issubclass(origin, Mapping):
        if not isinstance(raw, dict):
            raise _name(path, "must be a table")
        key_type, value_type = args or (str, Any)
        if key_type is not str:
            raise _name(path, "only string-keyed tables are supported")
        return {key: _decode(value_type, child, f"{path}.{key}") for key, child in raw.items()}
    if is_dataclass(annotation):
        return from_mapping(annotation, raw, path=path)
    if annotation is bool:
        if type(raw) is not bool:
            raise _name(path, "must be a boolean")
        return raw
    if annotation is int:
        if type(raw) is not int:
            raise _name(path, "must be an integer")
        return raw
    if annotation is float:
        if type(raw) not in (int, float):
            raise _name(path, "must be a number")
        return float(raw)
    if annotation is str:
        if not isinstance(raw, str):
            raise _name(path, "must be a string")
        return raw
    raise _name(path, f"unsupported declared type {annotation!r}")


def from_mapping(cls: type[T], raw: Any, *, path: str = "$") -> T:
    """Decode one table into ``cls`` and refuse every undeclared key."""

    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    if not isinstance(raw, dict):
        raise _name(path, "must be a table")
    declared = {field.name: field for field in fields(cls)}
    unknown = sorted(set(raw) - set(declared))
    if unknown:
        raise _name(path, f"unknown key(s): {', '.join(unknown)}")
    hints = get_type_hints(cls)
    values: dict[str, Any] = {}
    for name, field in declared.items():
        if name in raw:
            values[name] = _decode(hints[name], raw[name], f"{path}.{name}")
        elif field.default is MISSING and field.default_factory is MISSING:
            raise _name(path, f"missing required key {name!r}")
    return cls(**values)


def load(path: str | Path, cls: type[T]) -> T:
    """Read a UTF-8 TOML document and strictly decode it as ``cls``."""

    source = Path(path)
    try:
        raw = tomllib.loads(source.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"{source}: {error}") from error
    return from_mapping(cls, raw, path=str(source))
