"""A put's read-back: the panel the route serves against the body sent, compared as numbers. The route writes a float
with no fractional part as an integer (`6.52837e+20` comes back `652837000000000000000`), which Python's `==` holds
unequal to the float it came from; as numbers, the double each parses to, they are one value."""


def same(a, b) -> bool:
    """`a` and `b` are one value: a number as the double it parses to, a list or a dict element by element and key for
    key, anything else (a string, a bool, null) by type and `==`."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def differ(sent: dict, back: dict) -> list[str]:
    """The keys whose values in the body `sent` and the panel `back` aren't `same` (one missing on either side counts),
    sorted."""
    return sorted(k for k in {*sent, *back} if k not in sent or k not in back or not same(sent[k], back[k]))
