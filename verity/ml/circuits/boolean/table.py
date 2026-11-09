"""A table read on bits: the entry of a table the program brings, selected by an index, as a multiplexer tree.

A table of 2^IB entries of EW 16-bit words is a value: an operand of rows of `ROW_WORDS` words, an entry's low word first, which
the read takes whole beside its index. ``read(C, idx, words, IB, EW)`` is the entry at idx's low IB bits, its words joined low
first: `gather` over the 2^IB entries (16 EW bits each) by those bits. Every bit of the table is an operand, never a constant,
so a read is (2^IB - 1) * 16 EW ANDs whatever the table holds (8,160 for 2^8 entries of two words), each an AND of an index bit
and an XOR of candidates: the circuit's own rows, with no lookup.

    TableRead{IB}x{EW}_v2   (index: Array<IW, Value<1>>, table: Array<2^IB EW / 64, Array<64, Array<16, Value<1>>>>)
                            -> Array<16 EW, Value<1>>; word view: TableRead{IB}x{EW}_v1, the word read it is made from

`table_read(word, IB)` is the Boolean form of a word read ``word`` (`verity.ml.definitions.registry.tables.table_read`, which makes
both): its signature on bits, the next version of its name, and ``word`` as its word. Flock lowers the word read by this
form's builder (`verity_flock.ir_lower._boolean_build`). A table of more than `MAX_WORDS` words has no form, so a read of it
stays a `NoPiece`.
"""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition
from verity.ml.circuits.types import Array, Value

from .gather import gather
from .trace import definition

__all__ = ["MAX_WORDS", "ROW_WORDS", "WORD_BITS", "read", "table_read"]

ROW_WORDS, WORD_BITS = 64, 16
#: the most table words a Boolean form selects among (65,520 ANDs a read). A registered form is traced whenever circuit-check's
#: catalog meets it, and past this the trace alone is too slow for `check`: 2^16 words (`gelu_tanh_bf16`) take 43 s, and 2^28
#: (`tanh_rn`, 4,294,967,264 ANDs) never finish
MAX_WORDS = 1 << 12

_FORMS: dict[str, CompositeDefinition] = {}


def read(C, idx, words, index_bits: int, entry_words: int):
    """The entry at ``idx``'s low ``index_bits`` bits (LSB first) of the table ``words``: 2^IB entries of ``entry_words`` 16-bit
    words, entry e's word k at ``e * EW + k``; its words joined low first."""
    n = 1 << index_bits
    if len(words) != n * entry_words:
        raise ValueError(f"2^{index_bits} entries of {entry_words} words are {n * entry_words} words, not {len(words)}")
    if len(idx) < index_bits:
        raise ValueError(f"a {len(idx)}-bit index has no {index_bits} low bits")
    entries = [[b for k in range(entry_words) for b in words[e * entry_words + k]] for e in range(n)]
    return gather(C, list(idx[:index_bits]), entries, 0)


def table_read(word: PrimitiveDefinition, index_bits: int) -> CompositeDefinition:
    """The Boolean form of the word read ``word`` (``(index: Value<IW>, table: Array<R, Array<64, Value<16>>>) -> Value<16 EW>``,
    reading its table at the index's low ``index_bits`` bits): ``{word.name}_v{word.version + 1}``, `read` on its signature in
    bits."""
    d = _FORMS.get(word.id)
    if d is not None:
        return d
    (iname, it), (tname, tt) = word.params
    ew, rem = divmod(word.ret.w, WORD_BITS)
    rows = (ew << index_bits) // ROW_WORDS
    if rem or not ew or not isinstance(it, Value) or not 0 < index_bits <= it.w or rows * ROW_WORDS != ew << index_bits \
            or tt != Array(rows, Array(ROW_WORDS, Value(WORD_BITS))):
        raise ValueError(f"{word.id}: not a read of 2^{index_bits} entries of 16-bit words in rows of {ROW_WORDS} by a wider index")
    if ew << index_bits > MAX_WORDS:
        raise ValueError(f"{word.id}: a table of {ew << index_bits} words has no Boolean form, over MAX_WORDS = {MAX_WORDS}")

    def build(C, idx, *words, _ib=index_bits, _ew=ew):
        return read(C, idx, words, _ib, _ew)

    d = _FORMS[word.id] = definition(
        word.name, word.version + 1, build, ((iname, BL.bit_type(it)), (tname, BL.bit_type(tt))), BL.bit_type(word.ret),
        word=word, doc=f"{word.id} on bits: the entry at the index's low {index_bits} bits, {ew} words joined low first; the "
        f"multiplexer tree over its 2^{index_bits} entries, {((1 << index_bits) - 1) * WORD_BITS * ew} ANDs")
    return d
