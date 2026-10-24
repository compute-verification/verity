"""Synthetic biographies whose information content is known exactly.

Each person has six attributes drawn independently and uniformly, so a population of N people carries
H = N * sum(log2 C_a) bits. Names are unique keys and carry nothing. Every attribute is rendered in four
biography paraphrases and three questions: two questions train, the third is held out for evaluation.
Documents are lists of (text, scored) segments; a scored segment's tokens are the only ones in the loss.
"""
import itertools
import math
import random

CITY = [a + b for a, b in itertools.product(
    ["Ash", "Bel", "Cor", "Dun", "El", "Fair", "Glen", "Hal", "Ival", "Kin",
     "Lan", "Mar", "Nor", "Oak", "Pen", "Ral", "Sel", "Tor", "Ul", "Wyn"],
    ["ford", "haven", "mouth", "bury", "field", "stead", "wick", "ton", "more", "dale"])]
YEAR = [str(1900 + i) for i in range(100)]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
BIRTHDAY = [f"{m} {d}" for m in MONTHS for d in range(1, 29)]
EMPLOYER = [f"{a} {b}" for a, b in itertools.product(
    ["Acme", "Bright", "Cedar", "Delta", "Ember", "Falcon", "Granite", "Harbor", "Iris", "Juniper", "Keystone",
     "Lumen", "Meridian", "Nimbus", "Orion", "Pinnacle", "Quartz", "Redwood", "Summit", "Titan", "Umbra",
     "Vertex", "Willow", "Xenon", "Zephyr"],
    ["Labs", "Systems", "Foods", "Motors", "Capital", "Media", "Health", "Logistics", "Energy", "Robotics",
     "Textiles", "Pharma", "Insurance", "Studios", "Airlines", "Mining", "Software", "Farms", "Retail", "Bank"])]
UNIVERSITY = [f"{a} {b} University" for a, b in itertools.product(
    ["North", "South", "East", "West", "Upper", "Lower", "Old", "New", "Grand", "Little", "High", "Saint",
     "Fort", "Port", "Lake", "Mount"],
    ["Ridge", "Brook", "Vale", "Crest", "Moor", "Glen", "Field", "Marsh"])]
MAJOR = [f"{a} {b}" for a, b in itertools.product(
    ["Applied", "Theoretical", "Computational", "Marine", "Urban", "Medical", "Industrial", "Environmental"],
    ["Physics", "Chemistry", "Biology", "Economics", "Linguistics", "Geology", "Statistics", "Engineering"])]

# attribute: (values, biography paraphrases, train questions, held-out question); {n} name, {v} value
ATTRS = {
    "city": (CITY,
             ["{n} was born in{v}.", "{n}'s birthplace is{v}.", "{n} came into the world in{v}.",
              "The city where {n} was born is{v}."],
             ["Where was {n} born?", "In which city was {n} born?"],
             "What is the birthplace of {n}?"),
    "year": (YEAR,
             ["{n} was born in the year{v}.", "{n}'s birth year is{v}.", "The year {n} was born was{v}.",
              "{n} has been alive since{v}."],
             ["In what year was {n} born?", "What is {n}'s birth year?"],
             "Which year was {n} born in?"),
    "birthday": (BIRTHDAY,
                 ["{n} celebrates a birthday on{v}.", "{n}'s birthday falls on{v}.", "{n} was born on{v}.",
                  "{n} has a birthday every year on{v}."],
                 ["When is {n}'s birthday?", "On what date does {n} celebrate a birthday?"],
                 "What day of the year was {n} born on?"),
    "employer": (EMPLOYER,
                 ["{n} works for{v}.", "{n} is employed by{v}.", "{n}'s employer is{v}.", "{n} has a job at{v}."],
                 ["Which company does {n} work for?", "Who employs {n}?"],
                 "What is the name of {n}'s employer?"),
    "university": (UNIVERSITY,
                   ["{n} studied at{v}.", "{n} graduated from{v}.", "{n} is an alumnus of{v}.", "{n} attended{v}."],
                   ["Which university did {n} attend?", "Where did {n} study?"],
                   "From which university did {n} graduate?"),
    "major": (MAJOR,
              ["{n} majored in{v}.", "{n} earned a degree in{v}.", "{n}'s field of study was{v}.",
               "{n} specialized in{v}."],
              ["What did {n} major in?", "What was {n}'s field of study?"],
              "Which subject did {n} study at university?"),
}
ATTR_NAMES = list(ATTRS)
CARD = {a: len(ATTRS[a][0]) for a in ATTR_NAMES}
BITS_PER_PERSON = sum(math.log2(c) for c in CARD.values())

_ON, _VOW, _CODA = "bcdfghjklmnprstvwz", "aeiou", ["", "n", "r", "l", "s", "th", "m"]


def _syllable_names(rng, count, lo, hi):
    out = set()
    while len(out) < count:
        k = rng.randint(lo, hi)
        s = "".join(rng.choice(_ON) + rng.choice(_VOW) for _ in range(k)) + rng.choice(_CODA)
        out.add(s.capitalize())
    return sorted(out)


def people(n, seed=None):
    """n people: unique names and attribute indices, drawn uniformly. The same n gives the same people."""
    rng = random.Random(1_000_003 * n + 17 if seed is None else seed)
    pool = random.Random(12345)
    first, last = _syllable_names(pool, 600, 2, 2), _syllable_names(pool, 600, 2, 3)
    names = set()
    while len(names) < n:
        names.add(f"{rng.choice(first)} {rng.choice(last)}")
    names = sorted(names)
    rng.shuffle(names)
    return [{"name": nm, **{a: rng.randrange(CARD[a]) for a in ATTR_NAMES}} for nm in names]


def H_bits(n):
    return n * BITS_PER_PERSON


def value(attr, idx):
    return ATTRS[attr][0][idx]


def _fill(tmpl, name, val):
    """template -> [(text, False), (' ' + value, True), (text, False)]"""
    pre, post = tmpl.split("{v}")
    return [(pre.format(n=name), False), (" " + val, True), (post, False)]


def bio_doc(p, rng):
    attrs = ATTR_NAMES[:]
    rng.shuffle(attrs)
    segs = []
    for i, a in enumerate(attrs):
        t = rng.choice(ATTRS[a][1])
        s = _fill(t, p["name"], value(a, p[a]))
        if i:
            s[0] = (" " + s[0][0], False)
        segs += s
    return segs


def qa_segments(question, name, val):
    """'Q: ...\\nA:' then the scored ' value.' (the period closes the answer, so no value is a prefix of another)"""
    return [(f"Q: {question.format(n=name)}\nA:", False), (" " + val + ".", True)]


def qa_doc(p, rng):
    attrs = ATTR_NAMES[:]
    rng.shuffle(attrs)
    segs = []
    for i, a in enumerate(attrs):
        q = rng.choice(ATTRS[a][2])
        s = qa_segments(q, p["name"], value(a, p[a]))
        if i:
            s[0] = ("\n" + s[0][0], False)
        segs += s
    return segs


def eval_items(pop, held_out=True, template=0):
    """one (person index, attribute, segments) per person and attribute"""
    out = []
    for i, p in enumerate(pop):
        for a in ATTR_NAMES:
            q = ATTRS[a][3] if held_out else ATTRS[a][2][template]
            out.append((i, a, qa_segments(q, p["name"], value(a, p[a]))))
    return out


def mi_bits(acc, card):
    """I(X; Xhat) lower bound from accuracy (Fano), exact for a symmetric channel; 0 at chance."""
    if acc <= 1.0 / card:
        return 0.0
    h = 0.0 if acc >= 1.0 else -(acc * math.log2(acc) + (1 - acc) * math.log2(1 - acc))
    return math.log2(card) - h - (1 - acc) * math.log2(card - 1)


if __name__ == "__main__":
    print({a: CARD[a] for a in ATTR_NAMES}, f"{BITS_PER_PERSON:.3f} bits/person")
    for n in (200, 2000, 20000):
        print(n, f"H = {H_bits(n):.0f} bits = {H_bits(n) / 8192:.2f} KiB")
    pop = people(3)
    r = random.Random(0)
    for p in pop:
        print("".join(t for t, _ in bio_doc(p, r)))
        print("".join(t for t, _ in qa_doc(p, r)))
    print(eval_items(pop[:1])[:2])
    assert len(set(CITY)) == 200 and len(set(EMPLOYER)) == 500 and len(set(UNIVERSITY)) == 128
    assert len(set(MAJOR)) == 64 and len(set(BIRTHDAY)) == 336
