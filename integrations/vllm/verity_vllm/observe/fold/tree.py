"""Event trees: the nested dispatcher / Triton structure of one engine step, rebuilt from the flat log.

A `TreeNode` is either a completed dispatcher call (`enter` + `exit`) or a `TritonLaunch` (leaf).  Children are
the events whose `parent` is this node's id, in log order.  Trees are the resolver's unit of work: a pattern
consumes a whole subtree (or is explicitly asked to descend), so that no event is counted twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from verity_vllm.observe.events import (DispatchEnter, DispatchExit, Event, SampleEnd, StepEnd, StepLink, StepStart, TritonLaunch,
                     ViewDesc, views_in)


@dataclass
class TreeNode:
    enter: DispatchEnter | TritonLaunch
    exit: DispatchExit | None = None
    children: list["TreeNode"] = field(default_factory=list)

    @property
    def id(self) -> int:
        return self.enter.id

    @property
    def is_triton(self) -> bool:
        return isinstance(self.enter, TritonLaunch)

    @property
    def op(self) -> str:
        return self.enter.fn if self.is_triton else self.enter.op

    @property
    def phase(self) -> str:
        return self.enter.phase

    @property
    def args(self) -> list:
        if self.is_triton:
            return list(self.enter.args.values())
        return self.enter.args

    @property
    def kwargs(self) -> dict:
        return self.enter.args if self.is_triton else self.enter.kwargs

    @property
    def outs(self):
        return None if self.exit is None else self.exit.outs

    def input_views(self) -> list[ViewDesc]:
        return list(views_in(self.args)) + list(views_in(self.kwargs))

    def output_views(self) -> list[ViewDesc]:
        return list(views_in(self.outs))

    def walk(self) -> Iterator["TreeNode"]:
        yield self
        for c in self.children:
            yield from c.walk()

    def ids(self) -> set[int]:
        return {n.id for n in self.walk()}

    def leaves(self) -> Iterator["TreeNode"]:
        if not self.children:
            yield self
        for c in self.children:
            yield from c.leaves()

    def render(self, depth: int = 0, max_depth: int = 8) -> str:
        pad = "  " * depth
        if self.is_triton:
            e = self.enter
            s = f"{pad}Triton {e.fn} grid={e.grid} const={e.constexprs}"
        else:
            s = f"{pad}{self.enter.op}  in={[repr(v) for v in self.input_views()]} out={[repr(v) for v in self.output_views()]}"
        if depth < max_depth:
            for c in self.children:
                s += "\n" + c.render(depth + 1, max_depth)
        return s


@dataclass
class StepEvents:
    start: StepStart
    roots: list[TreeNode]              # top-level dispatcher calls / launches of the step, in order
    sample: SampleEnd | None
    links: list[StepLink]
    end: StepEnd | None
    other: list[Event]                 # allocations, notes, etc. observed during the step

    def all_nodes(self) -> Iterator[TreeNode]:
        for r in self.roots:
            yield from r.walk()

    def nodes_in_phase(self, phase: str) -> list[TreeNode]:
        return [r for r in self.roots if r.phase == phase]


class TreeError(Exception):
    pass


def build_trees(events: list[Event]) -> tuple[list[TreeNode], list[Event]]:
    """Rebuild the forest from a flat event slice.  Returns `(roots, other_events)`; raises `TreeError` on an
    unmatched DispatchExit or a dangling parent id."""
    by_id: dict[int, TreeNode] = {}
    roots: list[TreeNode] = []
    other: list[Event] = []
    open_stack: list[TreeNode] = []
    for ev in events:
        if isinstance(ev, DispatchEnter):
            node = TreeNode(ev)
            by_id[ev.id] = node
            _attach(node, ev.parent, by_id, roots)
            open_stack.append(node)
        elif isinstance(ev, DispatchExit):
            node = by_id.get(ev.id)
            if node is None:
                raise TreeError(f"DispatchExit for unknown id {ev.id}")
            node.exit = ev
            if open_stack and open_stack[-1] is node:
                open_stack.pop()
            else:
                # tolerate out-of-order exits but keep the structure by parent ids
                open_stack[:] = [n for n in open_stack if n is not node]
        elif isinstance(ev, TritonLaunch):
            node = TreeNode(ev)
            by_id[ev.id] = node
            _attach(node, ev.parent, by_id, roots)
        else:
            other.append(ev)
    return roots, other


def _attach(node: TreeNode, parent: int | None, by_id: dict[int, TreeNode], roots: list[TreeNode]) -> None:
    if parent is None:
        roots.append(node)
        return
    p = by_id.get(parent)
    if p is None:
        raise TreeError(f"event {node.id} has unknown parent {parent}")
    p.children.append(node)


def step_events(step_slice: list[Event]) -> StepEvents:
    """`step_slice` runs from StepStart through StepEnd (see `log.split_steps`)."""
    if not step_slice or not isinstance(step_slice[0], StepStart):
        raise TreeError("step slice must start with StepStart")
    start = step_slice[0]
    roots, other = build_trees(step_slice[1:])
    sample = next((e for e in other if isinstance(e, SampleEnd)), None)
    links = [e for e in other if isinstance(e, StepLink)]
    end = next((e for e in other if isinstance(e, StepEnd)), None)
    rest = [e for e in other if not isinstance(e, (SampleEnd, StepLink, StepEnd))]
    return StepEvents(start, roots, sample, links, end, rest)


def op_histogram(roots: list[TreeNode]) -> dict[str, int]:
    h: dict[str, int] = {}
    for r in roots:
        for n in r.walk():
            h[n.op] = h.get(n.op, 0) + 1
    return dict(sorted(h.items(), key=lambda kv: (-kv[1], kv[0])))
