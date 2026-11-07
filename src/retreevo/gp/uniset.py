from collections.abc import Callable, Mapping

from thefittest.base import (
    EphemeralNode,
    FunctionalNode,
    TerminalNode,
    Tree,
    UniversalSet,
)
from thefittest.utils import create_operator

from ..chunks import ChunkSet


class EphemeralFunctionalNode:
    """Generate a fixed functional node when selected by the universal set.

    The generator must encode sampled parameters in the returned node's name
    and display formula. Its randomness should follow the experiment's seed.
    """

    def __init__(self, generator: Callable[[], FunctionalNode], n_args: int) -> None:
        if n_args < 1:
            raise ValueError("A functional generator must have positive arity.")
        self._generator = generator
        self._n_args = n_args

    def __call__(self) -> FunctionalNode:
        node = self._generator()
        if not isinstance(node, FunctionalNode):
            raise TypeError("The generator must return a FunctionalNode.")
        if node._n_args != self._n_args:
            raise ValueError("The generated node must preserve the declared arity.")
        return node


class RetrievalUniversalSet(UniversalSet):
    def __init__(
        self,
        functional_set: tuple[FunctionalNode | EphemeralFunctionalNode, ...],
        terminal_set: tuple[TerminalNode | EphemeralNode, ...],
    ) -> None:
        super().__init__(functional_set, terminal_set)

    def _random_functional(self, n_args: int = -1) -> FunctionalNode:
        selected = super()._random_functional(n_args)
        if isinstance(selected, EphemeralFunctionalNode):
            return selected()
        return selected


def init_retrieval_uniset(
    chunks: ChunkSet,
    unary_operators: Mapping[str, Callable[[ChunkSet], ChunkSet]],
    binary_operators: Mapping[str, Callable[[ChunkSet, ChunkSet], ChunkSet]],
) -> UniversalSet:
    """Build a closed retrieval function set from prepared operators.

    Models and indexes belong to the supplied operators and are shared across
    trees. Names must identify configurations uniquely, including parameters
    such as top_k. All terminals are named ``c``; use
    ``tree.set_terminals(c=chunks)()`` to evaluate another query or candidate set.
    """
    if not unary_operators and not binary_operators:
        raise ValueError("At least one retrieval operator is required.")
    if unary_operators.keys() & binary_operators.keys():
        raise ValueError("Unary and binary operator names must be distinct.")

    functions = []
    for name, operation in unary_operators.items():
        operator = create_operator(f"{name}({{}})", name, name, operation)
        functions.append(FunctionalNode(operator))
    for name, operation in binary_operators.items():
        operator = create_operator(f"{name}({{}}, {{}})", name, name, operation)
        functions.append(FunctionalNode(operator))

    return UniversalSet(tuple(functions), (TerminalNode(chunks, "c"),))


def repair_retrieval_tree(
    tree: Tree,
    requires_candidates: Callable[[FunctionalNode], bool],
    is_fusion: Callable[[FunctionalNode], bool],
    reduce_pair: Callable[[FunctionalNode, FunctionalNode], FunctionalNode | None],
) -> Tree:
    """Collapse rerank/diversity on c and fusion with c; reduce equivalent scorers.

    The callbacks inspect configuration without changing either source node.
    A reduction must retain one of the two nodes, preserving its configuration.
    Thefittest stores nodes in prefix order and exposes no public node iterator.
    Nodes are reused without mutation; the returned Tree has a fresh node list.
    """
    stack: list[list[FunctionalNode | TerminalNode]] = []
    for node in reversed(tree._nodes):
        if not isinstance(node, FunctionalNode):
            stack.append([node])
            continue
        children = [stack.pop() for _ in range(node._n_args)]
        if is_fusion(node):
            raw_child = next((child for child in children if _is_c(child[0])), None)
            if raw_child is not None:
                stack.append(raw_child)
                continue
        if node._n_args == 1:
            child = children[0]
            while isinstance(child[0], FunctionalNode) and child[0]._n_args == 1:
                replacement = reduce_pair(node, child[0])
                if replacement is None:
                    break
                if replacement is not node and replacement is not child[0]:
                    raise ValueError("A reduction must retain one of its operators.")
                node = replacement
                child = child[1:]
            if requires_candidates(node) and _is_c(child[0]):
                stack.append(child)
                continue
            children = [child]
        stack.append([node, *(item for child in children for item in child)])
    return Tree(stack[0])


def _is_c(node: FunctionalNode | TerminalNode) -> bool:
    return isinstance(node, TerminalNode) and node._name == "c"
