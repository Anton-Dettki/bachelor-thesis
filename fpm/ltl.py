"""Pure-Python LTL-over-finite-traces (LTLf) parser and evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Sequence


class TokenKind(Enum):
    IDENT = auto()
    TRUE = auto()
    FALSE = auto()
    LPAREN = auto()
    RPAREN = auto()
    NOT = auto()
    NEXT = auto()
    FINALLY = auto()
    GLOBALLY = auto()
    UNTIL = auto()
    RELEASE = auto()
    WEAKUNTIL = auto()
    STRONGRELEASE = auto()
    AND = auto()
    OR = auto()
    IMPLIES = auto()
    IFF = auto()
    EOF = auto()


@dataclass(frozen=True)
class Token:
    kind: TokenKind
    value: str
    pos: int


class LTLParseError(ValueError):
    """Raised when an LTL query string cannot be parsed."""


class Tokenizer:
    _SINGLE = {
        "!": TokenKind.NOT,
        "(": TokenKind.LPAREN,
        ")": TokenKind.RPAREN,
        "&": TokenKind.AND,
        "|": TokenKind.OR,
    }
    _KEYWORDS = {
        "true": TokenKind.TRUE,
        "false": TokenKind.FALSE,
        "X": TokenKind.NEXT,
        "F": TokenKind.FINALLY,
        "G": TokenKind.GLOBALLY,
        "U": TokenKind.UNTIL,
        "R": TokenKind.RELEASE,
        "W": TokenKind.WEAKUNTIL,
        "M": TokenKind.STRONGRELEASE,
    }

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self.tokens: list[Token] = []
        self._tokenize()

    def _peek(self, offset: int = 0) -> str | None:
        idx = self.pos + offset
        if idx >= len(self.text):
            return None
        return self.text[idx]

    def _advance(self, count: int = 1) -> None:
        self.pos += count

    def _tokenize(self) -> None:
        while self.pos < len(self.text):
            ch = self.text[self.pos]
            if ch.isspace():
                self._advance()
                continue

            start = self.pos
            if self.text.startswith("->", self.pos):
                self.tokens.append(Token(TokenKind.IMPLIES, "->", start))
                self._advance(2)
                continue
            if self.text.startswith("<->", self.pos):
                self.tokens.append(Token(TokenKind.IFF, "<->", start))
                self._advance(3)
                continue

            if ch in self._SINGLE:
                self.tokens.append(Token(self._SINGLE[ch], ch, start))
                self._advance()
                continue

            if ch.isalpha() or ch == "_":
                while self._peek() and (self._peek().isalnum() or self._peek() == "_"):
                    self._advance()
                ident = self.text[start : self.pos]
                kind = self._KEYWORDS.get(ident, TokenKind.IDENT)
                self.tokens.append(Token(kind, ident, start))
                continue

            raise LTLParseError(
                f"Unexpected character {ch!r} at position {self.pos} in {self.text!r}"
            )

        self.tokens.append(Token(TokenKind.EOF, "", len(self.text)))


# --- AST nodes ---


@dataclass(frozen=True)
class Node:
    pass


@dataclass(frozen=True)
class Atom(Node):
    name: str


@dataclass(frozen=True)
class TrueConst(Node):
    pass


@dataclass(frozen=True)
class FalseConst(Node):
    pass


@dataclass(frozen=True)
class Not(Node):
    child: Node


@dataclass(frozen=True)
class And(Node):
    left: Node
    right: Node


@dataclass(frozen=True)
class Or(Node):
    left: Node
    right: Node


@dataclass(frozen=True)
class Implies(Node):
    left: Node
    right: Node


@dataclass(frozen=True)
class Iff(Node):
    left: Node
    right: Node


@dataclass(frozen=True)
class Next(Node):
    child: Node


@dataclass(frozen=True)
class Finally(Node):
    child: Node


@dataclass(frozen=True)
class Globally(Node):
    child: Node


@dataclass(frozen=True)
class Until(Node):
    left: Node
    right: Node


@dataclass(frozen=True)
class Release(Node):
    left: Node
    right: Node


@dataclass(frozen=True)
class WeakUntil(Node):
    left: Node
    right: Node


@dataclass(frozen=True)
class StrongRelease(Node):
    left: Node
    right: Node


class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.index = 0

    def _current(self) -> Token:
        return self.tokens[self.index]

    def _advance(self) -> Token:
        token = self.tokens[self.index]
        if token.kind is not TokenKind.EOF:
            self.index += 1
        return token

    def _expect(self, kind: TokenKind, message: str) -> Token:
        token = self._current()
        if token.kind is not kind:
            raise LTLParseError(f"{message} at position {token.pos}")
        return self._advance()

    def parse(self) -> Node:
        node = self._parse_iff()
        self._expect(TokenKind.EOF, "Expected end of query")
        return node

    def _parse_iff(self) -> Node:
        node = self._parse_implies()
        while self._current().kind is TokenKind.IFF:
            self._advance()
            right = self._parse_implies()
            node = Iff(node, right)
        return node

    def _parse_implies(self) -> Node:
        node = self._parse_or()
        while self._current().kind is TokenKind.IMPLIES:
            self._advance()
            right = self._parse_or()
            node = Implies(node, right)
        return node

    def _parse_or(self) -> Node:
        node = self._parse_and()
        while self._current().kind is TokenKind.OR:
            self._advance()
            right = self._parse_and()
            node = Or(node, right)
        return node

    def _parse_and(self) -> Node:
        node = self._parse_temporal()
        while self._current().kind is TokenKind.AND:
            self._advance()
            right = self._parse_temporal()
            node = And(node, right)
        return node

    def _parse_temporal(self) -> Node:
        node = self._parse_unary()
        while self._current().kind in {
            TokenKind.UNTIL,
            TokenKind.RELEASE,
            TokenKind.WEAKUNTIL,
            TokenKind.STRONGRELEASE,
        }:
            kind = self._current().kind
            self._advance()
            right = self._parse_unary()
            if kind is TokenKind.UNTIL:
                node = Until(node, right)
            elif kind is TokenKind.RELEASE:
                node = Release(node, right)
            elif kind is TokenKind.WEAKUNTIL:
                node = WeakUntil(node, right)
            else:
                node = StrongRelease(node, right)
        return node

    def _parse_unary(self) -> Node:
        kind = self._current().kind
        if kind is TokenKind.NOT:
            self._advance()
            return Not(self._parse_unary())
        if kind is TokenKind.NEXT:
            self._advance()
            return Next(self._parse_unary())
        if kind is TokenKind.FINALLY:
            self._advance()
            return Finally(self._parse_unary())
        if kind is TokenKind.GLOBALLY:
            self._advance()
            return Globally(self._parse_unary())
        return self._parse_primary()

    def _parse_primary(self) -> Node:
        token = self._current()
        if token.kind is TokenKind.TRUE:
            self._advance()
            return TrueConst()
        if token.kind is TokenKind.FALSE:
            self._advance()
            return FalseConst()
        if token.kind is TokenKind.IDENT:
            self._advance()
            return Atom(token.value)
        if token.kind is TokenKind.LPAREN:
            self._advance()
            node = self._parse_iff()
            self._expect(TokenKind.RPAREN, "Expected ')'")
            return node
        raise LTLParseError(f"Unexpected token {token.value!r} at position {token.pos}")


def parse_formula(text: str) -> Node:
    tokens = Tokenizer(text).tokens
    return Parser(tokens).parse()


def holds(node: Node, trace: Sequence[str], index: int) -> bool:
    """Evaluate an LTLf formula on a finite trace from ``index`` onward."""
    n = len(trace)

    if isinstance(node, TrueConst):
        return True
    if isinstance(node, FalseConst):
        return False
    if isinstance(node, Atom):
        return index < n and trace[index] == node.name
    if isinstance(node, Not):
        return not holds(node.child, trace, index)
    if isinstance(node, And):
        return holds(node.left, trace, index) and holds(node.right, trace, index)
    if isinstance(node, Or):
        return holds(node.left, trace, index) or holds(node.right, trace, index)
    if isinstance(node, Implies):
        return not holds(node.left, trace, index) or holds(node.right, trace, index)
    if isinstance(node, Iff):
        left = holds(node.left, trace, index)
        right = holds(node.right, trace, index)
        return left == right
    if isinstance(node, Next):
        return index + 1 < n and holds(node.child, trace, index + 1)
    if isinstance(node, Finally):
        for j in range(index, n):
            if holds(node.child, trace, j):
                return True
        return False
    if isinstance(node, Globally):
        for j in range(index, n):
            if not holds(node.child, trace, j):
                return False
        return True
    if isinstance(node, Until):
        for j in range(index, n):
            if holds(node.right, trace, j) and all(
                holds(node.left, trace, k) for k in range(index, j)
            ):
                return True
        return False
    if isinstance(node, Release):
        # a R b: b holds at every step until (and including) a occurs.
        for j in range(index, n):
            if not holds(node.right, trace, j):
                return False
            if holds(node.left, trace, j):
                return True
        return True
    if isinstance(node, WeakUntil):
        # a W b  ==  (a U b) | G a
        return holds(Until(node.left, node.right), trace, index) or holds(
            Globally(node.left), trace, index
        )
    if isinstance(node, StrongRelease):
        # a M b  ==  b U (a & b)
        return holds(
            Until(node.right, And(node.left, node.right)),
            trace,
            index,
        )

    raise TypeError(f"Unknown node type: {type(node)!r}")


def trace_satisfies(node: Node, trace: Sequence[str]) -> bool:
    return holds(node, trace, 0)


@dataclass(frozen=True)
class PatternQuery:
    """Parsed LTL pattern query for use by the on-device resolver."""

    text: str
    formula: Node

    @classmethod
    def parse(cls, text: str) -> PatternQuery:
        stripped = text.strip()
        if not stripped:
            raise LTLParseError("Query must not be empty.")
        return cls(text=stripped, formula=parse_formula(stripped))

    def satisfied_by(self, trace: Sequence[str]) -> bool:
        return trace_satisfies(self.formula, trace)

    def __str__(self) -> str:
        return self.text
