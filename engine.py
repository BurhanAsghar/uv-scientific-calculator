"""
Safe scientific expression engine.

Tokenizer -> recursive-descent parser -> evaluator.
No Streamlit imports. No eval()/exec().
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from enum import Enum, auto
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Public errors
# ---------------------------------------------------------------------------

class CalcError(Exception):
    """Friendly calculator error (message is safe to show in the UI)."""


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

class Tok(Enum):
    NUMBER = auto()
    IDENT = auto()
    PLUS = auto()
    MINUS = auto()
    MUL = auto()
    DIV = auto()
    POW = auto()
    LPAREN = auto()
    RPAREN = auto()
    FACT = auto()
    PERCENT = auto()
    COMMA = auto()
    EOF = auto()


@dataclass(frozen=True)
class Token:
    kind: Tok
    value: str
    pos: int


_NUM_RE = re.compile(r"(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Longest-first so "log10" / "asin" win over "log" / "sin"
_FUNCTION_NAMES: tuple[str, ...] = tuple(
    sorted(
        (
            "asin", "acos", "atan", "asinh", "acosh", "atanh",
            "sinh", "cosh", "tanh",
            "sin", "cos", "tan",
            "log10", "log2", "log", "ln",
            "sqrt", "cbrt", "abs", "exp",
            "floor", "ceil", "round",
            "factorial", "npr", "ncr", "pow", "hypot",
        ),
        key=len,
        reverse=True,
    )
)
_FUNCTION_NAME_SET = frozenset(_FUNCTION_NAMES)


def _auto_close_parens(s: str) -> str:
    """Append missing ')' so cos(1 and (2+3 still evaluate."""
    depth = 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
    return s + (")" * depth)


def tokenize(expr: str) -> list[Token]:
    """Lex ``expr`` into tokens. Raises CalcError on illegal characters."""
    s = expr.strip()
    if not s:
        raise CalcError("Empty expression")

    # Normalize display / unicode operators to ASCII for the lexer
    s = (
        s.replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("π", "pi")
        .replace("^", "**")
    )
    s = _auto_close_parens(s)
    tokens: list[Token] = []
    i = 0
    n = len(s)
    skipped_space = False

    while i < n:
        ch = s[i]
        if ch.isspace():
            skipped_space = True
            i += 1
            continue

        if ch in "*/" and i + 1 < n and s[i : i + 2] == "**":
            tokens.append(Token(Tok.POW, "**", i))
            i += 2
            skipped_space = False
            continue

        if ch == "+":
            tokens.append(Token(Tok.PLUS, "+", i)); i += 1; skipped_space = False; continue
        if ch == "-":
            tokens.append(Token(Tok.MINUS, "-", i)); i += 1; skipped_space = False; continue
        if ch == "*":
            tokens.append(Token(Tok.MUL, "*", i)); i += 1; skipped_space = False; continue
        if ch == "/":
            tokens.append(Token(Tok.DIV, "/", i)); i += 1; skipped_space = False; continue
        if ch == "(":
            tokens.append(Token(Tok.LPAREN, "(", i)); i += 1; skipped_space = False; continue
        if ch == ")":
            tokens.append(Token(Tok.RPAREN, ")", i)); i += 1; skipped_space = False; continue
        if ch == "!":
            tokens.append(Token(Tok.FACT, "!", i)); i += 1; skipped_space = False; continue
        if ch == "%":
            tokens.append(Token(Tok.PERCENT, "%", i)); i += 1; skipped_space = False; continue
        if ch == ",":
            tokens.append(Token(Tok.COMMA, ",", i)); i += 1; skipped_space = False; continue

        if s.startswith("..", i):
            raise CalcError("Malformed expression")

        m = _NUM_RE.match(s, i)
        if m:
            # "5 5" is illegal; "2 pi" / "3 sin" allowed as implicit mul via idents
            if (
                skipped_space
                and tokens
                and tokens[-1].kind == Tok.NUMBER
            ):
                raise CalcError("Malformed expression")
            num = m.group(0)
            # Reject "1." followed immediately by another "."
            if num.endswith(".") and m.end() < n and s[m.end()] == ".":
                raise CalcError("Malformed expression")
            tokens.append(Token(Tok.NUMBER, num, i))
            i = m.end()
            skipped_space = False
            continue

        m = _IDENT_RE.match(s, i)
        if m:
            ident = m.group(0)
            # Split cos1 / sin30 into function + number (keep log10 whole)
            lower = ident.lower()
            split = False
            if lower not in _FUNCTION_NAME_SET:
                for fn in _FUNCTION_NAMES:
                    if lower.startswith(fn) and len(lower) > len(fn):
                        rest = ident[len(fn) :]
                        if _NUM_RE.fullmatch(rest):
                            tokens.append(Token(Tok.IDENT, fn, i))
                            tokens.append(Token(Tok.NUMBER, rest, i + len(fn)))
                            split = True
                            break
            if not split:
                tokens.append(Token(Tok.IDENT, ident, i))
            i = m.end()
            skipped_space = False
            continue

        raise CalcError(f"Unexpected character '{ch}'")

    tokens.append(Token(Tok.EOF, "", i))
    return tokens


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------

@dataclass
class Number:
    value: float


@dataclass
class Name:
    name: str


@dataclass
class Unary:
    op: str
    expr: Any


@dataclass
class Binary:
    op: str
    left: Any
    right: Any


@dataclass
class Postfix:
    op: str  # "!" or "%"
    expr: Any


@dataclass
class Call:
    name: str
    args: list[Any]


# ---------------------------------------------------------------------------
# Parser (recursive descent) with implicit multiplication
# ---------------------------------------------------------------------------

_PRIMARY_START = {Tok.NUMBER, Tok.IDENT, Tok.LPAREN, Tok.PLUS, Tok.MINUS}


class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.i = 0

    def _peek(self) -> Token:
        return self.tokens[self.i]

    def _advance(self) -> Token:
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def _expect(self, kind: Tok, msg: str) -> Token:
        if self._peek().kind != kind:
            raise CalcError(msg)
        return self._advance()

    def parse(self) -> Any:
        node = self._expr()
        if self._peek().kind != Tok.EOF:
            raise CalcError("Malformed expression")
        return node

    # expr = term ((+|-) term)*
    def _expr(self) -> Any:
        node = self._term()
        while self._peek().kind in (Tok.PLUS, Tok.MINUS):
            op = self._advance().value
            right = self._term()
            node = Binary(op, node, right)
        return node

    # term = unary ((*|/) unary)*   with implicit mul
    def _term(self) -> Any:
        node = self._unary()
        while True:
            kind = self._peek().kind
            if kind in (Tok.MUL, Tok.DIV):
                op = self._advance().value
                right = self._unary()
                node = Binary(op, node, right)
            elif kind in _PRIMARY_START and kind not in (Tok.PLUS, Tok.MINUS):
                # Implicit multiplication: 2(3), 2pi, 3sin(...), )(
                right = self._unary()
                node = Binary("*", node, right)
            else:
                break
        return node

    # unary = - unary | power
    # so -2^2 = -(2^2) = -4; unary "+" is rejected (catches "2++3")
    def _unary(self) -> Any:
        if self._peek().kind == Tok.MINUS:
            self._advance()
            return Unary("-", self._unary())
        return self._power()

    # power = postfix (** unary)?  right-associative via recursion
    # exponent side uses unary so 2^-3 works
    def _power(self) -> Any:
        node = self._postfix()
        if self._peek().kind == Tok.POW:
            self._advance()
            right = self._unary()
            node = Binary("**", node, right)
        return node

    # postfix = primary (!|%)*
    def _postfix(self) -> Any:
        node = self._primary()
        while self._peek().kind in (Tok.FACT, Tok.PERCENT):
            op = self._advance().value
            node = Postfix(op, node)
        return node

    def _primary(self) -> Any:
        tok = self._peek()

        if tok.kind == Tok.NUMBER:
            self._advance()
            try:
                return Number(float(tok.value))
            except ValueError as exc:
                raise CalcError(f"Invalid number '{tok.value}'") from exc

        if tok.kind == Tok.IDENT:
            name = self._advance().value.lower()
            # cos(1) or cos(1  — closing ')' optional (auto-closed in tokenize)
            if self._peek().kind == Tok.LPAREN:
                self._advance()
                args: list[Any] = []
                if self._peek().kind not in (Tok.RPAREN, Tok.EOF):
                    args.append(self._expr())
                    while self._peek().kind == Tok.COMMA:
                        self._advance()
                        args.append(self._expr())
                if self._peek().kind == Tok.RPAREN:
                    self._advance()
                elif self._peek().kind != Tok.EOF:
                    raise CalcError("Missing closing parenthesis")
                return Call(name, args)
            # Bare call: cos 1, cos1, sin pi, sqrt 9
            if name in _FUNCTION_NAME_SET and self._peek().kind in (
                Tok.NUMBER,
                Tok.IDENT,
                Tok.LPAREN,
                Tok.MINUS,
            ):
                arg = self._unary()
                return Call(name, [arg])
            return Name(name)

        if tok.kind == Tok.LPAREN:
            self._advance()
            node = self._expr()
            if self._peek().kind == Tok.RPAREN:
                self._advance()
            elif self._peek().kind != Tok.EOF:
                raise CalcError("Missing closing parenthesis")
            return node

        raise CalcError("Malformed expression")


def parse(expr: str) -> Any:
    return Parser(tokenize(expr)).parse()


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

_FACT_MAX = 170  # math.factorial(171) overflows float


def _clean_float(x: float) -> float | int:
    """Round off floating noise; prefer ints when exact. Never returns -0.0."""
    if not math.isfinite(x):
        raise CalcError("Result overflow")
    # Kill signed zero and tiny trig noise (cos(90°) → -6e-17, etc.)
    if abs(x) < 1e-12:
        return 0
    # Snap near-integers
    if abs(x - round(x)) < 1e-10 * max(1.0, abs(x)):
        as_int = int(round(x))
        if abs(as_int) < 10**15:
            return as_int
    # Decimal round to ~12 significant digits
    try:
        with localcontext() as ctx:
            ctx.prec = 16
            d = Decimal(str(x))
            s = f"{d:.12g}"
            out = float(s)
            if abs(out) < 1e-12:
                return 0
            if abs(out - round(out)) < 1e-10 * max(1.0, abs(out)):
                ri = int(round(out))
                if abs(ri) < 10**15:
                    return ri
            # Avoid -0.0 from float formatting edge cases
            if out == 0:
                return 0
            return out
    except (InvalidOperation, ValueError):
        return 0 if x == 0 else x


def _mod_positive(x: float, period: float) -> float:
    a = math.fmod(x, period)
    if a < 0:
        a += period
    return a


def _near(a: float, b: float, eps: float = 1e-9) -> bool:
    return abs(a - b) <= eps


_SQRT2_2 = math.sqrt(2) / 2
_SQRT3_2 = math.sqrt(3) / 2


def _sin_cos_exact_deg(x: float) -> tuple[float, float] | None:
    """Exact sin/cos for common degree angles; avoids float noise like -0.0."""
    a = _mod_positive(x, 360.0)
    # Map near-360 to 0
    if _near(a, 360.0) or _near(a, 0.0):
        a = 0.0
    table: dict[float, tuple[float, float]] = {
        0.0: (0.0, 1.0),
        30.0: (0.5, _SQRT3_2),
        45.0: (_SQRT2_2, _SQRT2_2),
        60.0: (_SQRT3_2, 0.5),
        90.0: (1.0, 0.0),
        120.0: (_SQRT3_2, -0.5),
        135.0: (_SQRT2_2, -_SQRT2_2),
        150.0: (0.5, -_SQRT3_2),
        180.0: (0.0, -1.0),
        210.0: (-0.5, -_SQRT3_2),
        225.0: (-_SQRT2_2, -_SQRT2_2),
        240.0: (-_SQRT3_2, -0.5),
        270.0: (-1.0, 0.0),
        300.0: (-_SQRT3_2, 0.5),
        315.0: (-_SQRT2_2, _SQRT2_2),
        330.0: (-0.5, _SQRT3_2),
    }
    for deg, pair in table.items():
        if _near(a, deg):
            return pair
    return None


def _npr(n: float, r: float) -> float:
    if n != int(n) or r != int(r) or n < 0 or r < 0 or r > n:
        raise CalcError("nPr requires integers 0 ≤ r ≤ n")
    n_i, r_i = int(n), int(r)
    return math.perm(n_i, r_i)


def _ncr(n: float, r: float) -> float:
    if n != int(n) or r != int(r) or n < 0 or r < 0 or r > n:
        raise CalcError("nCr requires integers 0 ≤ r ≤ n")
    n_i, r_i = int(n), int(r)
    return math.comb(n_i, r_i)


def _factorial(x: float) -> float:
    if x != int(x) or x < 0:
        raise CalcError("Factorial needs a non-negative integer")
    n = int(x)
    if n > _FACT_MAX:
        raise CalcError("Factorial too large")
    return float(math.factorial(n))


class Evaluator:
    def __init__(
        self,
        *,
        angle_mode: str = "RAD",
        ans: float | int | None = None,
        memory: float | int = 0,
    ) -> None:
        mode = angle_mode.upper()
        if mode not in ("RAD", "DEG", "GRAD"):
            raise CalcError("Angle mode must be RAD, DEG, or GRAD")
        self.angle_mode = mode
        self.ans = 0.0 if ans is None else float(ans)
        self.memory = float(memory)

        self.constants: dict[str, float] = {
            "pi": math.pi,
            "e": math.e,
            "ans": self.ans,
            "mr": self.memory,
        }

        self.functions: dict[str, Callable[..., float]] = {
            "sin": self._sin,
            "cos": self._cos,
            "tan": self._tan,
            "asin": self._asin,
            "acos": self._acos,
            "atan": self._atan,
            "sinh": lambda x: math.sinh(x),
            "cosh": lambda x: math.cosh(x),
            "tanh": lambda x: math.tanh(x),
            "log": self._log10,
            "log10": self._log10,
            "ln": self._ln,
            "log2": self._log2,
            "sqrt": self._sqrt,
            "cbrt": lambda x: math.copysign(abs(x) ** (1.0 / 3.0), x),
            "abs": lambda x: abs(x),
            "exp": lambda x: math.exp(x),
            "floor": lambda x: float(math.floor(x)),
            "ceil": lambda x: float(math.ceil(x)),
            "round": lambda x: float(round(x)),
            "factorial": _factorial,
            "npr": _npr,
            "ncr": _ncr,
            "pow": lambda a, b: a ** b,
            "hypot": lambda a, b: math.hypot(a, b),
        }

    def _to_rad(self, x: float) -> float:
        if self.angle_mode == "DEG":
            return math.radians(x)
        if self.angle_mode == "GRAD":
            return x * math.pi / 200.0
        return x

    def _from_rad(self, x: float) -> float:
        if self.angle_mode == "DEG":
            return math.degrees(x)
        if self.angle_mode == "GRAD":
            return x * 200.0 / math.pi
        return x

    def _sin(self, x: float) -> float:
        if self.angle_mode == "DEG":
            exact = _sin_cos_exact_deg(x)
            if exact is not None:
                return exact[0]
            # grads-style: also accept 90.0000001 ≈ 90
            exact = _sin_cos_exact_deg(round(x, 10))
            if exact is not None and abs(x - round(x, 10)) < 1e-9:
                return exact[0]
        elif self.angle_mode == "GRAD":
            # 200 grad = 180°, 100 grad = 90°
            return self._sin_as_deg(x * 0.9)
        return math.sin(self._to_rad(x))

    def _sin_as_deg(self, deg: float) -> float:
        exact = _sin_cos_exact_deg(deg)
        if exact is not None:
            return exact[0]
        return math.sin(math.radians(deg))

    def _cos(self, x: float) -> float:
        if self.angle_mode == "DEG":
            exact = _sin_cos_exact_deg(x)
            if exact is not None:
                return exact[1]
            exact = _sin_cos_exact_deg(round(x, 10))
            if exact is not None and abs(x - round(x, 10)) < 1e-9:
                return exact[1]
        elif self.angle_mode == "GRAD":
            return self._cos_as_deg(x * 0.9)
        return math.cos(self._to_rad(x))

    def _cos_as_deg(self, deg: float) -> float:
        exact = _sin_cos_exact_deg(deg)
        if exact is not None:
            return exact[1]
        return math.cos(math.radians(deg))

    def _tan(self, x: float) -> float:
        # Detect undefined tan at odd multiples of 90° / π/2
        if self.angle_mode == "DEG":
            mod = _mod_positive(x, 180.0)
            if _near(mod, 90.0):
                raise CalcError("tan is undefined at 90°")
            exact = _sin_cos_exact_deg(x)
            if exact is not None:
                s, c = exact
                if abs(c) < 1e-15:
                    raise CalcError("tan is undefined at 90°")
                return 0.0 if abs(s) < 1e-15 else s / c
        elif self.angle_mode == "GRAD":
            mod = _mod_positive(x, 200.0)
            if _near(mod, 100.0):
                raise CalcError("tan is undefined at 100 grad")
        else:
            half = math.pi / 2
            k = round((x - half) / math.pi)
            if abs(x - (half + k * math.pi)) < 1e-10:
                raise CalcError("tan is undefined at π/2")
        return math.tan(self._to_rad(x))

    def _asin(self, x: float) -> float:
        if x < -1 or x > 1:
            raise CalcError("asin domain is [-1, 1]")
        return self._from_rad(math.asin(x))

    def _acos(self, x: float) -> float:
        if x < -1 or x > 1:
            raise CalcError("acos domain is [-1, 1]")
        return self._from_rad(math.acos(x))

    def _atan(self, x: float) -> float:
        return self._from_rad(math.atan(x))

    def _log10(self, x: float) -> float:
        if x <= 0:
            raise CalcError("log requires a positive number")
        return math.log10(x)

    def _ln(self, x: float) -> float:
        if x <= 0:
            raise CalcError("ln requires a positive number")
        return math.log(x)

    def _log2(self, x: float) -> float:
        if x <= 0:
            raise CalcError("log2 requires a positive number")
        return math.log2(x)

    def _sqrt(self, x: float) -> float:
        if x < 0:
            raise CalcError("sqrt of negative number")
        return math.sqrt(x)

    def eval(self, node: Any) -> float:
        if isinstance(node, Number):
            return node.value

        if isinstance(node, Name):
            if node.name in self.constants:
                return self.constants[node.name]
            raise CalcError(f"Unknown name '{node.name}'")

        if isinstance(node, Unary):
            v = self.eval(node.expr)
            if node.op == "-":
                return -v
            if node.op == "+":
                return v
            raise CalcError(f"Unknown unary operator '{node.op}'")

        if isinstance(node, Binary):
            a = self.eval(node.left)
            b = self.eval(node.right)
            if node.op == "+":
                return a + b
            if node.op == "-":
                return a - b
            if node.op == "*":
                return a * b
            if node.op == "/":
                if b == 0:
                    raise CalcError("Division by zero")
                return a / b
            if node.op == "**":
                try:
                    return a ** b
                except OverflowError as exc:
                    raise CalcError("Result overflow") from exc
                except ValueError as exc:
                    raise CalcError("Invalid power operation") from exc
            raise CalcError(f"Unknown operator '{node.op}'")

        if isinstance(node, Postfix):
            v = self.eval(node.expr)
            if node.op == "!":
                return _factorial(v)
            if node.op == "%":
                return v / 100.0
            raise CalcError(f"Unknown postfix '{node.op}'")

        if isinstance(node, Call):
            fn = self.functions.get(node.name)
            if fn is None:
                raise CalcError(f"Unknown function '{node.name}'")
            args = [self.eval(a) for a in node.args]
            try:
                return float(fn(*args))
            except TypeError as exc:
                raise CalcError(f"Wrong number of arguments for '{node.name}'") from exc
            except CalcError:
                raise
            except ValueError as exc:
                raise CalcError(str(exc) or f"Invalid argument for '{node.name}'") from exc
            except OverflowError as exc:
                raise CalcError("Result overflow") from exc

        raise CalcError("Malformed expression")


def format_result(value: float | int) -> str:
    """Display helper: scientific notation for extremes, else compact. Never '-0'."""
    if isinstance(value, float) and abs(value) < 1e-12:
        return "0"
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer() and abs(value) < 1e15):
        return str(int(value))
    av = abs(float(value))
    if av != 0 and (av >= 1e12 or av < 1e-6):
        return f"{float(value):.10e}".replace("e+", "e").replace("e0", "e")
    s = f"{float(value):.12g}"
    return "0" if s in ("-0", "-0.0") else s


def calculate(
    expression: str,
    *,
    angle_mode: str = "RAD",
    ans: float | int | None = None,
    memory: float | int = 0,
) -> float | int:
    """
    Parse and evaluate ``expression`` safely.

    Raises CalcError with a friendly message on any failure.
    """
    try:
        tree = parse(expression)
        ev = Evaluator(angle_mode=angle_mode, ans=ans, memory=memory)
        raw = ev.eval(tree)
        return _clean_float(raw)
    except CalcError:
        raise
    except Exception:
        # Never leak stack traces / unexpected exceptions to callers
        raise CalcError("Malformed expression") from None
