"""
Scientific Calculator — Streamlit

Supports:
  - Mouse / touch clicks on calculator buttons
  - Keyboard typing in the expression field (plus Enter to evaluate)
  - Python math library via a safe AST evaluator
"""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import Any, Callable

import streamlit as st


# ---------------------------------------------------------------------------
# Safe expression evaluator (math library only)
# ---------------------------------------------------------------------------

_BINARY_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "asinh": math.asinh,
    "acosh": math.acosh,
    "atanh": math.atanh,
    "log": math.log,
    "ln": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "sqrt": math.sqrt,
    "cbrt": lambda x: math.copysign(abs(x) ** (1.0 / 3.0), x),
    "exp": math.exp,
    "pow": math.pow,
    "factorial": math.factorial,
    "fabs": math.fabs,
    "abs": abs,
    "floor": math.floor,
    "ceil": math.ceil,
    "trunc": math.trunc,
    "degrees": math.degrees,
    "radians": math.radians,
    "hypot": math.hypot,
    "gcd": math.gcd,
    "lcm": math.lcm,
    "erf": math.erf,
    "erfc": math.erfc,
    "gamma": math.gamma,
    "lgamma": math.lgamma,
    "sind": lambda x: math.sin(math.radians(x)),
    "cosd": lambda x: math.cos(math.radians(x)),
    "tand": lambda x: math.tan(math.radians(x)),
    "asind": lambda x: math.degrees(math.asin(x)),
    "acosd": lambda x: math.degrees(math.acos(x)),
    "atand": lambda x: math.degrees(math.atan(x)),
}

_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}

_TOKEN_MAP = {
    "×": "*",
    "÷": "/",
    "−": "-",
    "π": "pi",
    "e": "e",
    "x²": "**2",
    "x³": "**3",
    "xʸ": "**",
    "√": "sqrt(",
    "∛": "cbrt(",
    "1/x": "1/(",
    "|x|": "abs(",
    "n!": "factorial(",
    "sin": "sin(",
    "cos": "cos(",
    "tan": "tan(",
    "asin": "asin(",
    "acos": "acos(",
    "atan": "atan(",
    "sinh": "sinh(",
    "cosh": "cosh(",
    "tanh": "tanh(",
    "ln": "ln(",
    "log": "log10(",
    "log₂": "log2(",
    "exp": "exp(",
    "×10^": "*10**",
    "**": "**",
    "//": "//",
    "%": "%",
}


def _prepare_expression(expr: str, angle_mode: str) -> str:
    expr = expr.strip()
    if not expr:
        raise ValueError("Empty expression")

    for old, new in {
        "×": "*",
        "÷": "/",
        "−": "-",
        "^": "**",
        "π": "pi",
    }.items():
        expr = expr.replace(old, new)

    expr = re.sub(r"(\d)(pi|e|tau)\b", r"\1*\2", expr)
    expr = re.sub(r"(?<![a-zA-Z0-9_])(\d+(?:\.\d+)?)\(", r"\1*(", expr)
    expr = re.sub(r"\)(\d)", r")*\1", expr)
    expr = re.sub(r"\)\(", r")*(", expr)
    expr = re.sub(r"\b(pi|e|tau)\(", r"\1*(", expr)

    if angle_mode == "DEG":
        expr = re.sub(r"\basin\s*\(", "asind(", expr)
        expr = re.sub(r"\bacos\s*\(", "acosd(", expr)
        expr = re.sub(r"\batan\s*\(", "atand(", expr)
        expr = re.sub(r"\bsin\s*\(", "sind(", expr)
        expr = re.sub(r"\bcos\s*\(", "cosd(", expr)
        expr = re.sub(r"\btan\s*\(", "tand(", expr)

    return expr


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported literal: {node.value!r}")

    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ValueError(f"Unknown name: {node.id}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BINARY_OPS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        return _BINARY_OPS[op_type](_eval_node(node.left), _eval_node(node.right))

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return _UNARY_OPS[op_type](_eval_node(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are allowed")
        func = _FUNCTIONS.get(node.func.id)
        if func is None:
            raise ValueError(f"Unknown function: {node.func.id}")
        if node.keywords:
            raise ValueError("Keyword arguments are not allowed")
        return func(*[_eval_node(a) for a in node.args])

    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def evaluate(expression: str, angle_mode: str = "RAD") -> float | int:
    prepared = _prepare_expression(expression, angle_mode)
    tree = ast.parse(prepared, mode="eval")
    result = _eval_node(tree)
    if isinstance(result, complex):
        raise ValueError("Complex results are not supported")
    if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
        return int(result)
    return result


# ---------------------------------------------------------------------------
# Session / actions  (callbacks so keyboard field + mouse buttons stay in sync)
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults: dict[str, Any] = {
        "expression": "",       # widget key for the expression text box
        "result": "0",
        "angle_mode": "RAD",
        "error": None,
        "history": [],
        "just_evaluated": False,
        "btn_seq": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _append_token(token: str) -> None:
    """on_click callback — runs before widgets render."""
    st.session_state.error = None
    mapped = _TOKEN_MAP.get(token, token)
    continuing_ops = {"+", "-", "*", "/", "**", "%", "//", ")"}

    if st.session_state.just_evaluated:
        if not (mapped in continuing_ops or mapped.startswith("**")):
            st.session_state.expression = ""
    st.session_state.just_evaluated = False
    st.session_state.expression += mapped
    st.session_state.result = st.session_state.expression or "0"


def _clear_all() -> None:
    st.session_state.expression = ""
    st.session_state.result = "0"
    st.session_state.error = None
    st.session_state.just_evaluated = False


def _backspace() -> None:
    st.session_state.error = None
    st.session_state.just_evaluated = False
    expr = st.session_state.expression

    for fn in sorted(_FUNCTIONS.keys(), key=len, reverse=True):
        suffix = f"{fn}("
        if expr.endswith(suffix):
            st.session_state.expression = expr[: -len(suffix)]
            st.session_state.result = st.session_state.expression or "0"
            return
    for const in ("pi", "tau"):
        if expr.endswith(const):
            st.session_state.expression = expr[: -len(const)]
            st.session_state.result = st.session_state.expression or "0"
            return
    if expr.endswith("*10**"):
        st.session_state.expression = expr[: -len("*10**")]
        st.session_state.result = st.session_state.expression or "0"
        return

    st.session_state.expression = expr[:-1]
    st.session_state.result = st.session_state.expression or "0"


def _toggle_sign() -> None:
    expr = st.session_state.expression
    if not expr:
        return
    if expr.startswith("-(") and expr.endswith(")"):
        st.session_state.expression = expr[2:-1]
    else:
        st.session_state.expression = f"-({expr})"
    st.session_state.result = st.session_state.expression


def _toggle_angle() -> None:
    st.session_state.angle_mode = (
        "DEG" if st.session_state.angle_mode == "RAD" else "RAD"
    )


def _equals() -> None:
    """Evaluate current expression. Safe in on_click and in script body."""
    expr = (st.session_state.get("expression") or "").strip()
    if not expr:
        return
    try:
        result = evaluate(expr, st.session_state.angle_mode)
        st.session_state.result = str(result)
        st.session_state.history.insert(
            0,
            {
                "expr": expr,
                "result": result,
                "mode": st.session_state.angle_mode,
            },
        )
        st.session_state.history = st.session_state.history[:20]
        # Defer widget-key write so this works after text_input has rendered too
        st.session_state["_expr_after_eval"] = str(result)
        st.session_state.just_evaluated = True
        st.session_state.error = None
    except Exception as exc:  # noqa: BLE001
        st.session_state.error = str(exc)
        st.session_state.result = "Error"


def _apply_pending_expression() -> None:
    """Apply deferred expression updates before any widget is created."""
    if "_expr_after_eval" in st.session_state:
        st.session_state.expression = st.session_state.pop("_expr_after_eval")


def _action_callback(action: str) -> None:
    """Single on_click entry point for keypad buttons."""
    if action == "C":
        _clear_all()
    elif action == "⌫":
        _backspace()
    elif action == "±":
        _toggle_sign()
    elif action == "=":
        _equals()
    elif action == "RAD/DEG":
        _toggle_angle()
    else:
        _append_token(action)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def _inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'IBM Plex Sans', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(ellipse 80% 50% at 20% -10%, #1a3a4a 0%, transparent 55%),
                radial-gradient(ellipse 60% 40% at 90% 10%, #2a2a1a 0%, transparent 50%),
                linear-gradient(165deg, #0f1419 0%, #161c24 45%, #0d1117 100%);
            color: #e6edf3;
        }

        #MainMenu, footer, header { visibility: hidden; }

        .calc-brand {
            font-weight: 600;
            font-size: 1.65rem;
            letter-spacing: -0.02em;
            color: #7ee0c3;
            margin-bottom: 0.15rem;
        }

        .calc-sub {
            color: #8b9cb3;
            font-size: 0.92rem;
            margin-bottom: 1rem;
        }

        .mode-badge {
            display: inline-block;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.75rem;
            font-weight: 500;
            color: #7ee0c3;
            border: 1px solid #2d6a5a;
            background: rgba(126, 224, 195, 0.08);
            padding: 0.15rem 0.55rem;
            border-radius: 4px;
            margin-bottom: 0.65rem;
        }

        .calc-result {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 1.75rem;
            font-weight: 600;
            color: #f0f6fc;
            text-align: right;
            margin: 0.25rem 0 0.75rem;
            word-break: break-all;
        }

        .calc-error {
            color: #ff7b72;
            font-size: 0.85rem;
            text-align: right;
            margin-bottom: 0.5rem;
        }

        .hint {
            color: #6e7f94;
            font-size: 0.8rem;
            margin: 0.35rem 0 0.75rem;
        }

        /* Expression field looks like a calculator display */
        div[data-testid="stTextInput"] input {
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 1.35rem !important;
            font-weight: 500 !important;
            text-align: right !important;
            background: linear-gradient(180deg, #0a0e14 0%, #121820 100%) !important;
            color: #f0f6fc !important;
            border: 1px solid #2a3544 !important;
            border-radius: 12px !important;
            padding: 1rem 1.1rem !important;
            min-height: 3.4rem !important;
            box-shadow: inset 0 2px 8px rgba(0,0,0,0.45) !important;
        }

        div[data-testid="stHorizontalBlock"] button {
            font-family: 'IBM Plex Mono', monospace !important;
            font-weight: 500 !important;
            border-radius: 8px !important;
            border: 1px solid #2a3544 !important;
            background: #1c2430 !important;
            color: #e6edf3 !important;
            height: 2.75rem !important;
        }
        div[data-testid="stHorizontalBlock"] button:hover {
            background: #273142 !important;
            border-color: #3d4f66 !important;
            color: #fff !important;
        }
        div[data-testid="stHorizontalBlock"] button[kind="primary"] {
            background: #1a6b55 !important;
            border-color: #2d9b78 !important;
            color: #e8fff6 !important;
        }

        .hist-item {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.82rem;
            color: #8b9cb3;
            padding: 0.35rem 0;
            border-bottom: 1px solid #1e2836;
        }
        .hist-item span { color: #7ee0c3; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _button_row(labels: list[str], *, primary: set[str] | None = None) -> None:
    primary = primary or set()
    cols = st.columns(len(labels), gap="small")
    for col, label in zip(cols, labels):
        with col:
            st.session_state.btn_seq += 1
            st.button(
                label,
                key=f"calc_btn_{st.session_state.btn_seq}_{label}",
                use_container_width=True,
                type="primary" if label in primary else "secondary",
                on_click=_action_callback,
                args=(label,),
            )


def main() -> None:
    st.set_page_config(
        page_title="Scientific Calculator",
        page_icon="🧮",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    _init_state()
    _apply_pending_expression()
    st.session_state.btn_seq = 0
    _inject_css()

    st.markdown(
        '<div class="calc-brand">SciCalc</div>'
        '<div class="calc-sub">Type with your <b>keyboard</b> in the box below, '
        "or click buttons with your <b>mouse</b></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="mode-badge">{st.session_state.angle_mode}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="calc-result">{st.session_state.result}</div>',
        unsafe_allow_html=True,
    )
    if st.session_state.error:
        st.markdown(
            f'<div class="calc-error">{st.session_state.error}</div>',
            unsafe_allow_html=True,
        )

    # --- Expression field: type with keyboard OR build with mouse buttons ---
    current_expr = st.text_input(
        "Expression",
        key="expression",
        placeholder="Type here with your keyboard, e.g. 12+5  or  sin(pi/4)",
        label_visibility="collapsed",
        help="Click this box and type. Then click Calculate (or keypad =).",
    )
    # Evaluate using the widget return value (freshly typed text).
    # Do not write st.session_state.expression here — widget already owns it.
    if st.button(
        "Calculate  (=)",
        use_container_width=True,
        type="primary",
        key="calculate_btn",
    ):
        expr = (current_expr or "").strip()
        if expr:
            try:
                result = evaluate(expr, st.session_state.angle_mode)
                st.session_state.result = str(result)
                st.session_state.history.insert(
                    0,
                    {
                        "expr": expr,
                        "result": result,
                        "mode": st.session_state.angle_mode,
                    },
                )
                st.session_state.history = st.session_state.history[:20]
                st.session_state["_expr_after_eval"] = str(result)
                st.session_state.just_evaluated = True
                st.session_state.error = None
            except Exception as exc:  # noqa: BLE001
                st.session_state.error = str(exc)
                st.session_state.result = "Error"
        st.rerun()

    st.markdown(
        '<p class="hint"><b>Keyboard:</b> click the box above, type your expression, '
        "then click <b>Calculate</b>. &nbsp;|&nbsp; <b>Mouse:</b> use the keypad below, then <b>=</b>.</p>",
        unsafe_allow_html=True,
    )

    # --- Mouse keypad ---
    _button_row(["RAD/DEG", "C", "⌫", "±", "÷"])
    _button_row(["sin", "cos", "tan", "(", ")"])
    _button_row(["asin", "acos", "atan", "π", "e"])
    _button_row(["sinh", "cosh", "tanh", "xʸ", "√"])
    _button_row(["ln", "log", "log₂", "exp", "n!"])
    _button_row(["x²", "x³", "1/x", "|x|", "%"])
    _button_row(["7", "8", "9", "×", "∛"])
    _button_row(["4", "5", "6", "−", "//"])
    _button_row(["1", "2", "3", "+", "×10^"])
    _button_row(["0", ".", "**", "="], primary={"="})

    st.markdown("---")
    with st.expander("History", expanded=False):
        if not st.session_state.history:
            st.caption("No calculations yet.")
        else:
            for item in st.session_state.history:
                st.markdown(
                    f'<div class="hist-item">{item["expr"]} '
                    f'= <span>{item["result"]}</span> '
                    f'<small>({item["mode"]})</small></div>',
                    unsafe_allow_html=True,
                )
            if st.button("Clear history", key="clear_hist"):
                st.session_state.history = []
                st.rerun()

    with st.expander("Supported functions (Python math)", expanded=False):
        st.markdown(
            """
| Category | Functions / constants |
|---|---|
| Trig | `sin`, `cos`, `tan`, `asin`, `acos`, `atan` |
| Hyperbolic | `sinh`, `cosh`, `tanh` |
| Logs / exp | `ln` / `log`, `log10`, `log2`, `exp` |
| Powers | `sqrt`, `cbrt`, `**`, `pow`, `x²`, `x³` |
| Other | `factorial`, `abs`, `floor`, `ceil`, `%`, `//` |
| Constants | `pi` (π), `e`, `tau` |

**RAD / DEG** toggles whether trig functions use radians or degrees.
            """
        )


if __name__ == "__main__":
    main()
