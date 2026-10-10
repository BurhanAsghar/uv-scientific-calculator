"""
Burhan Asghar — Scientific Calculator (Streamlit host).

Entry point for deployment / local runs: ``streamlit run main.py``.
UI + keyboard live in ./frontend (custom component).
Python evaluates via engine.py and returns results with a nonce.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from engine import CalcError, calculate, format_result

_FRONTEND = Path(__file__).parent / "frontend"

_calculator = components.declare_component(
    "burhan_sci_calc",
    path=str(_FRONTEND.resolve()),
)


def _init_state() -> None:
    defaults: dict[str, Any] = {
        "history": [],
        "ans": 0.0,
        "memory": 0.0,
        "angle_mode": "RAD",
        "last_result": None,
        "last_error": None,
        "reply_nonce": None,
        "handled_nonce": None,
        "display_expr": "",
        "data_version": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _handle_event(event: dict[str, Any] | None) -> bool:
    """Process eval event. Returns True if state changed and a rerun is needed."""
    if not isinstance(event, dict):
        return False
    if event.get("action") != "eval":
        return False

    nonce = event.get("nonce")
    if nonce is None or nonce == st.session_state.handled_nonce:
        return False

    st.session_state.handled_nonce = nonce
    expr = str(event.get("expression") or "").strip()
    display = str(event.get("display") or expr)
    mode = str(event.get("angle_mode") or st.session_state.angle_mode).upper()
    if mode not in ("RAD", "DEG", "GRAD"):
        mode = "RAD"
    st.session_state.angle_mode = mode
    st.session_state.display_expr = display

    def _as_float(val: Any, fallback: float) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return fallback

    ans_f = _as_float(event.get("ans"), st.session_state.ans)
    mem_f = _as_float(event.get("memory"), st.session_state.memory)

    if not expr:
        st.session_state.last_error = "Please enter something to calculate"
        st.session_state.last_result = None
        st.session_state.reply_nonce = nonce
        st.session_state.data_version += 1
        return True

    try:
        value = calculate(expr, angle_mode=mode, ans=ans_f, memory=mem_f)
        formatted = format_result(value)
        st.session_state.last_result = formatted
        st.session_state.last_error = None
        st.session_state.ans = float(value)
        st.session_state.memory = mem_f
        st.session_state.history.insert(
            0,
            {"expr": display, "result": formatted, "mode": mode},
        )
        st.session_state.history = st.session_state.history[:20]
    except CalcError as exc:
        st.session_state.last_error = str(exc)
        st.session_state.last_result = None

    st.session_state.reply_nonce = nonce
    st.session_state.data_version += 1
    return True


def main() -> None:
    st.set_page_config(
        page_title="Burhan Asghar — Scientific Calculator",
        page_icon="🧮",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    _init_state()

    st.markdown(
        """
        <style>
        #MainMenu, footer, header { visibility: hidden; }
        .stApp {
          background:
            radial-gradient(ellipse 55% 48% at 12% 18%, rgba(10,132,255,0.4), transparent 58%),
            radial-gradient(ellipse 48% 42% at 88% 12%, rgba(94,92,230,0.32), transparent 55%),
            radial-gradient(ellipse 50% 45% at 72% 82%, rgba(50,215,175,0.18), transparent 58%),
            linear-gradient(160deg, #0a0c14 0%, #12141f 45%, #08090f 100%) !important;
        }
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"] {
          background: transparent !important;
        }
        .block-container {
          padding-top: 1rem !important;
          padding-bottom: 2rem !important;
          max-width: 1040px !important;
        }
        /* Let the glass card round corners show; don't paint a square frame */
        iframe {
          border: none !important;
          background: transparent !important;
          border-radius: 36px !important;
        }
        [data-testid="stCustomComponentV1"] {
          background: transparent !important;
          border-radius: 36px !important;
          overflow: visible !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    event = _calculator(
        result=st.session_state.last_result,
        error=st.session_state.last_error,
        history=st.session_state.history,
        ans=float(st.session_state.ans),
        memory=float(st.session_state.memory),
        angle_mode=st.session_state.angle_mode,
        reply_nonce=st.session_state.reply_nonce,
        display_expr=st.session_state.display_expr,
        data_version=int(st.session_state.data_version),
        key="burhan_sci_calc",
        default=None,
    )

    if _handle_event(event):
        st.rerun()


if __name__ == "__main__":
    main()
