# Burhan Asghar — Scientific Calculator

Industry-grade scientific calculator built with **Python**, **Streamlit**, and a
**bidirectional custom component** for focus-free keyboard + mouse input.

## Architecture

1. `frontend/` — Streamlit custom component (HTML/JS/CSS). Expression state and
   keyboard handling live here for instant UI response.
2. `main.py` — **App entry point** (`streamlit run main.py`). Thin wrapper that
   starts the Streamlit host in `app.py`.
3. `app.py` — Streamlit host. Declares the component, receives evaluate events,
   returns `{result|error, history, ans, memory}` with a nonce.
4. `engine.py` — Safe tokenizer → recursive-descent parser → evaluator.
   **No** `eval()` / `exec()`, **no** Streamlit imports (unit-testable).
5. Global `keydown` on `window` (once). Enter evaluates; Esc clears; no need to
   click an input first.

## Run

```bash
pip install -r requirements.txt
streamlit run main.py
```

Or with uv:

```bash
uv sync
uv run streamlit run main.py
```

Open the URL shown in the terminal (usually http://localhost:8501).

## Keyboard

| Key | Action |
|-----|--------|
| `0-9` `.` `+` `-` `*` `/` `^` `(` `)` `%` `!` `,` | Insert |
| `Enter` or `=` | Evaluate (ignored if empty / ignore key-repeat) |
| `Esc` | Clear expression/result/error (keeps history) |
| `Backspace` | Delete one token (`sin(` as one unit); repeat allowed |
| `Delete` | Delete forward one token |
| `←` `→` `Home` `End` | Move cursor |
| `s` `c` `t` `l` `n` `r` `p` `e` | sin / cos / tan / log / ln / sqrt / π / e |
| `d` | Toggle RAD → DEG → GRAD |
| `Ctrl+C` / `Ctrl+V` | Copy result / paste sanitized expression |

Mouse clicks and keyboard work at the same time. Buttons flash when their key is pressed.

## Tests

```bash
pytest tests/test_engine.py -v
```

## Known limitations

- **Python 3.14 + Streamlit custom components:** some environments have hit a
  fatal interpreter error when mounting `declare_component` (pyarrow/numpy
  interaction). If the app crashes on load, try Python 3.11–3.13, or run
  `pytest` to verify the engine while we wait on upstream fixes.
- Keyboard events are captured inside the component iframe. The page loads with
  the component focused; if another browser UI steals focus, click once on the
  calculator panel.
- Streamlit still does a full script rerun when evaluation results return from
  Python; expression editing itself stays instant in JS.
- GRAD mode is supported in the engine and UI toggle.
