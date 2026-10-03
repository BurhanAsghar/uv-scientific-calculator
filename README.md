# Scientific Calculator (Streamlit)

A scientific calculator built with **Streamlit** and Python's **math** library.

## Features

- Click any button on the keypad
- Use your physical keyboard (numbers, operators, Enter, Backspace, Esc)
- Type a full expression and press Enter
- Scientific functions: trig, hyperbolic, logs, roots, factorial, and more
- RAD / DEG angle mode
- Calculation history
- Safe evaluation (AST-based — no arbitrary `eval`)

## Run

```bash
# From the project folder (with uv)
uv sync
uv run streamlit run main.py
```

Or with a normal venv:

```bash
pip install streamlit
streamlit run main.py
```

Then open the URL shown in the terminal (usually http://localhost:8501).

## Keyboard + mouse

**Keyboard**
1. Click the expression box
2. Type your calculation (example: `12+5` or `sin(pi/4)`)
3. Click **Calculate (=)**

**Mouse**
- Click keypad buttons to build the expression
- Click **=** (or **Calculate**) to evaluate

Both methods use the same expression box and result display.

## Example expressions

```text
sin(pi/4) + cos(pi/4)
sqrt(2) ** 2
log10(1000)
factorial(6)
(2 + 3) * 4
```
