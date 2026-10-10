"""Unit tests for engine.py — safe scientific calculator evaluator."""

from __future__ import annotations

import math
import random
import string

import pytest

from engine import CalcError, calculate, format_result, parse, tokenize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def eq(a, b, tol=1e-10):
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


def calc(expr: str, mode: str = "RAD"):
    return calculate(expr, angle_mode=mode)


# ---------------------------------------------------------------------------
# Basic arithmetic & precedence
# ---------------------------------------------------------------------------

class TestArithmetic:
    def test_add(self):
        assert calc("2+3") == 5

    def test_sub(self):
        assert calc("10-4") == 6

    def test_mul(self):
        assert calc("3*4") == 12

    def test_div(self):
        assert calc("8/2") == 4

    def test_precedence_mul_over_add(self):
        assert calc("2+3*4") == 14

    def test_parentheses(self):
        assert calc("(2+3)*4") == 20

    def test_power(self):
        assert calc("2^3") == 8

    def test_power_right_assoc(self):
        assert calc("2^3^2") == 512  # 2^(3^2)

    def test_unicode_ops(self):
        assert calc("6×7") == 42
        assert calc("8÷2") == 4

    def test_float_add_clean(self):
        assert calc("0.1+0.2") == 0.3 or eq(calc("0.1+0.2"), 0.3)

    def test_chained(self):
        assert calc("1+2+3+4") == 10


class TestUnaryMinus:
    def test_neg_power(self):
        # -2^2 = -(2^2) = -4
        assert calc("-2^2") == -4

    def test_power_neg_exp(self):
        assert eq(calc("2^-3"), 0.125)

    def test_double_neg(self):
        assert calc("--5") == 5

    def test_neg_paren(self):
        assert calc("-(3+4)") == -7

    def test_mul_neg(self):
        assert calc("3*-2") == -6


class TestImplicitMultiplication:
    def test_paren(self):
        assert calc("2(3+4)") == 14

    def test_pi(self):
        assert eq(calc("2pi"), 2 * math.pi)

    def test_juxtapose_funcs(self):
        assert eq(calc("3sin(0)"), 0)

    def test_close_open(self):
        assert calc("(2+1)(3+1)") == 12

    def test_space_number_ident(self):
        assert eq(calc("2 pi"), 2 * math.pi)


class TestPercentFactorial:
    def test_percent(self):
        assert eq(calc("50%"), 0.5)

    def test_factorial(self):
        assert calc("5!") == 120

    def test_factorial_zero(self):
        assert calc("0!") == 1

    def test_factorial_func(self):
        assert calc("factorial(6)") == 720


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

class TestTrigRad:
    def test_sin_pi2(self):
        assert calc("sin(pi/2)") == 1

    def test_cos_0(self):
        assert calc("cos(0)") == 1

    def test_tan_0(self):
        assert calc("tan(0)") == 0

    def test_asin(self):
        assert eq(calc("asin(0)"), 0)

    def test_acos(self):
        assert eq(calc("acos(1)"), 0)

    def test_atan(self):
        assert eq(calc("atan(0)"), 0)


class TestTrigDeg:
    def test_sin_30(self):
        assert calc("sin(30)", "DEG") == 0.5 or eq(calc("sin(30)", "DEG"), 0.5)

    def test_cos_60(self):
        assert eq(calc("cos(60)", "DEG"), 0.5)

    def test_sin_90(self):
        assert calc("sin(90)", "DEG") == 1

    def test_cos_90_noise(self):
        assert calc("cos(90)", "DEG") == 0
        assert format_result(calc("cos(90)", "DEG")) == "0"

    def test_trig_no_signed_zero(self):
        for expr, mode in [
            ("cos(90)", "DEG"),
            ("sin(180)", "DEG"),
            ("sin(pi)", "RAD"),
            ("cos(pi/2)", "RAD"),
            ("tan(180)", "DEG"),
            ("cos(270)", "DEG"),
        ]:
            v = calc(expr, mode)
            assert v == 0
            assert format_result(v) == "0"

    def test_tan_45(self):
        assert eq(calc("tan(45)", "DEG"), 1)


class TestHyperbolicLogs:
    def test_sinh_0(self):
        assert calc("sinh(0)") == 0

    def test_cosh_0(self):
        assert calc("cosh(0)") == 1

    def test_tanh_0(self):
        assert calc("tanh(0)") == 0

    def test_log10(self):
        assert calc("log(100)") == 2

    def test_ln_e(self):
        assert calc("ln(e)") == 1

    def test_log2(self):
        assert calc("log2(8)") == 3

    def test_exp_0(self):
        assert calc("exp(0)") == 1


class TestRootsAbsRound:
    def test_sqrt(self):
        assert calc("sqrt(9)") == 3

    def test_cbrt(self):
        assert calc("cbrt(8)") == 2

    def test_abs(self):
        assert calc("abs(-7)") == 7

    def test_floor(self):
        assert calc("floor(3.9)") == 3

    def test_ceil(self):
        assert calc("ceil(3.1)") == 4

    def test_round(self):
        assert calc("round(3.5)") == 4 or calc("round(2.5)") in (2, 3)


class TestCombinatorics:
    def test_npr(self):
        assert calc("nPr(5,2)") == 20

    def test_ncr(self):
        assert calc("nCr(5,2)") == 10


class TestConstantsAnsMemory:
    def test_pi(self):
        assert eq(calc("pi"), math.pi)

    def test_e(self):
        assert eq(calc("e"), math.e)

    def test_ans(self):
        assert calculate("ans+1", ans=10) == 11

    def test_mr(self):
        assert calculate("mr*2", memory=5) == 10


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class TestErrors:
    def test_empty(self):
        with pytest.raises(CalcError, match="Empty"):
            calc("   ")

    def test_div_zero(self):
        with pytest.raises(CalcError, match="Division by zero"):
            calc("1/0")

    def test_sqrt_neg(self):
        with pytest.raises(CalcError, match="sqrt"):
            calc("sqrt(-1)")

    def test_log_zero(self):
        with pytest.raises(CalcError, match="log"):
            calc("log(0)")

    def test_ln_neg(self):
        with pytest.raises(CalcError, match="ln"):
            calc("ln(-2)")

    def test_asin_domain(self):
        with pytest.raises(CalcError, match="asin"):
            calc("asin(2)")

    def test_tan_90_deg(self):
        with pytest.raises(CalcError, match="tan"):
            calc("tan(90)", "DEG")

    def test_unbalanced_extra_close(self):
        with pytest.raises(CalcError):
            calc("2+3)")

    def test_double_plus(self):
        with pytest.raises(CalcError):
            calc("2++3")

    def test_trailing_op(self):
        with pytest.raises(CalcError):
            calc("5+")

    def test_double_dot(self):
        with pytest.raises(CalcError):
            calc("1..2")

    def test_space_numbers(self):
        with pytest.raises(CalcError):
            calc("5 5")

    def test_fact_neg(self):
        with pytest.raises(CalcError, match="Factorial"):
            calc("(-3)!")

    def test_fact_too_large(self):
        with pytest.raises(CalcError, match="too large"):
            calc("200!")

    def test_unknown_func(self):
        with pytest.raises(CalcError, match="Unknown"):
            calc("foo(1)")

    def test_npr_bad(self):
        with pytest.raises(CalcError):
            calc("nPr(2,5)")


# ---------------------------------------------------------------------------
# Formatting & modes
# ---------------------------------------------------------------------------

class TestFormatAndMode:
    def test_format_int(self):
        assert format_result(5.0) == "5"

    def test_grad_sin(self):
        # 100 grad = 90 deg = π/2 rad
        assert calc("sin(100)", "GRAD") == 1

    def test_bad_mode(self):
        with pytest.raises(CalcError):
            calculate("1", angle_mode="FOO")


# ---------------------------------------------------------------------------
# Fuzz — must never raise anything except CalcError
# ---------------------------------------------------------------------------

class TestFuzz:
    def test_random_strings_no_crash(self):
        alphabet = string.ascii_letters + string.digits + "+-*/^()%!,. " + "π×÷"
        rng = random.Random(42)
        for _ in range(200):
            s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
            try:
                calculate(s)
            except CalcError:
                pass
            except Exception as exc:  # noqa: BLE001
                pytest.fail(f"Unhandled {type(exc).__name__} for {s!r}: {exc}")

    def test_tokenize_empty(self):
        with pytest.raises(CalcError):
            tokenize("")

    def test_parse_simple(self):
        assert parse("1+2") is not None


# ---------------------------------------------------------------------------
# Extra edge cases to push past 80 tests
# ---------------------------------------------------------------------------

class TestMoreEdges:
    def test_nested(self):
        assert calc("((2))") == 2

    def test_power_frac(self):
        assert eq(calc("9^0.5"), 3)

    def test_mixed(self):
        assert eq(calc("sin(pi/6)+cos(0)"), 1.5)

    def test_deg_asin(self):
        assert eq(calc("asin(0.5)", "DEG"), 30)

    def test_many_parens(self):
        assert calc("((((5))))") == 5

    def test_sub_neg(self):
        assert calc("5--2") == 7

    def test_div_frac(self):
        assert eq(calc("1/4"), 0.25)

    def test_log10_alias(self):
        assert calc("log10(1000)") == 3

    def test_abs_expr(self):
        assert calc("abs(3-10)") == 7

    def test_hypot(self):
        assert calc("hypot(3,4)") == 5

    def test_pow_func(self):
        assert calc("pow(2,10)") == 1024

    def test_percent_of_chain(self):
        assert eq(calc("200*10%"), 20)

    def test_sci_notation_input(self):
        assert calc("1e3") == 1000

    def test_leading_dot(self):
        assert eq(calc(".5*2"), 1)

    def test_trailing_dot(self):
        assert calc("2.") == 2


# ---------------------------------------------------------------------------
# Optional / missing brackets & bare function calls
# ---------------------------------------------------------------------------

class TestOptionalBrackets:
    def test_cos_open_no_close(self):
        assert eq(calc("cos(0"), 1)

    def test_cos_space_arg(self):
        assert eq(calc("cos 0"), 1)

    def test_cos1_bare(self):
        assert eq(calc("cos0"), 1)

    def test_sin_deg_bare(self):
        assert eq(calc("sin 90", "DEG"), 1)

    def test_sqrt_bare(self):
        assert calc("sqrt 9") == 3

    def test_nested_open(self):
        assert calc("(2+3") == 5

    def test_log_open(self):
        assert calc("log(100") == 2

    def test_chain_bare(self):
        assert eq(calc("sin(pi/2)+cos 0"), 2)
