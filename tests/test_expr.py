"""The evaluator must run real mathematics and refuse everything else."""

from __future__ import annotations

import numpy as np
import pytest

from dimensionmorph.expr import ExpressionError, as_array, evaluate, evaluate_scalar

X = np.linspace(-3.0, 3.0, 21)


def test_basic_arithmetic():
    assert evaluate('2 + 3 * 4') == 14


def test_variables_and_parameters():
    result = evaluate('sin(x) * a', {'x': X}, {'a': 2.0})
    np.testing.assert_allclose(result, np.sin(X) * 2.0)


def test_generator_expressions_see_the_function_table():
    # Regression: eval() locals are invisible inside a generator expression, so
    # this raised NameError until the environment was passed as globals.
    result = evaluate('sum(sin(i*x) for i in range(3))', {'x': X})
    np.testing.assert_allclose(result, sum(np.sin(i * X) for i in range(3)))


def test_prod_over_a_series():
    result = evaluate('prod(1 + i*x for i in range(1, 4))', {'x': X})
    expected = (1 + X) * (1 + 2 * X) * (1 + 3 * X)
    np.testing.assert_allclose(result, expected)


def test_complex_variable():
    z = np.array([1 + 1j, 2j])
    np.testing.assert_allclose(evaluate('z**2', {'z': z}), z ** 2)


def test_division_by_zero_is_not_an_error():
    result = evaluate('1/x', {'x': np.array([0.0, 2.0])})
    assert np.isinf(result[0]) and result[1] == 0.5


def test_scalar_expressions():
    assert evaluate_scalar('2*pi') == pytest.approx(2 * np.pi)
    assert evaluate_scalar(-4) == -4.0
    assert evaluate_scalar('-2') == -2.0


def test_scalar_rejects_arrays_and_infinities():
    with pytest.raises(ExpressionError):
        evaluate_scalar('log(0)')


@pytest.mark.parametrize('source', [
    '().__class__',
    '().__class__.__base__.__subclasses__()',
    "__import__('os')",
    'x.__class__',
    "open('secrets')",
    'globals()',
    'exec("1")',
    '(lambda: 1)()',
    '[c for c in x.__dir__()]',
    'x[0]',
    '10**10**10',
    '2**99999',
    'unknown_name',
    'f"{x}"',
])
def test_dangerous_expressions_are_refused(source):
    with pytest.raises(ExpressionError):
        evaluate(source, {'x': X})


def test_empty_and_oversized_expressions():
    with pytest.raises(ExpressionError):
        evaluate('', {'x': X})
    with pytest.raises(ExpressionError):
        evaluate('x+' * 600 + 'x', {'x': X})


def test_syntax_errors_are_reported_clearly():
    with pytest.raises(ExpressionError, match='syntax error'):
        evaluate('sin(x', {'x': X})


def test_as_array_broadcasts_scalars():
    np.testing.assert_array_equal(as_array(3.0, X), np.full(X.shape, 3.0))


def test_as_array_drops_the_imaginary_part_for_float_output():
    np.testing.assert_allclose(as_array(np.full(X.shape, 1 + 2j), X), np.ones(X.shape))
