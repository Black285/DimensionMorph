"""Safe evaluation of user-supplied mathematical expressions.

Expressions are validated against an AST allow-list before they are compiled,
so the evaluator never sees attribute access, imports, lambdas or any name that
is not an approved function, constant, variable or user parameter.  Compiled
code objects are cached, which matters because dragging a parameter slider
re-evaluates the same expression hundreds of times per second.
"""

from __future__ import annotations

import ast
import functools
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

MAX_EXPRESSION_LENGTH = 1024
MAX_CONSTANT_EXPONENT = 64


class ExpressionError(ValueError):
    """Raised when an expression is rejected or fails to evaluate."""


def _is_iterable(value: Any) -> bool:
    return hasattr(value, '__iter__') and not isinstance(value, (str, bytes, np.ndarray))


def _safe_sum(*args: Any) -> Any:
    """sum(array) or sum(generator_of_arrays) -- numpy-friendly either way."""
    values = list(args[0]) if len(args) == 1 and _is_iterable(args[0]) else list(args)
    if len(values) == 1 and isinstance(values[0], np.ndarray):
        return np.sum(values[0])
    if not values:
        return 0
    return functools.reduce(lambda a, b: a + b, values)


def _safe_prod(*args: Any) -> Any:
    values = list(args[0]) if len(args) == 1 and _is_iterable(args[0]) else list(args)
    if len(values) == 1 and isinstance(values[0], np.ndarray):
        return np.prod(values[0])
    if not values:
        return 1
    return functools.reduce(lambda a, b: a * b, values)


def _erf(value: Any) -> Any:
    return np.vectorize(lambda v: math.erf(float(v)), otypes=[float])(value)


def _gamma(value: Any) -> Any:
    def one(v: float) -> float:
        try:
            return math.gamma(float(v))
        except (ValueError, OverflowError):
            return float('nan')

    return np.vectorize(one, otypes=[float])(value)


FUNCTIONS: dict[str, Any] = {
    # Trigonometric
    'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
    'asin': np.arcsin, 'arcsin': np.arcsin,
    'acos': np.arccos, 'arccos': np.arccos,
    'atan': np.arctan, 'arctan': np.arctan, 'atan2': np.arctan2,
    'sec': lambda v: 1.0 / np.cos(v),
    'csc': lambda v: 1.0 / np.sin(v),
    'cot': lambda v: 1.0 / np.tan(v),
    # Hyperbolic
    'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
    'asinh': np.arcsinh, 'arcsinh': np.arcsinh,
    'acosh': np.arccosh, 'arccosh': np.arccosh,
    'atanh': np.arctanh, 'arctanh': np.arctanh,
    # Exponential and logarithmic
    'exp': np.exp, 'log': np.log, 'ln': np.log,
    'log10': np.log10, 'log2': np.log2, 'sqrt': np.sqrt,
    'cbrt': np.cbrt, 'pow': np.power, 'power': np.power,
    # Rounding and magnitude
    'abs': np.abs, 'ceil': np.ceil, 'floor': np.floor,
    'round': np.round, 'sign': np.sign, 'trunc': np.trunc,
    'mod': np.mod, 'fmod': np.fmod, 'hypot': np.hypot,
    # Complex
    're': np.real, 'real': np.real, 'im': np.imag, 'imag': np.imag,
    'conj': np.conj, 'arg': np.angle, 'angle': np.angle,
    # Comparison and selection
    'min': np.minimum, 'max': np.maximum,
    'clamp': np.clip, 'clip': np.clip, 'where': np.where,
    # Aggregation over series
    'sum': _safe_sum, 'prod': _safe_prod, 'mul': _safe_prod,
    'diff': np.diff, 'dif': np.diff, 'range': range,
    # Special
    'erf': _erf, 'gamma': _gamma, 'sinc': np.sinc,
    'heaviside': lambda v, h=0.5: np.heaviside(v, h),
    # Constants
    'pi': np.pi, 'tau': 2.0 * np.pi, 'e': np.e,
    'phi': (1.0 + np.sqrt(5.0)) / 2.0, 'i': 1j, 'j': 1j, 'inf': np.inf,
}

_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Tuple, ast.List, ast.Compare, ast.BoolOp, ast.IfExp,
    ast.GeneratorExp, ast.ListComp, ast.comprehension, ast.keyword, ast.Store,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or, ast.Invert,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)


def _comprehension_targets(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.comprehension):
            for target in ast.walk(node.target):
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def _check_constant_power(node: ast.BinOp) -> None:
    """Reject 10**10**10 and friends, which would hang on big-int arithmetic."""
    if not isinstance(node.op, ast.Pow):
        return
    exponent = node.right
    if isinstance(exponent, ast.BinOp) and isinstance(exponent.op, ast.Pow):
        raise ExpressionError('nested constant powers are not allowed')
    if (isinstance(exponent, ast.Constant) and
            isinstance(exponent.value, (int, float)) and
            isinstance(node.left, ast.Constant) and
            abs(exponent.value) > MAX_CONSTANT_EXPONENT):
        raise ExpressionError(
            f'exponent {exponent.value} is too large (limit {MAX_CONSTANT_EXPONENT})'
        )


def validate(source: str, allowed_names: Iterable[str]) -> ast.Expression:
    """Parse and check an expression, returning its AST."""
    source = source.strip()
    if not source:
        raise ExpressionError('expression is empty')
    if len(source) > MAX_EXPRESSION_LENGTH:
        raise ExpressionError(f'expression exceeds {MAX_EXPRESSION_LENGTH} characters')

    try:
        tree = ast.parse(source, mode='eval')
    except SyntaxError as exc:
        raise ExpressionError(f'syntax error: {exc.msg}') from exc

    permitted = set(allowed_names) | set(FUNCTIONS) | _comprehension_targets(tree)

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ExpressionError(f'{type(node).__name__} is not allowed in expressions')
        if isinstance(node, ast.BinOp):
            _check_constant_power(node)
        if isinstance(node, ast.Name):
            if node.id.startswith('_'):
                raise ExpressionError('names starting with "_" are not allowed')
            if node.id not in permitted:
                raise ExpressionError(f'unknown name "{node.id}"')
    return tree


@functools.lru_cache(maxsize=512)
def _compile(source: str, allowed_names: tuple[str, ...]):
    return compile(validate(source, allowed_names), '<expression>', 'eval')


def evaluate(
    source: str,
    variables: Mapping[str, Any] | None = None,
    parameters: Mapping[str, Any] | None = None,
) -> Any:
    """Evaluate `source` with the given variables and user parameters."""
    variables = variables or {}
    parameters = parameters or {}
    code = _compile(source, tuple(sorted(set(variables) | set(parameters))))

    # Everything goes into globals rather than locals: a generator expression
    # body cannot see an eval() locals mapping, so `sum(sin(i*x) for i in ...)`
    # would fail with a NameError if the functions lived in locals.
    env: dict[str, Any] = {'__builtins__': {}}
    env.update(FUNCTIONS)
    env.update(parameters)
    env.update(variables)

    with np.errstate(all='ignore'):
        try:
            return eval(code, env)  # noqa: S307 - AST-validated
        except ExpressionError:
            raise
        except ZeroDivisionError:
            return np.nan
        except Exception as exc:
            raise ExpressionError(f'could not evaluate "{source}": {exc}') from exc


def as_array(value: Any, like: np.ndarray, dtype: Any = np.float64) -> np.ndarray:
    """Broadcast an evaluation result to the shape of `like`."""
    if np.isscalar(value) or (isinstance(value, np.ndarray) and value.ndim == 0):
        return np.full(like.shape, value, dtype=dtype)
    array: np.ndarray = np.asarray(value)
    if dtype is np.float64 and np.iscomplexobj(array):
        array = np.real(array)  # type: ignore[arg-type]
    array = array.astype(dtype, copy=False)
    if array.shape != like.shape:
        array = np.broadcast_to(array, like.shape).copy()
    return array


def evaluate_scalar(source: Any, parameters: Mapping[str, Any] | None = None) -> float:
    """Parse a range bound, which may be a number or an expression such as '2*pi'."""
    try:
        return float(source)
    except (TypeError, ValueError):
        pass
    array = np.asarray(evaluate(str(source), parameters=parameters), dtype=np.float64)
    if array.size != 1:
        raise ExpressionError(f'"{source}" is not a single number')
    value = float(array.reshape(-1)[0])
    if not np.isfinite(value):
        raise ExpressionError(f'"{source}" is not finite')
    return value


def function_catalog() -> list[dict[str, Sequence[str]]]:
    """Grouped function names, for the in-app reference panel."""
    return [
        {'group': 'Trigonometric',
         'names': ['sin', 'cos', 'tan', 'sec', 'csc', 'cot', 'asin', 'acos', 'atan', 'atan2']},
        {'group': 'Hyperbolic', 'names': ['sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh']},
        {'group': 'Exponential', 'names': ['exp', 'log', 'log2', 'log10', 'sqrt', 'cbrt', 'pow']},
        {'group': 'Rounding', 'names': ['abs', 'ceil', 'floor', 'round', 'sign', 'trunc', 'mod']},
        {'group': 'Complex', 'names': ['re', 'im', 'abs', 'arg', 'conj']},
        {'group': 'Selection', 'names': ['min', 'max', 'clamp', 'where', 'heaviside']},
        {'group': 'Series', 'names': ['sum', 'prod', 'diff', 'range']},
        {'group': 'Special', 'names': ['erf', 'gamma', 'sinc', 'hypot']},
        {'group': 'Constants', 'names': ['pi', 'tau', 'e', 'phi', 'i']},
    ]
