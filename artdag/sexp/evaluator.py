"""
Expression evaluator for S-expression DSL.

Supports:
- Arithmetic: +, -, *, /, mod, sqrt, pow, abs, floor, ceil, round, min, max, clamp
- Comparison: =, <, >, <=, >=
- Logic: and, or, not
- Predicates: odd?, even?, zero?, nil?
- Conditionals: if, cond, case
- Data: list, dict/map construction, get
- Lambda calls
"""

from typing import Any, Dict, List, Callable
from .parser import Symbol, Keyword, Lambda, Binding


class EvalError(Exception):
    """Error during expression evaluation."""
    pass


# Built-in functions
BUILTINS: Dict[str, Callable] = {}


def builtin(name: str):
    """Decorator to register a builtin function."""
    def decorator(fn):
        BUILTINS[name] = fn
        return fn
    return decorator


@builtin("+")
def add(*args):
    return sum(args)


@builtin("-")
def sub(a, b=None):
    if b is None:
        return -a
    return a - b


@builtin("*")
def mul(*args):
    result = 1
    for a in args:
        result *= a
    return result


@builtin("/")
def div(a, b):
    return a / b


@builtin("mod")
def mod(a, b):
    return a % b


@builtin("sqrt")
def sqrt(x):
    return x ** 0.5


@builtin("pow")
def power(x, n):
    return x ** n


@builtin("abs")
def absolute(x):
    return abs(x)


@builtin("floor")
def floor_fn(x):
    import math
    return math.floor(x)


@builtin("ceil")
def ceil_fn(x):
    import math
    return math.ceil(x)


@builtin("round")
def round_fn(x, ndigits=0):
    return round(x, int(ndigits))


@builtin("min")
def min_fn(*args):
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        return min(args[0])
    return min(args)


@builtin("max")
def max_fn(*args):
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        return max(args[0])
    return max(args)


@builtin("clamp")
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


@builtin("=")
def eq(a, b):
    return a == b


@builtin("<")
def lt(a, b):
    return a < b


@builtin(">")
def gt(a, b):
    return a > b


@builtin("<=")
def lte(a, b):
    return a <= b


@builtin(">=")
def gte(a, b):
    return a >= b


@builtin("odd?")
def is_odd(n):
    return n % 2 == 1


@builtin("even?")
def is_even(n):
    return n % 2 == 0


@builtin("zero?")
def is_zero(n):
    return n == 0


@builtin("nil?")
def is_nil(x):
    return x is None


@builtin("not")
def not_fn(x):
    return not x


@builtin("inc")
def inc(n):
    return n + 1


@builtin("dec")
def dec(n):
    return n - 1


@builtin("list")
def make_list(*args):
    return list(args)


@builtin("assert")
def assert_true(condition, message="Assertion failed"):
    if not condition:
        raise RuntimeError(f"Assertion error: {message}")
    return True


@builtin("get")
def get(coll, key, default=None):
    if isinstance(coll, dict):
        # Try the key directly first
        result = coll.get(key, None)
        if result is not None:
            return result
        # If key is a Keyword, also try its string name (for Python dicts with string keys)
        if isinstance(key, Keyword):
            result = coll.get(key.name, None)
            if result is not None:
                return result
        # Return the default
        return default
    elif isinstance(coll, list):
        return coll[key] if 0 <= key < len(coll) else default
    else:
        raise EvalError(f"get: expected dict or list, got {type(coll).__name__}: {str(coll)[:100]}")


@builtin("dict?")
def is_dict(x):
    return isinstance(x, dict)


@builtin("list?")
def is_list(x):
    return isinstance(x, list)


@builtin("nil?")
def is_nil(x):
    return x is None


@builtin("number?")
def is_number(x):
    return isinstance(x, (int, float))


@builtin("string?")
def is_string(x):
    return isinstance(x, str)


@builtin("len")
def length(coll):
    return len(coll)


@builtin("first")
def first(coll):
    return coll[0] if coll else None


@builtin("last")
def last(coll):
    return coll[-1] if coll else None


@builtin("chunk-every")
def chunk_every(coll, n):
    """Split collection into chunks of n elements."""
    n = int(n)
    return [coll[i:i+n] for i in range(0, len(coll), n)]


@builtin("rest")
def rest(coll):
    return coll[1:] if coll else []


@builtin("nth")
def nth(coll, n):
    return coll[n] if 0 <= n < len(coll) else None


@builtin("concat")
def concat(*colls):
    """Concatenate multiple lists/sequences."""
    result = []
    for c in colls:
        if c is not None:
            result.extend(c)
    return result


@builtin("cons")
def cons(x, coll):
    """Prepend x to collection."""
    return [x] + list(coll) if coll else [x]


@builtin("append")
def append(coll, x):
    """Append x to collection."""
    return list(coll) + [x] if coll else [x]


@builtin("range")
def make_range(start, end, step=1):
    """Create a range of numbers."""
    return list(range(int(start), int(end), int(step)))


@builtin("zip-pairs")
def zip_pairs(coll):
    """Zip consecutive pairs: [a,b,c,d] -> [[a,b],[b,c],[c,d]]."""
    if not coll or len(coll) < 2:
        return []
    return [[coll[i], coll[i+1]] for i in range(len(coll)-1)]


@builtin("dict")
def make_dict(*pairs):
    """Create dict from key-value pairs: (dict :a 1 :b 2)."""
    result = {}
    i = 0
    while i < len(pairs) - 1:
        key = pairs[i]
        if isinstance(key, Keyword):
            key = key.name
        result[key] = pairs[i + 1]
        i += 2
    return result


def evaluate(expr: Any, env: Dict[str, Any] = None) -> Any:
    """
    Evaluate an S-expression in the given environment.

    Args:
        expr: The expression to evaluate
        env: Variable bindings (name -> value)

    Returns:
        The result of evaluation
    """
    if env is None:
        env = {}

    # Literals
    if isinstance(expr, (int, float, str, bool)) or expr is None:
        return expr

    # Symbol - variable lookup
    if isinstance(expr, Symbol):
        name = expr.name
        if name in env:
            return env[name]
        if name in BUILTINS:
            return BUILTINS[name]
        if name == "true":
            return True
        if name == "false":
            return False
        if name == "nil":
            return None
        raise EvalError(f"Undefined symbol: {name}")

    # Keyword - return as-is (used as map keys)
    if isinstance(expr, Keyword):
        return expr.name

    # Lambda - return as-is (it's a value)
    if isinstance(expr, Lambda):
        return expr

    # Binding - return as-is (resolved at execution time)
    if isinstance(expr, Binding):
        return expr

    # Dict literal
    if isinstance(expr, dict):
        return {k: evaluate(v, env) for k, v in expr.items()}

    # List - function call or special form
    if isinstance(expr, list):
        if not expr:
            return []

        head = expr[0]

        # If head is a string/number/etc (not Symbol), treat as data list
        if not isinstance(head, (Symbol, Lambda, list)):
            return [evaluate(x, env) for x in expr]

        # Special forms
        if isinstance(head, Symbol):
            name = head.name

            # if - conditional
            if name == "if":
                if len(expr) < 3:
                    raise EvalError("if requires condition and then-branch")
                cond_result = evaluate(expr[1], env)
                if cond_result:
                    return evaluate(expr[2], env)
                elif len(expr) > 3:
                    return evaluate(expr[3], env)
                return None

            # cond - multi-way conditional
            # Supports both Clojure style: (cond test1 result1 test2 result2 :else default)
            # and Scheme style: (cond (test1 result1) (test2 result2) (else default))
            if name == "cond":
                clauses = expr[1:]
                # Check if Clojure style (flat list) or Scheme style (nested pairs)
                # Scheme style: first clause is (test result) - exactly 2 elements
                # Clojure style: first clause is a test expression like (= x 0) - 3+ elements
                first_is_scheme_clause = (
                    clauses and
                    isinstance(clauses[0], list) and
                    len(clauses[0]) == 2 and
                    not (isinstance(clauses[0][0], Symbol) and clauses[0][0].name in ('=', '<', '>', '<=', '>=', '!=', 'not=', 'and', 'or'))
                )
                if first_is_scheme_clause:
                    # Scheme style: ((test result) ...)
                    for clause in clauses:
                        if not isinstance(clause, list) or len(clause) < 2:
                            raise EvalError("cond clause must be (test result)")
                        test = clause[0]
                        # Check for else/default
                        if isinstance(test, Symbol) and test.name in ("else", ":else"):
                            return evaluate(clause[1], env)
                        if isinstance(test, Keyword) and test.name == "else":
                            return evaluate(clause[1], env)
                        if evaluate(test, env):
                            return evaluate(clause[1], env)
                else:
                    # Clojure style: test1 result1 test2 result2 ...
                    i = 0
                    while i < len(clauses) - 1:
                        test = clauses[i]
                        result = clauses[i + 1]
                        # Check for :else
                        if isinstance(test, Keyword) and test.name == "else":
                            return evaluate(result, env)
                        if isinstance(test, Symbol) and test.name == ":else":
                            return evaluate(result, env)
                        if evaluate(test, env):
                            return evaluate(result, env)
                        i += 2
                return None

            # case - switch on value
            # (case expr val1 result1 val2 result2 :else default)
            if name == "case":
                if len(expr) < 2:
                    raise EvalError("case requires expression to match")
                match_val = evaluate(expr[1], env)
                clauses = expr[2:]
                i = 0
                while i < len(clauses) - 1:
                    test = clauses[i]
                    result = clauses[i + 1]
                    # Check for :else / else
                    if isinstance(test, Keyword) and test.name == "else":
                        return evaluate(result, env)
                    if isinstance(test, Symbol) and test.name in (":else", "else"):
                        return evaluate(result, env)
                    # Evaluate test value and compare
                    test_val = evaluate(test, env)
                    if match_val == test_val:
                        return evaluate(result, env)
                    i += 2
                return None

            # and - short-circuit
            if name == "and":
                result = True
                for arg in expr[1:]:
                    result = evaluate(arg, env)
                    if not result:
                        return result
                return result

            # or - short-circuit
            if name == "or":
                result = False
                for arg in expr[1:]:
                    result = evaluate(arg, env)
                    if result:
                        return result
                return result

            # let and let* - local bindings (both bind sequentially in this impl)
            if name in ("let", "let*"):
                if len(expr) < 3:
                    raise EvalError(f"{name} requires bindings and body")
                bindings = expr[1]

                local_env = dict(env)

                if isinstance(bindings, list):
                    # Check if it's ((name value) ...) style (Lisp let* style)
                    if bindings and isinstance(bindings[0], list):
                        for binding in bindings:
                            if len(binding) != 2:
                                raise EvalError(f"{name} binding must be (name value)")
                            var_name = binding[0]
                            if isinstance(var_name, Symbol):
                                var_name = var_name.name
                            value = evaluate(binding[1], local_env)
                            local_env[var_name] = value
                    # Vector-style [name value ...]
                    elif len(bindings) % 2 == 0:
                        for i in range(0, len(bindings), 2):
                            var_name = bindings[i]
                            if isinstance(var_name, Symbol):
                                var_name = var_name.name
                            value = evaluate(bindings[i + 1], local_env)
                            local_env[var_name] = value
                    else:
                        raise EvalError(f"{name} bindings must be [name value ...] or ((name value) ...)")
                else:
                    raise EvalError(f"{name} bindings must be a list")

                return evaluate(expr[2], local_env)

            # lambda / fn - create function with closure
            if name in ("lambda", "fn"):
                if len(expr) < 3:
                    raise EvalError("lambda requires params and body")
                params = expr[1]
                if not isinstance(params, list):
                    raise EvalError("lambda params must be a list")
                param_names = []
                for p in params:
                    if isinstance(p, Symbol):
                        param_names.append(p.name)
                    elif isinstance(p, str):
                        param_names.append(p)
                    else:
                        raise EvalError(f"Invalid param: {p}")
                # Capture current environment as closure
                return Lambda(param_names, expr[2], dict(env))

            # quote - return unevaluated
            if name == "quote":
                return expr[1] if len(expr) > 1 else None

            # bind - create binding to analysis data
            # (bind analysis-var)
            # (bind analysis-var :range [0.3 1.0])
            # (bind analysis-var :range [0 100] :transform sqrt)
            if name == "bind":
                if len(expr) < 2:
                    raise EvalError("bind requires analysis reference")
                analysis_ref = expr[1]
                if isinstance(analysis_ref, Symbol):
                    symbol_name = analysis_ref.name
                    # Look up the symbol in environment
                    if symbol_name in env:
                        resolved = env[symbol_name]
                        # If resolved is actual analysis data (dict with times/values or
                        # S-expression list with Keywords), keep the symbol name as reference
                        # for later lookup at execution time
                        if isinstance(resolved, dict) and ("times" in resolved or "values" in resolved):
                            analysis_ref = symbol_name  # Use name as reference, not the data
                        elif isinstance(resolved, list) and any(isinstance(x, Keyword) for x in resolved):
                            # Parsed S-expression analysis data ([:times [...] :duration ...])
                            analysis_ref = symbol_name
                        else:
                            analysis_ref = resolved
                    else:
                        raise EvalError(f"bind: undefined symbol '{symbol_name}' - must reference analysis data")

                # Parse optional :range [min max] and :transform
                range_min, range_max = 0.0, 1.0
                transform = None
                i = 2
                while i < len(expr):
                    if isinstance(expr[i], Keyword):
                        kw = expr[i].name
                        if kw == "range" and i + 1 < len(expr):
                            range_val = evaluate(expr[i + 1], env)  # Evaluate to get actual value
                            if isinstance(range_val, list) and len(range_val) >= 2:
                                range_min = float(range_val[0])
                                range_max = float(range_val[1])
                            i += 2
                        elif kw == "transform" and i + 1 < len(expr):
                            t = expr[i + 1]
                            if isinstance(t, Symbol):
                                transform = t.name
                            elif isinstance(t, str):
                                transform = t
                            i += 2
                        else:
                            i += 1
                    else:
                        i += 1

                return Binding(analysis_ref, range_min=range_min, range_max=range_max, transform=transform)

            # Vector literal [a b c]
            if name == "vec" or name == "vector":
                return [evaluate(e, env) for e in expr[1:]]

            # map - (map fn coll)
            if name == "map":
                if len(expr) != 3:
                    raise EvalError("map requires fn and collection")
                fn = evaluate(expr[1], env)
                coll = evaluate(expr[2], env)
                if not isinstance(fn, Lambda):
                    raise EvalError(f"map requires lambda, got {type(fn)}")
                result = []
                for item in coll:
                    local_env = {}
                    if fn.closure:
                        local_env.update(fn.closure)
                    local_env.update(env)
                    local_env[fn.params[0]] = item
                    result.append(evaluate(fn.body, local_env))
                return result

            # map-indexed - (map-indexed fn coll)
            if name == "map-indexed":
                if len(expr) != 3:
                    raise EvalError("map-indexed requires fn and collection")
                fn = evaluate(expr[1], env)
                coll = evaluate(expr[2], env)
                if not isinstance(fn, Lambda):
                    raise EvalError(f"map-indexed requires lambda, got {type(fn)}")
                if len(fn.params) < 2:
                    raise EvalError("map-indexed lambda needs (i item) params")
                result = []
                for i, item in enumerate(coll):
                    local_env = {}
                    if fn.closure:
                        local_env.update(fn.closure)
                    local_env.update(env)
                    local_env[fn.params[0]] = i
                    local_env[fn.params[1]] = item
                    result.append(evaluate(fn.body, local_env))
                return result

            # reduce - (reduce fn init coll)
            if name == "reduce":
                if len(expr) != 4:
                    raise EvalError("reduce requires fn, init, and collection")
                fn = evaluate(expr[1], env)
                acc = evaluate(expr[2], env)
                coll = evaluate(expr[3], env)
                if not isinstance(fn, Lambda):
                    raise EvalError(f"reduce requires lambda, got {type(fn)}")
                if len(fn.params) < 2:
                    raise EvalError("reduce lambda needs (acc item) params")
                for item in coll:
                    local_env = {}
                    if fn.closure:
                        local_env.update(fn.closure)
                    local_env.update(env)
                    local_env[fn.params[0]] = acc
                    local_env[fn.params[1]] = item
                    acc = evaluate(fn.body, local_env)
                return acc

        # Function call
        fn = evaluate(head, env)
        args = [evaluate(arg, env) for arg in expr[1:]]

        # Call builtin
        if callable(fn):
            return fn(*args)

        # Call lambda
        if isinstance(fn, Lambda):
            if len(args) != len(fn.params):
                raise EvalError(f"Lambda expects {len(fn.params)} args, got {len(args)}")
            # Start with closure (captured env), then overlay calling env, then params
            local_env = {}
            if fn.closure:
                local_env.update(fn.closure)
            local_env.update(env)
            for name, value in zip(fn.params, args):
                local_env[name] = value
            return evaluate(fn.body, local_env)

        raise EvalError(f"Not callable: {fn}")

    raise EvalError(f"Cannot evaluate: {expr!r}")


def make_env(**kwargs) -> Dict[str, Any]:
    """Create an environment with initial bindings."""
    return dict(kwargs)
