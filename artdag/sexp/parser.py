"""
S-expression parser for ArtDAG recipes and plans.

Supports:
- Lists: (a b c)
- Symbols: foo, bar-baz, ->
- Keywords: :key
- Strings: "hello world"
- Numbers: 42, 3.14, -1.5
- Comments: ; to end of line
- Vectors: [a b c] (syntactic sugar for lists)
- Maps: {:key1 val1 :key2 val2} (parsed as Python dicts)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import re


@dataclass
class Symbol:
    """An unquoted symbol/identifier."""
    name: str

    def __repr__(self):
        return f"Symbol({self.name!r})"

    def __eq__(self, other):
        if isinstance(other, Symbol):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other
        return False

    def __hash__(self):
        return hash(self.name)


@dataclass
class Keyword:
    """A keyword starting with colon."""
    name: str

    def __repr__(self):
        return f"Keyword({self.name!r})"

    def __eq__(self, other):
        if isinstance(other, Keyword):
            return self.name == other.name
        return False

    def __hash__(self):
        return hash((':' , self.name))


@dataclass
class Lambda:
    """A lambda/anonymous function with closure."""
    params: List[str]  # Parameter names
    body: Any  # Expression body
    closure: Dict = None  # Captured environment (optional for backwards compat)

    def __repr__(self):
        return f"Lambda({self.params}, {self.body!r})"


@dataclass
class Binding:
    """A binding to analysis data for dynamic effect parameters."""
    analysis_ref: str  # Name of analysis variable
    track: str = None  # Optional track name (e.g., "bass", "energy")
    range_min: float = 0.0  # Output range minimum
    range_max: float = 1.0  # Output range maximum
    transform: str = None  # Optional transform: "sqrt", "pow2", "log", etc.

    def __repr__(self):
        t = f", transform={self.transform!r}" if self.transform else ""
        return f"Binding({self.analysis_ref!r}, track={self.track!r}, range=[{self.range_min}, {self.range_max}]{t})"


class ParseError(Exception):
    """Error during S-expression parsing."""
    def __init__(self, message: str, position: int = 0, line: int = 1, col: int = 1):
        self.position = position
        self.line = line
        self.col = col
        super().__init__(f"{message} at line {line}, column {col}")


class Tokenizer:
    """Tokenize S-expression text into tokens."""

    # Token patterns
    WHITESPACE = re.compile(r'\s+')
    COMMENT = re.compile(r';[^\n]*')
    STRING = re.compile(r'"(?:[^"\\]|\\.)*"')
    NUMBER = re.compile(r'-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?')
    KEYWORD = re.compile(r':[a-zA-Z_][a-zA-Z0-9_-]*')
    SYMBOL = re.compile(r'[a-zA-Z_*+\-><=/!?][a-zA-Z0-9_*+\-><=/!?.]*')

    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.line = 1
        self.col = 1

    def _advance(self, count: int = 1):
        """Advance position, tracking line/column."""
        for _ in range(count):
            if self.pos < len(self.text):
                if self.text[self.pos] == '\n':
                    self.line += 1
                    self.col = 1
                else:
                    self.col += 1
                self.pos += 1

    def _skip_whitespace_and_comments(self):
        """Skip whitespace and comments."""
        while self.pos < len(self.text):
            # Whitespace
            match = self.WHITESPACE.match(self.text, self.pos)
            if match:
                self._advance(match.end() - self.pos)
                continue

            # Comments
            match = self.COMMENT.match(self.text, self.pos)
            if match:
                self._advance(match.end() - self.pos)
                continue

            break

    def peek(self) -> str | None:
        """Peek at current character."""
        self._skip_whitespace_and_comments()
        if self.pos >= len(self.text):
            return None
        return self.text[self.pos]

    def next_token(self) -> Any:
        """Get the next token."""
        self._skip_whitespace_and_comments()

        if self.pos >= len(self.text):
            return None

        char = self.text[self.pos]
        start_line, start_col = self.line, self.col

        # Single-character tokens (parens, brackets, braces)
        if char in '()[]{}':
            self._advance()
            return char

        # String
        if char == '"':
            match = self.STRING.match(self.text, self.pos)
            if not match:
                raise ParseError("Unterminated string", self.pos, self.line, self.col)
            self._advance(match.end() - self.pos)
            # Parse escape sequences
            content = match.group()[1:-1]
            content = content.replace('\\n', '\n')
            content = content.replace('\\t', '\t')
            content = content.replace('\\"', '"')
            content = content.replace('\\\\', '\\')
            return content

        # Keyword
        if char == ':':
            match = self.KEYWORD.match(self.text, self.pos)
            if match:
                self._advance(match.end() - self.pos)
                return Keyword(match.group()[1:])  # Strip leading colon
            raise ParseError(f"Invalid keyword", self.pos, self.line, self.col)

        # Number (must check before symbol due to - prefix)
        if char.isdigit() or (char == '-' and self.pos + 1 < len(self.text) and
                              (self.text[self.pos + 1].isdigit() or self.text[self.pos + 1] == '.')):
            match = self.NUMBER.match(self.text, self.pos)
            if match:
                self._advance(match.end() - self.pos)
                num_str = match.group()
                if '.' in num_str or 'e' in num_str or 'E' in num_str:
                    return float(num_str)
                return int(num_str)

        # Symbol
        match = self.SYMBOL.match(self.text, self.pos)
        if match:
            self._advance(match.end() - self.pos)
            return Symbol(match.group())

        raise ParseError(f"Unexpected character: {char!r}", self.pos, self.line, self.col)


def parse(text: str) -> Any:
    """
    Parse an S-expression string into Python data structures.

    Returns:
        Parsed S-expression as nested Python structures:
        - Lists become Python lists
        - Symbols become Symbol objects
        - Keywords become Keyword objects
        - Strings become Python strings
        - Numbers become int/float

    Example:
        >>> parse('(recipe "test" :version "1.0")')
        [Symbol('recipe'), 'test', Keyword('version'), '1.0']
    """
    tokenizer = Tokenizer(text)
    result = _parse_expr(tokenizer)

    # Check for trailing content
    if tokenizer.peek() is not None:
        raise ParseError("Unexpected content after expression",
                        tokenizer.pos, tokenizer.line, tokenizer.col)

    return result


def parse_all(text: str) -> List[Any]:
    """
    Parse multiple S-expressions from a string.

    Returns list of parsed expressions.
    """
    tokenizer = Tokenizer(text)
    results = []

    while tokenizer.peek() is not None:
        results.append(_parse_expr(tokenizer))

    return results


def _parse_expr(tokenizer: Tokenizer) -> Any:
    """Parse a single expression."""
    token = tokenizer.next_token()

    if token is None:
        raise ParseError("Unexpected end of input", tokenizer.pos, tokenizer.line, tokenizer.col)

    # List
    if token == '(':
        return _parse_list(tokenizer, ')')

    # Vector (sugar for list)
    if token == '[':
        return _parse_list(tokenizer, ']')

    # Map/dict: {:key1 val1 :key2 val2}
    if token == '{':
        return _parse_map(tokenizer)

    # Unexpected closers
    if token in (')', ']', '}'):
        raise ParseError(f"Unexpected {token!r}", tokenizer.pos, tokenizer.line, tokenizer.col)

    # Atom
    return token


def _parse_list(tokenizer: Tokenizer, closer: str) -> List[Any]:
    """Parse a list until the closing delimiter."""
    items = []

    while True:
        char = tokenizer.peek()

        if char is None:
            raise ParseError(f"Unterminated list, expected {closer!r}",
                           tokenizer.pos, tokenizer.line, tokenizer.col)

        if char == closer:
            tokenizer.next_token()  # Consume closer
            return items

        items.append(_parse_expr(tokenizer))


def _parse_map(tokenizer: Tokenizer) -> Dict[str, Any]:
    """Parse a map/dict: {:key1 val1 :key2 val2} -> {"key1": val1, "key2": val2}."""
    result = {}

    while True:
        char = tokenizer.peek()

        if char is None:
            raise ParseError("Unterminated map, expected '}'",
                           tokenizer.pos, tokenizer.line, tokenizer.col)

        if char == '}':
            tokenizer.next_token()  # Consume closer
            return result

        # Parse key (should be a keyword like :key)
        key_token = _parse_expr(tokenizer)
        if isinstance(key_token, Keyword):
            key = key_token.name
        elif isinstance(key_token, str):
            key = key_token
        else:
            raise ParseError(f"Map key must be keyword or string, got {type(key_token).__name__}",
                           tokenizer.pos, tokenizer.line, tokenizer.col)

        # Parse value
        value = _parse_expr(tokenizer)
        result[key] = value


def serialize(expr: Any, indent: int = 0, pretty: bool = False) -> str:
    """
    Serialize a Python data structure back to S-expression format.

    Args:
        expr: The expression to serialize
        indent: Current indentation level (for pretty printing)
        pretty: Whether to use pretty printing with newlines

    Returns:
        S-expression string
    """
    if isinstance(expr, list):
        if not expr:
            return "()"

        if pretty:
            return _serialize_pretty(expr, indent)
        else:
            items = [serialize(item, indent, False) for item in expr]
            return "(" + " ".join(items) + ")"

    if isinstance(expr, Symbol):
        return expr.name

    if isinstance(expr, Keyword):
        return f":{expr.name}"

    if isinstance(expr, Lambda):
        params = " ".join(expr.params)
        body = serialize(expr.body, indent, pretty)
        return f"(fn [{params}] {body})"

    if isinstance(expr, Binding):
        # analysis_ref can be a string, node ID, or dict - serialize it properly
        if isinstance(expr.analysis_ref, str):
            ref_str = f'"{expr.analysis_ref}"'
        else:
            ref_str = serialize(expr.analysis_ref, indent, pretty)
        s = f"(bind {ref_str} :range [{expr.range_min} {expr.range_max}]"
        if expr.transform:
            s += f" :transform {expr.transform}"
        return s + ")"

    if isinstance(expr, str):
        # Escape special characters
        escaped = expr.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\t', '\\t')
        return f'"{escaped}"'

    if isinstance(expr, bool):
        return "true" if expr else "false"

    if isinstance(expr, (int, float)):
        return str(expr)

    if expr is None:
        return "nil"

    if isinstance(expr, dict):
        # Serialize dict as property list: {:key1 val1 :key2 val2}
        items = []
        for k, v in expr.items():
            items.append(f":{k}")
            items.append(serialize(v, indent, pretty))
        return "{" + " ".join(items) + "}"

    raise ValueError(f"Cannot serialize {type(expr).__name__}: {expr!r}")


def _serialize_pretty(expr: List, indent: int) -> str:
    """Pretty-print a list expression with smart formatting."""
    if not expr:
        return "()"

    prefix = "  " * indent
    inner_prefix = "  " * (indent + 1)

    # Check if this is a simple list that fits on one line
    simple = serialize(expr, indent, False)
    if len(simple) < 60 and '\n' not in simple:
        return simple

    # Start building multiline output
    head = serialize(expr[0], indent + 1, False)
    parts = [f"({head}"]

    i = 1
    while i < len(expr):
        item = expr[i]

        # Group keyword-value pairs on same line
        if isinstance(item, Keyword) and i + 1 < len(expr):
            key = serialize(item, 0, False)
            val = serialize(expr[i + 1], indent + 1, False)

            # If value is short, put on same line
            if len(val) < 50 and '\n' not in val:
                parts.append(f"{inner_prefix}{key} {val}")
            else:
                # Value is complex, serialize it pretty
                val_pretty = serialize(expr[i + 1], indent + 1, True)
                parts.append(f"{inner_prefix}{key} {val_pretty}")
            i += 2
        else:
            # Regular item
            item_str = serialize(item, indent + 1, True)
            parts.append(f"{inner_prefix}{item_str}")
            i += 1

    return "\n".join(parts) + ")"
