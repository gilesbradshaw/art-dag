"""
Effect file loader.

Parses effect files with:
- PEP 723 inline script metadata for dependencies
- @-tag docstrings for effect metadata
- META object for programmatic access
"""

import ast
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .meta import EffectMeta, ParamSpec


@dataclass
class LoadedEffect:
    """
    A loaded effect with all metadata.

    Attributes:
        source: Original source code
        cid: SHA3-256 hash of source
        meta: Extracted EffectMeta
        dependencies: List of pip dependencies
        requires_python: Python version requirement
        module: Compiled module (if loaded)
    """

    source: str
    cid: str
    meta: EffectMeta
    dependencies: List[str] = field(default_factory=list)
    requires_python: str = ">=3.10"
    module: Any = None

    def has_frame_api(self) -> bool:
        """Check if effect has frame-by-frame API."""
        return self.meta.api_type == "frame"

    def has_video_api(self) -> bool:
        """Check if effect has whole-video API."""
        return self.meta.api_type == "video"


def compute_cid(source: str) -> str:
    """Compute SHA3-256 hash of effect source."""
    return hashlib.sha3_256(source.encode("utf-8")).hexdigest()


def parse_pep723_metadata(source: str) -> Tuple[List[str], str]:
    """
    Parse PEP 723 inline script metadata.

    Looks for:
        # /// script
        # requires-python = ">=3.10"
        # dependencies = ["numpy", "opencv-python"]
        # ///

    Returns:
        Tuple of (dependencies list, requires_python string)
    """
    dependencies = []
    requires_python = ">=3.10"

    # Match the script block
    pattern = r"# /// script\n(.*?)# ///"
    match = re.search(pattern, source, re.DOTALL)

    if not match:
        return dependencies, requires_python

    block = match.group(1)

    # Parse dependencies
    deps_match = re.search(r'# dependencies = \[(.*?)\]', block, re.DOTALL)
    if deps_match:
        deps_str = deps_match.group(1)
        # Extract quoted strings
        dependencies = re.findall(r'"([^"]+)"', deps_str)

    # Parse requires-python
    python_match = re.search(r'# requires-python = "([^"]+)"', block)
    if python_match:
        requires_python = python_match.group(1)

    return dependencies, requires_python


def parse_docstring_metadata(docstring: str) -> Dict[str, Any]:
    """
    Parse @-tag metadata from docstring.

    Supports:
        @effect name
        @version 1.0.0
        @author @user@domain
        @temporal false
        @description
        Multi-line description text.

        @param name type
          @range lo hi
          @default value
          Description text.

        @example
          (fx effect :param value)

    Returns:
        Dictionary with extracted metadata
    """
    if not docstring:
        return {}

    result = {
        "name": "",
        "version": "1.0.0",
        "author": "",
        "temporal": False,
        "description": "",
        "params": [],
        "examples": [],
    }

    lines = docstring.strip().split("\n")
    i = 0
    current_param = None

    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("@effect "):
            result["name"] = line[8:].strip()

        elif line.startswith("@version "):
            result["version"] = line[9:].strip()

        elif line.startswith("@author "):
            result["author"] = line[8:].strip()

        elif line.startswith("@temporal "):
            val = line[10:].strip().lower()
            result["temporal"] = val in ("true", "yes", "1")

        elif line.startswith("@description"):
            # Collect multi-line description
            desc_lines = []
            i += 1
            while i < len(lines):
                next_line = lines[i]
                if next_line.strip().startswith("@"):
                    i -= 1  # Back up to process this tag
                    break
                desc_lines.append(next_line)
                i += 1
            result["description"] = "\n".join(desc_lines).strip()

        elif line.startswith("@param "):
            # Parse parameter: @param name type
            parts = line[7:].split()
            if len(parts) >= 2:
                current_param = {
                    "name": parts[0],
                    "type": parts[1],
                    "range": None,
                    "default": None,
                    "description": "",
                }
                # Collect param details
                desc_lines = []
                i += 1
                while i < len(lines):
                    next_line = lines[i]
                    stripped = next_line.strip()

                    if stripped.startswith("@range "):
                        range_parts = stripped[7:].split()
                        if len(range_parts) >= 2:
                            try:
                                current_param["range"] = (
                                    float(range_parts[0]),
                                    float(range_parts[1]),
                                )
                            except ValueError:
                                pass

                    elif stripped.startswith("@default "):
                        current_param["default"] = stripped[9:].strip()

                    elif stripped.startswith("@param ") or stripped.startswith("@example"):
                        i -= 1  # Back up
                        break

                    elif stripped.startswith("@"):
                        i -= 1
                        break

                    elif stripped:
                        desc_lines.append(stripped)

                    i += 1

                current_param["description"] = " ".join(desc_lines)
                result["params"].append(current_param)
                current_param = None

        elif line.startswith("@example"):
            # Collect example
            example_lines = []
            i += 1
            while i < len(lines):
                next_line = lines[i]
                if next_line.strip().startswith("@") and not next_line.strip().startswith("@example"):
                    if next_line.strip().startswith("@example"):
                        i -= 1
                    break
                if next_line.strip().startswith("@example"):
                    i -= 1
                    break
                example_lines.append(next_line)
                i += 1
            example = "\n".join(example_lines).strip()
            if example:
                result["examples"].append(example)

        i += 1

    return result


def extract_meta_from_ast(source: str) -> Optional[Dict[str, Any]]:
    """
    Extract META object from source AST.

    Looks for:
        META = EffectMeta(...)

    Returns the keyword arguments if found.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "META":
                    if isinstance(node.value, ast.Call):
                        return _extract_call_kwargs(node.value)
    return None


def _extract_call_kwargs(call: ast.Call) -> Dict[str, Any]:
    """Extract keyword arguments from an AST Call node."""
    result = {}

    for keyword in call.keywords:
        if keyword.arg is None:
            continue
        value = _ast_to_value(keyword.value)
        if value is not None:
            result[keyword.arg] = value

    return result


def _ast_to_value(node: ast.expr) -> Any:
    """Convert AST node to Python value."""
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Str):  # Python 3.7 compat
        return node.s
    elif isinstance(node, ast.Num):  # Python 3.7 compat
        return node.n
    elif isinstance(node, ast.NameConstant):  # Python 3.7 compat
        return node.value
    elif isinstance(node, ast.List):
        return [_ast_to_value(elt) for elt in node.elts]
    elif isinstance(node, ast.Tuple):
        return tuple(_ast_to_value(elt) for elt in node.elts)
    elif isinstance(node, ast.Dict):
        return {
            _ast_to_value(k): _ast_to_value(v)
            for k, v in zip(node.keys, node.values)
            if k is not None
        }
    elif isinstance(node, ast.Call):
        # Handle ParamSpec(...) calls
        if isinstance(node.func, ast.Name) and node.func.id == "ParamSpec":
            return _extract_call_kwargs(node)
    return None


def get_module_docstring(source: str) -> str:
    """Extract the module-level docstring from source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""

    if tree.body and isinstance(tree.body[0], ast.Expr):
        if isinstance(tree.body[0].value, ast.Constant):
            return tree.body[0].value.value
        elif isinstance(tree.body[0].value, ast.Str):  # Python 3.7 compat
            return tree.body[0].value.s
    return ""


def load_effect(source: str) -> LoadedEffect:
    """
    Load an effect from source code.

    Parses:
    1. PEP 723 metadata for dependencies
    2. Module docstring for @-tag metadata
    3. META object for programmatic metadata

    Priority: META object > docstring > defaults

    Args:
        source: Effect source code

    Returns:
        LoadedEffect with all metadata

    Raises:
        ValueError: If effect is invalid
    """
    cid = compute_cid(source)

    # Parse PEP 723 metadata
    dependencies, requires_python = parse_pep723_metadata(source)

    # Parse docstring metadata
    docstring = get_module_docstring(source)
    doc_meta = parse_docstring_metadata(docstring)

    # Try to extract META from AST
    ast_meta = extract_meta_from_ast(source)

    # Build EffectMeta, preferring META object over docstring
    name = ""
    if ast_meta and "name" in ast_meta:
        name = ast_meta["name"]
    elif doc_meta.get("name"):
        name = doc_meta["name"]

    if not name:
        raise ValueError("Effect must have a name (@effect or META.name)")

    version = ast_meta.get("version") if ast_meta else doc_meta.get("version", "1.0.0")
    temporal = ast_meta.get("temporal") if ast_meta else doc_meta.get("temporal", False)
    author = ast_meta.get("author") if ast_meta else doc_meta.get("author", "")
    description = ast_meta.get("description") if ast_meta else doc_meta.get("description", "")
    examples = ast_meta.get("examples") if ast_meta else doc_meta.get("examples", [])

    # Build params
    params = []
    if ast_meta and "params" in ast_meta:
        for p in ast_meta["params"]:
            if isinstance(p, dict):
                type_map = {"float": float, "int": int, "bool": bool, "str": str}
                param_type = type_map.get(p.get("param_type", "float"), float)
                if isinstance(p.get("param_type"), type):
                    param_type = p["param_type"]
                params.append(
                    ParamSpec(
                        name=p.get("name", ""),
                        param_type=param_type,
                        default=p.get("default"),
                        range=p.get("range"),
                        description=p.get("description", ""),
                    )
                )
    elif doc_meta.get("params"):
        for p in doc_meta["params"]:
            type_map = {"float": float, "int": int, "bool": bool, "str": str}
            param_type = type_map.get(p.get("type", "float"), float)

            default = p.get("default")
            if default is not None:
                try:
                    default = param_type(default)
                except (ValueError, TypeError):
                    pass

            params.append(
                ParamSpec(
                    name=p["name"],
                    param_type=param_type,
                    default=default,
                    range=p.get("range"),
                    description=p.get("description", ""),
                )
            )

    # Determine API type by checking for function definitions
    api_type = "frame"  # default
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name == "process":
                    api_type = "video"
                    break
                elif node.name == "process_frame":
                    api_type = "frame"
                    break
    except SyntaxError:
        pass

    meta = EffectMeta(
        name=name,
        version=version if isinstance(version, str) else "1.0.0",
        temporal=bool(temporal),
        params=params,
        author=author if isinstance(author, str) else "",
        description=description if isinstance(description, str) else "",
        examples=examples if isinstance(examples, list) else [],
        dependencies=dependencies,
        requires_python=requires_python,
        api_type=api_type,
    )

    return LoadedEffect(
        source=source,
        cid=cid,
        meta=meta,
        dependencies=dependencies,
        requires_python=requires_python,
    )


def load_effect_file(path: Path) -> LoadedEffect:
    """Load an effect from a file path."""
    source = path.read_text(encoding="utf-8")
    return load_effect(source)


def compute_deps_hash(dependencies: List[str]) -> str:
    """
    Compute hash of sorted dependencies.

    Used for venv caching - same deps = same hash = reuse venv.
    """
    sorted_deps = sorted(dep.lower().strip() for dep in dependencies)
    deps_str = "\n".join(sorted_deps)
    return hashlib.sha3_256(deps_str.encode("utf-8")).hexdigest()
