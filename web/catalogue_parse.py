"""Read one Python file of the repo without running it: docstrings, functions, parameters and the files it reads/writes.

Everything comes from the file's own source (ast). Text that would attribute work to individuals ("Created on",
"@author", e-mail addresses) and absolute paths to private folders are left out.
"""
import ast
import re

PRIVATE_LINE = re.compile(r"created on|@author|@email|[\w.+-]+@[\w-]+\.[\w.]+", re.I)
READ_CALLS = {"glob", "read_csv", "read_table", "imread", "open_file", "loadtxt", "load_points", "read_json", "open"}
WRITE_CALLS = {"to_csv", "imsave", "imwrite", "save_file", "savetxt", "to_json", "savefig", "write_insight_bin",
               "export_histogram"}
FILE_EXT = re.compile(r"\.(csv|tif|tiff|txt|json|ijm|png|mp4|npy|h5|bin)\b", re.I)
BARE_EXT = re.compile(r"^\.\w+$")  # ".csv" alone: same name, new extension (not informative)
HEAVY = {"flika": "FLIKA", "qtpy": "Qt", "PyQt5": "Qt", "PyQt6": "Qt", "pyqtgraph": "pyqtgraph", "numba": "numba",
         "trackpy": "trackpy", "skimage": "scikit-image", "sklearn": "scikit-learn", "scipy": "SciPy",
         "pandas": "pandas", "numpy": "NumPy", "matplotlib": "matplotlib", "seaborn": "seaborn", "cv2": "OpenCV",
         "tifffile": "tifffile", "tqdm": "tqdm"}


def clean_doc(text):
    """Docstring without authorship/date lines and with surrounding blank lines trimmed."""
    if not text:
        return ""
    lines = [ln for ln in text.splitlines() if not PRIVATE_LINE.search(ln)]
    return "\n".join(lines).strip()


def parse_source(src):
    """ast.parse, tolerating IPython magics (lines starting with %), which some scripts contain."""
    magic = any(ln.lstrip().startswith("%") for ln in src.splitlines())
    if magic:
        src = "\n".join("pass  # " + ln if ln.lstrip().startswith("%") else ln for ln in src.splitlines())
    try:
        return ast.parse(src), magic, None
    except SyntaxError as e:
        return None, magic, f"{e.msg} (line {e.lineno})"


def _call_name(node):
    f = node.func
    return f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""


def _strings(node):
    return [n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def file_pattern(s):
    """A file-name pattern such as '*_locsID.csv' or '_NN.csv', or None for paths and non-file strings."""
    s = s.strip()
    if not FILE_EXT.search(s) or len(s) > 90:
        return None
    s = re.sub(r"^/?\*\*/", "", s)
    if "/" in s or "\\" in s or ":" in s or s.startswith("~"):
        return None
    return s


def files_used(tree):
    """Input and output file-name patterns found in read/write calls and save-name assignments."""
    ins, outs = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            target = ins if name in READ_CALLS else outs if name in WRITE_CALLS else None
            if target is not None:
                for a in list(node.args) + [k.value for k in node.keywords]:
                    target += [p for p in map(file_pattern, _strings(a)) if p]
        elif isinstance(node, ast.Assign):
            names = [t.id.lower() for t in node.targets if isinstance(t, ast.Name)]
            if any("save" in n or "outpath" in n for n in names):
                outs += [p for p in map(file_pattern, _strings(node.value)) if p]
            elif any(n.endswith(("file", "filename", "list")) for n in names) and isinstance(node.value, (ast.BinOp, ast.Call)):
                ins += [p for p in map(file_pattern, _strings(node.value)) if p]
    dedup = lambda xs: list(dict.fromkeys(x for x in xs if not BARE_EXT.match(x)))
    outs = dedup(outs)
    return dedup([i for i in ins if i not in outs]), outs


def _literal(node):
    try:
        v = ast.literal_eval(node)
    except Exception:
        return None
    if isinstance(v, bool) or isinstance(v, (int, float)):
        return v
    if isinstance(v, (list, tuple)) and v and all(isinstance(x, (int, float)) for x in v):
        return list(v)
    if isinstance(v, str) and len(v) < 40 and "/" not in v and "\\" not in v and not FILE_EXT.search(v):
        return v
    return None


def main_settings(tree):
    """Simple settings assigned in the `if __name__ == '__main__':` block (numbers, short strings, number lists)."""
    out = []
    for node in tree.body:
        if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test):
            for st in ast.walk(node):
                if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                    v = _literal(st.value)
                    if v is not None and v != "":
                        out.append({"name": st.targets[0].id, "value": v})
    seen, res = set(), []
    for s in out:
        if s["name"] not in seen:
            seen.add(s["name"])
            res.append(s)
    return res


def _func(node, owner=None):
    args = ast.unparse(node.args)
    params = []
    a = node.args
    pos = a.posonlyargs + a.args
    defaults = [None] * (len(pos) - len(a.defaults)) + list(a.defaults)
    for arg, d in zip(pos, defaults):
        if arg.arg != "self":
            params.append({"name": arg.arg, "default": None if d is None else ast.unparse(d)})
    for arg, d in zip(a.kwonlyargs, a.kw_defaults):
        params.append({"name": arg.arg, "default": None if d is None else ast.unparse(d)})
    return {"name": node.name, "owner": owner, "signature": f"{node.name}({args})", "params": params,
            "doc": clean_doc(ast.get_docstring(node)), "line": node.lineno}


def definitions(tree):
    """Top-level functions and classes (with their methods), in source order."""
    funcs, classes = [], []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append(_func(node))
        elif isinstance(node, ast.ClassDef):
            methods = [_func(m, node.name) for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
            classes.append({"name": node.name, "bases": [ast.unparse(b) for b in node.bases],
                            "doc": clean_doc(ast.get_docstring(node)), "line": node.lineno,
                            "methods": [m for m in methods if not m["name"].startswith("__") or m["name"] == "__init__"]})
    return funcs, classes


def imports(tree):
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module.split(".")[0])
    return sorted(mods)


def has_main(tree):
    return any(isinstance(n, ast.If) and "__main__" in ast.unparse(n.test) for n in tree.body)


def readme_note(md, fname):
    """The README's description of `fname`: a '- `f.py`: text' bullet, or the section under '**Script**: `f.py`'."""
    lines = md.splitlines()
    for i, line in enumerate(lines):
        if f"`{fname}`" not in line:
            continue
        m = re.match(r"\s*[-*]\s*`" + re.escape(fname) + r"`\s*:\s*(.+)", line)
        if m:
            return m.group(1).strip()
        if line.strip().startswith("**Script**"):
            heading = lines[i - 1].lstrip("#").strip() if i and lines[i - 1].startswith("#") else ""
            heading = re.sub(r"^\d+\.\s*", "", heading)
            bullets, code, k = [], False, i + 1
            while k < len(lines) and not lines[k].startswith("#"):
                s = lines[k]
                if s.strip().startswith("```"):
                    code = not code
                elif not code and s.startswith("- ") and not s.rstrip().endswith(":"):
                    bullets.append(s[2:].strip())
                k += 1
            return (heading + ". " if heading else "") + "; ".join(bullets)
    return None
