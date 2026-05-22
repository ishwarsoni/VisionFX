#!/usr/bin/env python3
"""
Static import analysis and annotation for unused-report candidates.
Generates archive/unused_report_annotated.txt with references and suggestions.
"""
import ast
import re
from pathlib import Path

ROOT = Path("d:/sharingan")
EXCLUDE_DIRS = {".venv", "archive", "repo", "__pycache__", "sharingan_pack"}
REPORT_IN = ROOT / "archive" / "unused_report.txt"
REPORT_OUT = ROOT / "archive" / "unused_report_annotated.txt"


def list_py_files(root):
    return [
        p
        for p in root.rglob("*.py")
        if not any(part in EXCLUDE_DIRS for part in p.parts)
    ]


def module_from_path(root, path):
    rel = path.relative_to(root)
    return str(rel.with_suffix("")).replace("\\", ".").replace("/", ".")


def parse_imports(path):
    try:
        src = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(src)
    except Exception as e:
        return {"imports": [], "text": src}
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module
            if mod:
                imports.append(mod)
    return {"imports": imports, "text": src}


def main():
    if not REPORT_IN.exists():
        print("Report not found:", REPORT_IN)
        return
    # read candidates
    raw_lines = [
        l.strip()
        for l in REPORT_IN.read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.startswith("#")
    ]
    # Keep only lines that look like relative python paths (end with .py)
    candidates = []
    for l in raw_lines:
        try:
            p = Path(l)
            if p.suffix == ".py":
                candidates.append(l)
        except Exception:
            continue

    py_files = list_py_files(ROOT)
    file_info = {}
    for p in py_files:
        file_info[str(p)] = parse_imports(p)

    # Build reverse index: module -> list(files importing it)
    module_index = {}
    basename_index = {}
    for file_path, info in file_info.items():
        for imp in info["imports"]:
            module_index.setdefault(imp, []).append(file_path)
        # also index words in text for heuristic matches
        txt = info["text"]
        words = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]+\b", txt))
        for w in words:
            basename_index.setdefault(w, []).append(file_path)

    out_lines = []
    out_lines.append("# Unused-file annotated report")
    out_lines.append(
        "This is a static heuristic annotation. Verify manually before deletion.\n"
    )

    for c in candidates:
        out_lines.append("---")
        out_lines.append("Candidate: " + c)
        mod = c.replace("\\", "/").replace("/", ".")
        mod = mod[:-3] if mod.endswith(".py") else mod
        basename = Path(c).stem
        out_lines.append("Module path: " + mod)
        out_lines.append("Basename: " + basename)

        refs = set()
        # direct module import
        if mod in module_index:
            refs.update(module_index[mod])
        # package prefix imports e.g., effects.cinematic_eye_compositor referenced by effects
        parts = mod.split(".")
        for i in range(1, len(parts)):
            prefix = ".".join(parts[:i])
            if prefix in module_index:
                refs.update(module_index[prefix])
        # basename imports or textual mentions
        if basename in basename_index:
            refs.update(basename_index[basename])
        # textual dotted mentions
        dotted = basename + "."
        for fp, info in file_info.items():
            if dotted in info["text"]:
                refs.add(fp)

        if refs:
            out_lines.append("Static references found (" + str(len(refs)) + "):")
            for r in sorted(refs):
                out_lines.append(" - " + str(Path(r).relative_to(ROOT)))
            out_lines.append("Recommendation: KEEP (referenced)")
        else:
            out_lines.append("Static references found: 0")
            out_lines.append("Recommendation: REVIEW (no static references detected)")

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text("\n".join(out_lines), encoding="utf-8")
    print("Wrote annotated report to:", REPORT_OUT)


if __name__ == "__main__":
    main()
