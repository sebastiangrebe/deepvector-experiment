"""Tree-sitter-based symbolic index for Python repos.

Phase 2.9: routes a query to candidate files by matching identifiers extracted
from the issue text against an inverted index of identifiers defined per file.

Index contents per file:
    - top-level function names
    - top-level + nested class names
    - top-level methods of classes
    - module-level constant assignments (NAME = ... where NAME is UPPER_CASE
      or simple ALL_CAPS)
    - imports: list of imported module dotted paths

The index is built once per repo and cached to data/tree_index_cache/.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import tree_sitter as ts
import tree_sitter_python as tsp

from src.utils import get_logger, list_eligible_files, project_root

log = get_logger("tree_index")
CACHE_DIR = project_root() / "data" / "tree_index_cache"


# ─────────────────────────────────────────────────────────────────────
# Tree-sitter parser (reused across files)
# ─────────────────────────────────────────────────────────────────────

_LANGUAGE = ts.Language(tsp.language())
_PARSER = ts.Parser(_LANGUAGE)


# ─────────────────────────────────────────────────────────────────────
# Per-file metadata
# ─────────────────────────────────────────────────────────────────────

@dataclass
class FileMetadata:
    path: str                           # repo-relative POSIX
    functions: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    constants: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    byte_size: int = 0

    def all_defined_names(self) -> set[str]:
        return set(self.functions) | set(self.classes) | set(self.methods) | set(self.constants)


_CONST_PAT = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _id_text(node, src: bytes) -> str | None:
    for c in node.children:
        if c.type == "identifier":
            return src[c.start_byte:c.end_byte].decode("utf-8", errors="replace")
    return None


def _walk_children(node, types: tuple[str, ...]):
    for c in node.children:
        if c.type in types:
            yield c


def _unwrap_decorated(node):
    """If `node` is a decorated_definition, return its inner function/class node."""
    if node.type != "decorated_definition":
        return node
    for c in node.children:
        if c.type in ("function_definition", "class_definition"):
            return c
    return node


def _walk_class_body(class_node, src: bytes, out_methods: list[str]) -> None:
    body = next((c for c in class_node.children if c.type == "block"), None)
    if not body:
        return
    for stmt in body.children:
        inner = _unwrap_decorated(stmt)
        if inner.type == "function_definition":
            m = _id_text(inner, src)
            if m:
                out_methods.append(m)


def _extract_metadata(rel_path: str, src: bytes) -> FileMetadata:
    tree = _PARSER.parse(src)
    root = tree.root_node
    md = FileMetadata(path=rel_path, byte_size=len(src))

    for top_raw in root.children:
        top = _unwrap_decorated(top_raw)
        if top.type == "function_definition":
            n = _id_text(top, src)
            if n:
                md.functions.append(n)
        elif top.type == "class_definition":
            n = _id_text(top, src)
            if n:
                md.classes.append(n)
            _walk_class_body(top, src, md.methods)
        elif top.type == "expression_statement":
            for c in top.children:
                if c.type == "assignment":
                    lhs = c.children[0] if c.children else None
                    if lhs and lhs.type == "identifier":
                        name = src[lhs.start_byte:lhs.end_byte].decode(
                            "utf-8", errors="replace")
                        if _CONST_PAT.match(name):
                            md.constants.append(name)
        elif top.type == "import_statement":
            # import a.b.c
            for c in _walk_children(top, ("dotted_name", "aliased_import")):
                if c.type == "dotted_name":
                    md.imports.append(src[c.start_byte:c.end_byte]
                                       .decode("utf-8", errors="replace"))
                elif c.type == "aliased_import":
                    inner = next((cc for cc in c.children
                                  if cc.type == "dotted_name"), None)
                    if inner:
                        md.imports.append(src[inner.start_byte:inner.end_byte]
                                           .decode("utf-8", errors="replace"))
        elif top.type == "import_from_statement":
            mod = next((c for c in top.children
                        if c.type in ("dotted_name", "relative_import")), None)
            if mod:
                md.imports.append(src[mod.start_byte:mod.end_byte]
                                   .decode("utf-8", errors="replace"))

    return md


# ─────────────────────────────────────────────────────────────────────
# Repo index
# ─────────────────────────────────────────────────────────────────────

class RepoTreeIndex:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.files: dict[str, FileMetadata] = {}
        self.identifier_to_files: dict[str, set[str]] = {}
        self.import_graph: dict[str, set[str]] = {}
        # Module-tail → file path (best-effort import resolution)
        self._tail_to_file: dict[str, set[str]] = {}

    @classmethod
    def _cache_signature(cls, repo_root: Path, files: list[Path]) -> str:
        h = hashlib.sha256()
        h.update(b"tree_index_v1")
        for f in files:
            try:
                st = f.stat()
            except OSError:
                continue
            h.update(f.relative_to(repo_root).as_posix().encode())
            h.update(f"{st.st_size}-{int(st.st_mtime)}".encode())
        return h.hexdigest()[:16]

    def build(self, force: bool = False) -> None:
        files = list_eligible_files(self.repo_root)
        sig = self._cache_signature(self.repo_root, files)
        cache_file = CACHE_DIR / f"{self.repo_root.name}_{sig}.json"

        if cache_file.exists() and not force:
            log.info("tree-index cache hit %s", cache_file.name)
            self._load_from(cache_file)
            return

        log.info("Building tree-index for %s (%d files)", self.repo_root.name, len(files))
        for p in files:
            try:
                src = p.read_bytes()
            except OSError:
                continue
            rel = p.relative_to(self.repo_root).as_posix()
            try:
                md = _extract_metadata(rel, src)
            except Exception as e:
                log.warning("parse failed on %s: %s", rel, e)
                continue
            self.files[rel] = md
            for name in md.all_defined_names():
                self.identifier_to_files.setdefault(name, set()).add(rel)
            # Tail-of-path → file for import resolution
            stem = Path(rel).stem
            self._tail_to_file.setdefault(stem, set()).add(rel)

        # Build import graph (best-effort: match imports to files in this repo)
        for rel, md in self.files.items():
            edges: set[str] = set()
            for imp in md.imports:
                tail = imp.rsplit(".", 1)[-1]
                if tail in self._tail_to_file:
                    edges.update(self._tail_to_file[tail])
            edges.discard(rel)
            self.import_graph[rel] = edges

        log.info("Indexed %d files, %d unique identifiers, %d import edges",
                 len(self.files), len(self.identifier_to_files),
                 sum(len(v) for v in self.import_graph.values()))

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._save_to(cache_file)

    # ─── persistence ──────────────────────────────────────────────────

    def _save_to(self, path: Path) -> None:
        payload = {
            "files": {rel: {
                "path": md.path,
                "functions": md.functions,
                "classes": md.classes,
                "methods": md.methods,
                "constants": md.constants,
                "imports": md.imports,
                "byte_size": md.byte_size,
            } for rel, md in self.files.items()},
            "identifier_to_files": {k: sorted(v) for k, v in self.identifier_to_files.items()},
            "import_graph": {k: sorted(v) for k, v in self.import_graph.items()},
        }
        path.write_text(json.dumps(payload))

    def _load_from(self, path: Path) -> None:
        d = json.loads(path.read_text())
        self.files = {rel: FileMetadata(**v) for rel, v in d["files"].items()}
        self.identifier_to_files = {k: set(v) for k, v in d["identifier_to_files"].items()}
        self.import_graph = {k: set(v) for k, v in d["import_graph"].items()}

    # ─── lookups ──────────────────────────────────────────────────────

    def lookup(self, identifier: str) -> set[str]:
        return self.identifier_to_files.get(identifier, set()).copy()

    def expand_via_imports(self, files: set[str], hops: int = 1) -> set[str]:
        out = set(files)
        frontier = set(files)
        for _ in range(hops):
            nxt: set[str] = set()
            for f in frontier:
                nxt.update(self.import_graph.get(f, set()))
            new = nxt - out
            if not new:
                break
            out.update(new)
            frontier = new
        return out

    def files_count(self) -> int:
        return len(self.files)

    def identifiers_count(self) -> int:
        return len(self.identifier_to_files)


# ─────────────────────────────────────────────────────────────────────
# Identifier extraction from issue text
# ─────────────────────────────────────────────────────────────────────

# Match: dotted paths, CamelCase, snake_case, backtick-quoted code, def/class
_DOTTED  = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b")
_CAMEL   = re.compile(r"\b[A-Z][a-zA-Z0-9]+(?:[A-Z][a-zA-Z0-9]+)+\b")
_SNAKE   = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_UPPER   = re.compile(r"\b[A-Z][A-Z0-9_]*[A-Z0-9]\b")  # FILE_UPLOAD_PERMISSIONS
_BACKTICK = re.compile(r"`([^`\n]{1,80})`")
_DEFCLASS = re.compile(r"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")
_IDENT_OK = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def extract_identifiers(text: str) -> list[str]:
    """Return distinct identifier candidates from issue text. Order-preserving."""
    seen: dict[str, None] = {}
    for pat in (_DOTTED, _CAMEL, _UPPER, _SNAKE):
        for m in pat.finditer(text):
            tok = m.group(0)
            if _IDENT_OK.match(tok) and tok not in seen:
                seen[tok] = None
    for m in _BACKTICK.finditer(text):
        for sub in re.split(r"[\s\(\)\[\],:=]+", m.group(1)):
            if _IDENT_OK.match(sub) and sub not in seen:
                seen[sub] = None
    for m in _DEFCLASS.finditer(text):
        tok = m.group(1)
        if _IDENT_OK.match(tok) and tok not in seen:
            seen[tok] = None
    # Drop tokens that are pure single-char or common stopwords
    drops = {"True", "False", "None", "self", "cls", "args", "kwargs",
             "lambda", "return", "yield", "import", "from", "def", "class"}
    return [k for k in seen.keys() if k not in drops]
