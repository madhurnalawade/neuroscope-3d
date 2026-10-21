"""Static scan of the repository for code that could send data elsewhere.

Python files are parsed with ``ast`` (nothing is imported or executed). Reported:
- imports of network, cloud, telemetry, crash-reporting, LLM and process-spawning modules;
- calls such as os.system, subprocess.*, webbrowser.open, urlopen, socket.*, torch.hub.*, model downloads;
- dynamic imports (importlib.import_module, __import__), which hide what is loaded;
- a server bound to every interface ("0.0.0.0") and CORS allowing any origin;
- production modules importing neuroscope.dev;
- served HTML/JS/CSS that loads resources from external URLs (scripts, styles, fonts, fetch, beacons).
Every hit is either ALLOWED (with a reason from ALLOWLIST) or a PROBLEM.
"""
import ast
import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ALL_INTERFACES = ".".join(["0"] * 4)   # built, so this file does not flag itself
EXCLUDED_DIRS = {".git", ".venv", "venv", "env", "node_modules", "__pycache__", ".pytest_cache", "data",
                 "outputs", "dataset", "demo_cases", "backup_case", "site-packages", "dist-packages", "vendor"}

CATEGORIES = {
    "network": ["socket", "ssl", "requests", "httpx", "httpx2", "urllib", "urllib3", "http.client", "aiohttp",
                "ftplib", "smtplib", "poplib", "imaplib", "telnetlib", "xmlrpc", "websocket", "websockets",
                "grpc", "paramiko", "pysftp", "smbclient", "pycurl", "tornado.httpclient", "asyncio.open_connection"],
    "cloud": ["boto3", "botocore", "s3fs", "gcsfs", "adlfs", "google.cloud", "google.auth", "googleapiclient",
              "azure", "dropbox", "onedrivesdk", "msal", "msgraph", "box_sdk", "boxsdk", "pydrive", "pydrive2",
              "huggingface_hub", "kaggle"],
    "telemetry": ["sentry_sdk", "raven", "rollbar", "bugsnag", "honeybadger", "newrelic", "datadog", "ddtrace",
                  "opentelemetry", "posthog", "mixpanel", "analytics", "segment", "amplitude", "statsd",
                  "prometheus_client", "wandb", "mlflow", "comet_ml", "neptune", "clearml", "logging.handlers"],
    "llm": ["openai", "anthropic", "google.generativeai", "google.genai", "vertexai", "cohere", "mistralai",
            "ollama", "langchain", "llama_index", "litellm", "groq", "replicate", "together"],
    "process": ["subprocess", "pty", "webbrowser", "multiprocessing.connection"],
    "server": ["fastapi", "uvicorn", "starlette", "flask", "http.server", "socketserver", "aiohttp.web"],
}
RISKY_CALLS = {
    "os.system": "process", "os.popen": "process", "os.startfile": "process", "os.execv": "process",
    "os.execvp": "process", "os.execl": "process", "os.spawnl": "process", "os.spawnv": "process",
    "os.posix_spawn": "process", "subprocess.": "process", "webbrowser.": "process",
    "urllib.request.urlopen": "network", "urllib.request.urlretrieve": "network", "socket.": "network",
    "requests.": "network", "httpx.": "network", "torch.hub.": "network", "monai.bundle.download": "network",
    "monai.apps.download_url": "network", "huggingface_hub.": "cloud",
    "importlib.import_module": "dynamic-import", "__import__": "dynamic-import",
}
WEB_PATTERNS = [
    (re.compile(r"<(?:script|link|img|iframe|source|video|audio|embed)\b[^>]*\b(?:src|href)\s*=\s*[\"']?(https?:)?//", re.I),
     "loads a resource from an external URL"),
    (re.compile(r"@import\s+(?:url\()?\s*[\"']?(https?:)?//", re.I), "CSS @import from an external URL"),
    (re.compile(r"url\(\s*[\"']?https?://", re.I), "CSS url() from an external URL"),
    (re.compile(r"\bfetch\(\s*[\"'`]https?://", re.I), "fetch() to an absolute external URL"),
    (re.compile(r"\bsendBeacon\s*\(", re.I), "navigator.sendBeacon (analytics-style upload)"),
    (re.compile(r"\bnew\s+WebSocket\s*\(\s*[\"'`]wss?://", re.I), "WebSocket to an absolute URL"),
    (re.compile(r"\bimport\s[^;]*?from\s*[\"']https?://", re.I), "ES module import from an external URL"),
]
WEB_GLOBS = ["Input/neuroscope_3d/backend/app/static/*.html", "Input/neuroscope_3d/backend/app/static/*.js",
             "Input/neuroscope_3d/backend/app/static/*.css", "Input/neuroscope_3d/frontend/dist/*.html",
             "Input/neuroscope_3d/frontend/index.html", "Frontend/index.html"]

# Production code that must never import neuroscope.dev.
PRODUCTION_GLOBS = ["neuroscope/io/*", "neuroscope/storage/*", "neuroscope/security/*", "neuroscope/__init__.py",
                    "brainseg/*", "Input/neuroscope_3d/backend/app/*", "run.py"]

# (path glob, detail glob, reason)
ALLOWLIST: List[Tuple[str, str, str]] = [
    ("neuroscope/security/netguard.py", "*socket*", "imports socket only to block it"),
    ("neuroscope/security/static_audit.py", "*", "only names these modules as strings to search for"),
    ("neuroscope/security/hygiene.py", "import logging.handlers", "imports the handler classes only to detect "
     "and refuse remote log handlers"),
    ("tests/test_security.py", "import logging.handlers", "creates a remote handler to prove it is refused"),
    ("tests/test_security.py", "*", "deliberately attempts connections to prove the network guard blocks them"),
    ("tests/conftest.py", "*socket*", "none expected; listed for completeness"),
    ("brainseg/inference.py", "*subprocess*", "runs the local 'git rev-parse/status' for the code-commit "
     "provenance; no data is passed"),
    ("run.py", "*monai.bundle*", "downloads the public model weights once if missing; sends no scan data"),
    ("run.py", "*uvicorn*", "local web server; see the bind-address check"),
    ("Input/neuroscope_3d/backend/app/*", "*fastapi*", "local web server framework"),
    ("Input/neuroscope_3d/backend/app/*", "*starlette*", "local web server framework"),
    ("Input/neuroscope_3d/backend/tests/*", "*fastapi*", "in-process TestClient; no network"),
    ("Input/neuroscope_3d/backend/tests/*", "*starlette.testclient*", "in-process TestClient; no network"),
    ("tests/judge_scenarios.py", "*requests*", "client for the server URL given on the command line; use only "
     "with http://127.0.0.1"),
    ("tests/http_end_to_end.py", "*requests*", "client for the server URL given on the command line; use only "
     "with http://127.0.0.1"),
    ("scripts/list_dependencies.py", "*importlib*", "reads installed package metadata locally"),
]


@dataclass
class Hit:
    path: str
    line: int
    category: str
    detail: str
    allowed_reason: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.allowed_reason is not None


def _matches(name: str, prefixes: Iterable[str]) -> Optional[str]:
    for p in prefixes:
        if name == p or name.startswith(p + "."):
            return p
    return None


def _category_of(module: str) -> Optional[str]:
    for cat, mods in CATEGORIES.items():
        if _matches(module, mods):
            return cat
    return None


def _dotted(node) -> Optional[str]:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _module_of(rel: str) -> str:
    parts = rel[:-3].split("/") if rel.endswith(".py") else rel.split("/")
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _resolve(rel: str, level: int, module: Optional[str]) -> str:
    if level == 0:
        return module or ""
    pkg = _module_of(rel).split(".")
    if not rel.endswith("__init__.py"):
        pkg = pkg[:-1]
    base = pkg[: len(pkg) - (level - 1)] if level > 1 else pkg
    return ".".join(base + ([module] if module else []))


def scan_python(path: Path, rel: str) -> List[Hit]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=rel)
    except SyntaxError as exc:
        return [Hit(rel, exc.lineno or 0, "parse-error", "file could not be parsed; review by hand")]
    hits: List[Hit] = []
    aliases: Dict[str, str] = {}
    production = any(fnmatch.fnmatch(rel, g) for g in PRODUCTION_GLOBS)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                aliases[(a.asname or a.name).split(".")[0] if not a.asname else a.asname] = \
                    a.name if a.asname else a.name.split(".")[0]
                cat = _category_of(a.name)
                if cat:
                    hits.append(Hit(rel, node.lineno, cat, f"import {a.name}"))
                if production and _matches(a.name, ["neuroscope.dev"]):
                    hits.append(Hit(rel, node.lineno, "dev-import", f"production code imports {a.name}"))
        elif isinstance(node, ast.ImportFrom):
            mod = _resolve(rel, node.level, node.module)
            for a in node.names:
                aliases[a.asname or a.name] = f"{mod}.{a.name}"
            full = [mod] + [f"{mod}.{a.name}" for a in node.names]
            cat = next((c for c in map(_category_of, full) if c), None)
            if cat:
                hits.append(Hit(rel, node.lineno, cat, f"from {mod} import {', '.join(a.name for a in node.names)}"))
            if production and any(_matches(m, ["neuroscope.dev"]) for m in full):
                hits.append(Hit(rel, node.lineno, "dev-import", f"production code imports from {mod}"))
        elif isinstance(node, ast.Call):
            name = _dotted(node.func)
            if name:
                head, _, rest = name.partition(".")
                full = aliases.get(head, head) + ("." + rest if rest else "")
                if full in ("__import__", "importlib.import_module") and node.args \
                        and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    cat = _category_of(node.args[0].value)        # literal name: judge the module itself
                    if cat:
                        hits.append(Hit(rel, node.lineno, cat, f"dynamic import {node.args[0].value}"))
                    continue
                for pat, cat in RISKY_CALLS.items():
                    if (pat.endswith(".") and full.startswith(pat)) or full == pat:
                        hits.append(Hit(rel, node.lineno, cat, f"call {full}()"))
                        break
            for kw in node.keywords:
                if kw.arg == "allow_origins" and "'*'" in ast.unparse(kw.value):
                    hits.append(Hit(rel, node.lineno, "exposure", "CORS allows any origin (allow_origins=['*'])"))
        elif isinstance(node, ast.Constant) and node.value == ALL_INTERFACES:
            hits.append(Hit(rel, node.lineno, "exposure", "binds a server to every network interface ('0.0.0.0'); "
                                                          "use 127.0.0.1"))
    return hits


def scan_web(path: Path, rel: str) -> List[Hit]:
    hits = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        for rx, what in WEB_PATTERNS:
            if rx.search(line):
                hits.append(Hit(rel, lineno, "web-external", what))
    return hits


def iter_python_files(root: Path):
    for p in sorted(root.rglob("*.py")):
        rel_parts = p.relative_to(root).parts
        if any(part in EXCLUDED_DIRS for part in rel_parts[:-1]):
            continue
        yield p, "/".join(rel_parts)


def run_audit(root) -> List[Hit]:
    root = Path(root).resolve()
    hits: List[Hit] = []
    for p, rel in iter_python_files(root):
        hits += scan_python(p, rel)
    for g in WEB_GLOBS:
        for p in sorted(root.glob(g)):
            hits += scan_web(p, p.relative_to(root).as_posix())
    for h in hits:
        if h.category in ("dev-import", "exposure", "parse-error"):
            continue                                   # never allowlisted
        for pglob, dglob, reason in ALLOWLIST:
            if fnmatch.fnmatch(h.path, pglob) and fnmatch.fnmatch(h.detail, dglob):
                h.allowed_reason = reason
                break
    return hits
