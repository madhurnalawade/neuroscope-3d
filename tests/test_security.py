"""Network guard, static audit and log/message hygiene."""
import logging
import logging.handlers
import socket
import urllib.request
from pathlib import Path

import pytest

from neuroscope.security import netguard
from neuroscope.security.hygiene import configure_local_logging, remote_handlers, safe_name
from neuroscope.security.static_audit import run_audit

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.allow_attempts
def test_guard_blocks_sockets_dns_and_urllib():
    with netguard.network_guard() as attempts:
        with pytest.raises(netguard.NetworkBlockedError):
            socket.create_connection(("127.0.0.1", 9))
        with pytest.raises(netguard.NetworkBlockedError):
            socket.getaddrinfo("example.com", 443)
        s = socket.socket()
        try:
            with pytest.raises(netguard.NetworkBlockedError):
                s.connect(("192.0.2.1", 80))
        finally:
            s.close()
        with pytest.raises(Exception) as exc:
            urllib.request.urlopen("http://example.com/", timeout=1)
        assert "NetworkBlocked" in repr(exc.value) or isinstance(exc.value.__context__, netguard.NetworkBlockedError) \
            or "network guard" in str(exc.value)
    ops = {op for op, _, _ in attempts}
    assert {"socket.create_connection", "socket.getaddrinfo", "socket.connect"} <= ops


@pytest.mark.allow_attempts
@pytest.mark.parametrize("mod", ["requests", "httpx"])
def test_guard_blocks_http_libraries_if_installed(mod):
    lib = pytest.importorskip(mod)
    with netguard.network_guard() as attempts:
        with pytest.raises(Exception):
            lib.get("http://example.com/", timeout=1)
    assert attempts, f"{mod} reached the network without the guard noticing"


@pytest.mark.allow_attempts
def test_guard_allows_only_in_process_socketpair():
    with netguard.network_guard() as attempts:
        a, b = socket.socketpair()                  # asyncio's self-pipe; on Windows built from loopback sockets
        try:
            a.sendall(b"ping")
            assert b.recv(4) == b"ping"
        finally:
            a.close()
            b.close()
        with pytest.raises(netguard.NetworkBlockedError):
            socket.create_connection(("127.0.0.1", 9))
    assert [op for op, _, _ in attempts] == ["socket.create_connection"]


def test_guard_restores_socket_api():
    with netguard.network_guard():
        pass
    assert netguard.is_enabled()                    # still on: the autouse fixture keeps it enabled
    netguard.disable()
    try:
        assert socket.getaddrinfo is not None and not netguard.is_enabled()
    finally:
        netguard.enable()


def test_static_audit_finds_planted_problems(tmp_path):
    pkg = tmp_path / "neuroscope" / "io"
    pkg.mkdir(parents=True)
    (pkg / "bad.py").write_text(
        "import requests\nimport subprocess as sp\nfrom ..dev import msd\nimport sentry_sdk\nfrom openai import OpenAI\n"
        "sp.run(['curl', 'x'])\nimport os\nos.system('x')\nHOST = '0.0.0.0'\n", encoding="utf-8")
    static = tmp_path / "Input" / "neuroscope_3d" / "backend" / "app" / "static"
    static.mkdir(parents=True)
    (static / "index.html").write_text('<script src="https://cdn.example.com/x.js"></script>', encoding="utf-8")
    hits = run_audit(tmp_path)
    cats = sorted({h.category for h in hits if not h.ok})
    assert cats == ["dev-import", "exposure", "llm", "network", "process", "telemetry", "web-external"]
    assert all(not h.ok for h in hits)


def test_production_code_does_not_import_dev():
    bad = [h for h in run_audit(REPO) if h.category == "dev-import"]
    assert not bad, bad


def test_neuroscope_package_has_no_unexplained_network_code():
    hits = [h for h in run_audit(REPO) if h.path.startswith(("neuroscope/", "scripts/", "tests/")) and not h.ok]
    assert not hits, [f"{h.path}:{h.line} {h.detail}" for h in hits]


def test_safe_name_strips_directories_and_odd_characters():
    assert safe_name("C:\\Users\\alice\\scans\\Doe^John t1.nii.gz") == "Doe_John_t1.nii.gz"
    assert safe_name("/home/bob/x/../a;b.nii") == "a_b.nii"
    long = safe_name("a" * 100 + "_t1.nii.gz")
    assert len(long) == 40 and long.endswith("_t1.nii.gz")


def test_remote_log_handlers_refused():
    lg = logging.getLogger("neuroscope")
    h = logging.handlers.HTTPHandler("example.invalid", "/log")
    lg.addHandler(h)
    try:
        assert remote_handlers(lg)
        with pytest.raises(RuntimeError):
            configure_local_logging()
    finally:
        lg.removeHandler(h)
    assert not remote_handlers(lg)
