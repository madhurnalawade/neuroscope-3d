"""Logging and message hygiene.

Rules enforced here:
- file names appear only as sanitised short names (no directories, no odd characters, truncated);
- exceptions from third-party libraries are reported by class name only, because their text can
  contain absolute paths (user names) or header contents;
- only local logging handlers (stream / file) are allowed; remote handlers are refused.
"""
import logging
import logging.handlers
import re
from pathlib import PurePath

MAX_SHORT_NAME = 40
_ALLOWED = re.compile(r"[^A-Za-z0-9._-]")

# Handlers that send log records off the machine.
REMOTE_HANDLER_TYPES = (
    logging.handlers.SocketHandler,       # also covers DatagramHandler
    logging.handlers.HTTPHandler,
    logging.handlers.SMTPHandler,
    logging.handlers.SysLogHandler,       # may target a remote syslog server
    logging.handlers.QueueHandler,        # destination unknown; refuse rather than guess
)


def safe_name(name, max_len: int = MAX_SHORT_NAME) -> str:
    """Short display name: last path component only, characters outside [A-Za-z0-9._-] replaced by '_',
    shortened in the middle when longer than ``max_len``. Never returns directories."""
    text = str(name).replace("\\", "/")
    base = PurePath(text).name or "unnamed"
    base = _ALLOWED.sub("_", base)
    if len(base) > max_len:
        keep = max_len - 3
        base = base[: keep // 2] + "..." + base[-(keep - keep // 2):]
    return base


def describe_exception(exc: BaseException) -> str:
    """Exception class name only; the message is dropped because it can contain paths or data."""
    return type(exc).__name__


def remote_handlers(logger: logging.Logger = None):
    """Return every handler on ``logger`` (default: root) and its ancestors that can send records elsewhere."""
    found = []
    lg = logger or logging.getLogger()
    while lg is not None:
        found += [h for h in lg.handlers if isinstance(h, REMOTE_HANDLER_TYPES)]
        lg = lg.parent if lg.propagate else None
    return found


def configure_local_logging(level: int = logging.INFO, log_file=None) -> logging.Logger:
    """Configure the 'neuroscope' logger with local handlers only (stderr, optional file) and refuse to run
    if a remote handler is attached anywhere up the chain."""
    logger = logging.getLogger("neuroscope")
    logger.setLevel(level)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(h)
        if log_file is not None:
            fh = logging.FileHandler(str(log_file), encoding="utf-8")
            fh.setFormatter(h.formatter)
            logger.addHandler(fh)
    bad = remote_handlers(logger)
    if bad:
        raise RuntimeError("Remote logging handler(s) attached: "
                           + ", ".join(type(h).__name__ for h in bad) + ". Remove them before handling scans.")
    return logger
