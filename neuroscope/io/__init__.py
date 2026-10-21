"""Production four-file input path: sequence assignment, per-file validation, cross-file checks and the
canonical case object. Imports nothing from neuroscope.dev."""
from .case import CanonicalCase, LoadResult, load_four_files  # noqa: F401
from .errors import Issue  # noqa: F401
from .sequences import SEQUENCES, assign_sequences  # noqa: F401
from .validate import FileCheck, validate_file  # noqa: F401
