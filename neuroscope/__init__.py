"""NeuroScope 3D data-input layer: validation and loading (io), upload storage and deletion (storage),
leak-prevention checks (security) and Decathlon dataset tools for development only (dev).

Production code must never import ``neuroscope.dev`` (checked by scripts/audit_no_leaks.py).
Research and teaching prototype; not a diagnostic tool.
"""
