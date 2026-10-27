"""PyArrow utility routines with safe extension unregistration."""
import pyarrow as pa


def sanitize_arrow_registry() -> None:
    """Safely unregisters legacy pyarrow extension types without throwing KeyError."""
    try:
        pa.unregister_extension_type("arrow.py_extension_type")
    except (pa.ArrowKeyError, KeyError):
        pass
    except Exception:
        pass
