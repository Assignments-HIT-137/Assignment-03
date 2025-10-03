"""
Utility decorators for HIT137 Assignment 3.

Why this file?
- The brief asks to demonstrate *multiple decorators*.  We provide two simple,
  dependency‑free decorators you can stack on functions/methods to satisfy that.
- Used by models.py to log calls and measure inference time.

Usage:
    from utils import logger, timed

    @logger
    @timed
    def run(...):
        ...

Both decorators are intentionally minimal and safe for a teaching context.
"""
from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable


def logger(func: Callable) -> Callable:
    """Print a short log line whenever *func* is called.

    This helps the marker see that decorators are active and is also handy
    when debugging Tkinter callbacks.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        # Preview at most a couple of arguments to keep logs readable.
        preview_pos = ", ".join(repr(a) for a in args[:2])
        preview_kw  = ", ".join(f"{k}={v!r}" for k, v in list(kwargs.items())[:2])
        preview = ", ".join(x for x in (preview_pos, preview_kw) if x)
        print(f"[LOG] {func.__name__}({preview}...)")
        return func(*args, **kwargs)
    return wrapper


def timed(func: Callable) -> Callable:
    """Report how long *func* took to run (seconds, milliseconds precision)."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        t0 = time.perf_counter()
        out = func(*args, **kwargs)
        dt = time.perf_counter() - t0
        print(f"[TIME] {func.__name__}: {dt:.3f}s")
        return out
    return wrapper
