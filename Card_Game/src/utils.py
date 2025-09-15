import os
import time
from functools import wraps
from typing import Any, Callable, Union


PathLike = Union[str, os.PathLike]

#Decorator for run times and file sizes
def time_and_size(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Minimal decorator: prints runtime and, if the function returns a path
    (or list/tuple of paths), prints the size of each file. Returns the
    function's original result unchanged.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        print(f"[time_and_size] {func.__name__} elapsed: {elapsed:.2f} ms")

        paths: list[str] = []
        if isinstance(result, (str, os.PathLike)):
            paths = [os.fspath(result)]
        elif isinstance(result, (list, tuple)):
            for x in result:
                if isinstance(x, (str, os.PathLike)):
                    paths.append(os.fspath(x))
        for p in paths:
            ap = os.path.abspath(p)
            try:
                size = os.path.getsize(ap)
                print(f"[time_and_size] saved: {ap} ({size} bytes)")
            except OSError:
                print(f"[time_and_size] saved: {ap} (missing)")
        return result
    return wrapper

