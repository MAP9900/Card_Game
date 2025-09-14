import os
import time
from functools import wraps
from typing import Any, Callable, Iterable, Optional, Union


PathLike = Union[str, os.PathLike]

# Simplified decorator: only runtime + sizes for returned file paths
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



#Over Complicated wrapper for calculating function run time and output sizes. 

# def _walk_filesizes(paths: Iterable[PathLike]) -> dict[str, int]:
#     """Return a mapping of absolute file paths -> size (bytes) for all files under given paths.
#     Each item in `paths` may be a file or a directory (recursed).
#     Non-existent paths are ignored.
#     """
#     out: dict[str, int] = {}
#     for p in paths:
#         if p is None:
#             continue
#         ap = os.path.abspath(os.fspath(p))
#         if not os.path.exists(ap):
#             continue
#         if os.path.isfile(ap):
#             try:
#                 out[ap] = os.path.getsize(ap)
#             except OSError:
#                 pass
#         else:
#             for root, _, files in os.walk(ap):
#                 for f in files:
#                     fp = os.path.join(root, f)
#                     try:
#                         out[fp] = os.path.getsize(fp)
#                     except OSError:
#                         pass
#     return out


# def _normalize_returned_paths(retval: Any) -> list[str]:
#     """Try to interpret a function's return value as one or more file paths.
#     Supports str, os.PathLike, list/tuple of paths. Otherwise returns empty list.
#     """
#     if isinstance(retval, (str, os.PathLike)):
#         return [os.fspath(retval)]
#     if isinstance(retval, (list, tuple)):
#         out: list[str] = []
#         for x in retval:
#             if isinstance(x, (str, os.PathLike)):
#                 out.append(os.fspath(x))
#         return out
#     return []


# def track_perf(
#     watch_paths: Optional[Iterable[PathLike]] = None,
#     returns_paths: bool = False,
# ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
#     """
#     Decorator that times a function and reports file sizes for outputs.

#     - watch_paths: optional iterable of files/dirs to snapshot before/after the call.
#                    Reports new/changed files created within these paths.
#     - returns_paths: if True, treats the function's return value as a path or
#                      collection of paths and reports their sizes.

#     Prints a concise report to stdout and returns the function's original result.
#     """
#     def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
#         @wraps(func)
#         def wrapper(*args: Any, **kwargs: Any) -> Any:
#             before: dict[str, int] = {}
#             if watch_paths:
#                 before = _walk_filesizes(watch_paths)

#             t0 = time.perf_counter()
#             result = func(*args, **kwargs)
#             elapsed = (time.perf_counter() - t0) * 1000.0  # ms

#             print(f"[track_perf] {func.__name__} elapsed: {elapsed:.2f} ms")

#             # Report returned paths (if any)
#             if returns_paths:
#                 paths = _normalize_returned_paths(result)
#                 if paths:
#                     for p in paths:
#                         ap = os.path.abspath(os.fspath(p))
#                         try:
#                             size = os.path.getsize(ap)
#                             print(f"[track_perf] output: {ap} ({size} bytes)")
#                         except OSError:
#                             print(f"[track_perf] output: {ap} (missing)")

#             # Report changes under watched paths
#             if watch_paths:
#                 after = _walk_filesizes(watch_paths)
#                 # New files
#                 new_files = {p: sz for p, sz in after.items() if p not in before}
#                 # Changed sizes
#                 changed = {p: (before[p], after[p]) for p in after if p in before and before[p] != after[p]}

#                 if new_files:
#                     print(f"[track_perf] new files: {len(new_files)}")
#                     for p, sz in sorted(new_files.items()):
#                         print(f"  + {p} ({sz} bytes)")
#                 if changed:
#                     print(f"[track_perf] modified files: {len(changed)}")
#                     for p, (old, new) in sorted(changed.items()):
#                         delta = new - old
#                         print(f"  ~ {p} ({old} -> {new} bytes, {delta:+d})")

#             return result

#         return wrapper

#     return decorator

