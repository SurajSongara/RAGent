"""Stage handler implementations.

Importing this package registers every handler. The worker imports it once at
startup so `get_handler` can resolve any stage; without that import the registry
is empty and every message dead-letters as unhandled.
"""

from ragent.ingest.stages import detect, enrich, index, parse  # noqa: F401

__all__ = ["detect", "enrich", "index", "parse"]
