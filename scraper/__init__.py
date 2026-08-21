"""FincaRaiz pipeline scraper.

An explicit, sequential, resumable pipeline. Each stage reads its
predecessor's artifact from disk and writes its own, so any stage can be
run in isolation and the whole run survives interruption.

Run everything with a single command::

    make scrape
    python -m scraper run --stage all --operation arriendo --city bogota
"""

__version__ = "1.0.0"
