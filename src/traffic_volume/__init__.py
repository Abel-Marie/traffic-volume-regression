"""Hourly traffic volume on I-94: the ML process (regression) as an installable package.

Rule for every module in this package: sibling modules are imported by name
(``from .plotting import finish_figure``), never module-qualified, and every
top-level name is unique across the package. The self-contained notebook
relies on this: it holds a verbatim copy of the functions it uses.
"""

__version__ = "1.0.0"
