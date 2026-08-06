"""Sphinx configuration for the simulatr documentation."""

import os
import sys
from importlib.metadata import version as _get_version

sys.path.insert(0, os.path.abspath(".."))

project = "simulatr"
copyright = "2026, Meagan Lang"
author = "Meagan Lang"

try:
    release = _get_version("simulatr")
except Exception:
    release = "0.0.1"
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autosummary_generate = True

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "gymnasium": ("https://gymnasium.farama.org/", None),
}

html_theme = "alabaster"
