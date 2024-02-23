# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
from shutil import copyfile

# import sys
# sys.path.insert(0, os.path.abspath('.'))
import datetime

# -- General configuration ---------------------------------------------------
# for parsing markdown files
# pip install recommonmark
from recommonmark.parser import CommonMarkParser

source_parsers = {".md": CommonMarkParser}

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Project information -----------------------------------------------------

project = "OWAI"
copyright = "2020-{}, Creare".format(datetime.datetime.now().year)
author = "Creare"

# import podpac for versioning
import owai.version

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = owai.version.semver()

# The full version, including alpha/beta/rc tags.
release = owai.version.version()

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "sphinx.ext.githubpages",
    "sphinx.ext.doctest",
    "recommonmark",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "default"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# generate autosummary files into the :toctree: directory
#   See http://www.sphinx-doc.org/en/master/ext/autosummary.html
# unfortunately this inherits all members of a class and no parameters below will help
#   See https://github.com/sphinx-doc/sphinx/pull/4029
# Chose to use templates in the _templates directory to override this
autosummary_generate = True

# autodoc options
autoclass_content = "class"  # only include docstring from Class (not __init__ method)
autodoc_inherit_docstrings = False
autodoc_default_options = {
    "members": None,  # means yes/true/on
    "undoc-members": None,
    "show-inheritance": None,
}

# -- Options for HTML output -------------------------------------------------

html_title = release

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "alabaster"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


def copy_changelog():
    """copy the changelog from the root of the repository"""

    path_to_changelog = "../../CHANGELOG.md"
    filepath = os.path.join(os.path.join(os.path.dirname(__file__), os.path.normpath(path_to_changelog)))
    destpath = os.path.join(os.path.join(os.path.dirname(__file__), "changelog.md"))
    # copy file to current directory
    copyfile(filepath, destpath)


def setup(app):
    copy_changelog()
