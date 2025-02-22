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
project = 'kessler'
copyright = "2020-2025, Kessler contributors"
author = 'Giacomo Acciarini, Atılım Güneş Baydin, Francesco Pinto'


# The full version, including alpha/beta/rc tags
import kessler

release = kessler.__version__



# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["myst_nb",
              "sphinx.ext.intersphinx",
              "sphinx.ext.autodoc",
              "sphinx.ext.autosummary",
              "sphinx.ext.doctest",
              ]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}


autoclass_content = 'both'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_logo = "_static/kessler_logo.png"

html_theme_options = {
    "repository_url": "https://github.com/kesslerlib/kessler/",
    "repository_branch": "master",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_issues_button": True,
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "notebook_interface": "jupyterlab"
    },
    "navigation_with_keys": False,
}

nb_execution_mode = "force"

nb_execution_excludepatterns = ['basics.ipynb','cdms_analysis_and_plotting.ipynb','LSTM_training.ipynb']

# Force printing traceback to stderr on execution error.
nb_execution_show_tb = True

# Set a longer timeout for notebook execution.
nb_execution_timeout = 120

latex_engine = "xelatex"

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
