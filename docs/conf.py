# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../src'))
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'odc-loader'
copyright = '2024, ODC'
author = 'ODC'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Intersphinx mapping -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-intersphinx_mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'xarray': ('https://xarray.pydata.org/en/stable/', None),
}

# -- Napoleon settings -------------------------------------------------------
# https://www.sphinx-doc.org/en/master/sphinx.ext.napoleon.html#configuration
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Autodoc settings --------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
