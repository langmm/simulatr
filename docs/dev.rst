============
Development
============

Links
-----

- `conda-forge simulatr feedstock <https://github.com/conda-forge/simulatr-feedstock>`_
- `simulatr on PyPI <https://pypi.org/project/simulatr/>`_

Release
-------

1. Create a branch for development
1. Make code changes
1. Ensure tests pass
1. Update version in recipe/recipe.yaml using symantic versioning
1. Create a pull request to merge in changes
1. Merge changes after tests pass
1. Publish a relase on Github and create a tag on release that matches the version in recipe/recipe.yaml with the "v" prefix
1. Verify that the publish-to-pypi GitHub action publishes to PyPI
1. Update the simulatr-feedstock repo to publish to the conda-forge

.. include:: ../TODO.md
   :parser: myst_parser.sphinx_

