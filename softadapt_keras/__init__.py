# ruff: noqa: F401
"""Second level module import for SoftAdapt."""

from softadapt_keras.algorithms import (
    LossWeightedSoftAdapt,
    NormalizedSoftAdapt,
    SoftAdapt,
)
from softadapt_keras.constants import _finite_difference_constants, _stability_constants
from softadapt_keras.utilities import _finite_difference

# # adding package information and version
# try:
#     import importlib.metadata as importlib_metadata
# except ModuleNotFoundError:
#     import importlib_metadata

# package_name = "softadapt_keras"
# __version__ = importlib_metadata.version(package_name)
