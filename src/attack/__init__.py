"""Script generating adversarial examples using YOLO model and FGSM(Fast Gradient Signed Method)."""

__version__ = "0.1.0"

# Define the __all__ variable
__all__ = ["fgsm"]

# Import the submodules
from . import fgsm
