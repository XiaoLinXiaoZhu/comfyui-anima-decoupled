"""comfyui-anima-decoupled: decoupled source/target text for Anima conditioning."""

from .nodes import AnimaDecoupledConditioning

__version__ = "1.0.1"

NODE_CLASS_MAPPINGS = {
    "AnimaDecoupledConditioning": AnimaDecoupledConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaDecoupledConditioning": "Anima Decoupled Conditioning",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
