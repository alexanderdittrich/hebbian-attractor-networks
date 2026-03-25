from .adaptive_es import AdaptiveES
from .cma_es import CMA_ES
from .es import ESComma
from .openai_es import OpenAI_ES

strategies = {
    "cma-es": CMA_ES,
    "openai-es": OpenAI_ES,
    "es": ESComma,
    "adaptive-es": AdaptiveES,
}
