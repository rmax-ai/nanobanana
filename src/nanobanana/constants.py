# =============================================================================
# Constants
# =============================================================================


class ExitCode:
    SUCCESS = 0
    INTERNAL = 1
    INVALID_ARGS = 2
    INPUT_FAILURE = 3
    CAPABILITY_MISMATCH = 4
    AUTH_FAILURE = 5
    QUOTA_EXCEEDED = 6
    SAFETY_REFUSAL = 7
    EMPTY_RESPONSE = 8
    OUTPUT_FAILURE = 9
    PARTIAL_BATCH = 10


SUPPORTED_ASPECT_RATIOS: tuple[str, ...] = (
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
)
FLASH_ONLY_ASPECT_RATIOS: tuple[str, ...] = ("1:4", "4:1", "1:8", "8:1")
SUPPORTED_SIZES: tuple[str, ...] = ("0.5K", "1K", "2K", "4K")
SUPPORTED_FORMATS: tuple[str, ...] = ("png", "jpeg")
MIME_MAP: dict[str, str] = {"png": "image/png", "jpeg": "image/jpeg"}


# =============================================================================
# Capability Registry
# =============================================================================

CAPABILITIES: dict[str, dict] = {
    "gemini-3.1-flash-lite-image": {
        "alias": "lite",
        "supported_sizes": ("1K",),
        "max_references": 14,
        "character_consistency": False,
        "style_references": False,
        "grounding": False,
        "thinking_levels": (),
        "image_search": False,
    },
    "gemini-3.1-flash-image": {
        "alias": "flash",
        "supported_sizes": ("0.5K", "1K", "2K", "4K"),
        "max_references": 10,
        "character_consistency": True,
        "style_references": True,
        "grounding": True,
        "thinking_levels": ("minimal", "high"),
        "image_search": True,
    },
    "gemini-3-pro-image": {
        "alias": "pro",
        "supported_sizes": ("1K", "2K", "4K"),
        "max_references": 6,
        "character_consistency": True,
        "style_references": True,
        "grounding": True,
        "thinking_levels": (),
        "image_search": False,
    },
    "gemini-2.5-flash-image": {
        "alias": "legacy",
        "supported_sizes": ("1K",),
        "max_references": 4,
        "character_consistency": False,
        "style_references": False,
        "grounding": False,
        "thinking_levels": (),
        "image_search": False,
    },
}

ALIAS_TO_MODEL: dict[str, str] = {v["alias"]: k for k, v in CAPABILITIES.items()}


# =============================================================================
# Presets
# =============================================================================

PRESETS: dict[str, dict[str, str]] = {
    "architecture-diagram": {
        "model": "pro",
        "aspect": "16:9",
        "size": "2K",
        "thinking": "high",
        "prompt_prefix": "Create a technically precise software architecture diagram...",
    },
    "icon": {
        "model": "lite",
        "aspect": "1:1",
        "size": "1K",
        "prompt_prefix": "Create a simple, clean icon...",
    },
    "photo": {"model": "flash", "aspect": "3:2", "size": "2K"},
    "portrait": {"model": "flash", "aspect": "2:3", "size": "2K"},
    "product": {"model": "pro", "aspect": "1:1", "size": "2K"},
    "editorial": {"model": "pro", "aspect": "4:5", "size": "2K"},
    "sticker": {"model": "lite", "aspect": "1:1", "size": "1K"},
    "logo-concept": {"model": "pro", "aspect": "1:1", "size": "2K", "thinking": "high"},
    "social-card": {"model": "flash", "aspect": "16:9", "size": "1K"},
    "slide-hero": {"model": "pro", "aspect": "16:9", "size": "4K", "thinking": "high"},
    "infographic": {"model": "pro", "aspect": "9:16", "size": "4K", "thinking": "high"},
    "character-sheet": {"model": "flash", "aspect": "2:3", "size": "2K"},
    "storyboard": {"model": "flash", "aspect": "16:9", "size": "2K"},
    "texture": {"model": "lite", "aspect": "1:1", "size": "1K"},
    "background": {"model": "flash", "aspect": "16:9", "size": "2K"},
    "thumbnail": {"model": "lite", "aspect": "16:9", "size": "1K"},
    "wireframe": {"model": "flash", "aspect": "16:9", "size": "1K"},
}


# =============================================================================
# Cost Estimates
# =============================================================================

COST_TABLE: dict[str, dict[str, float]] = {
    "gemini-3.1-flash-lite-image": {"1K": 0.0336},
    "gemini-3.1-flash-image": {"0.5K": 0.045, "1K": 0.067, "2K": 0.101, "4K": 0.151},
    "gemini-3-pro-image": {"1K": 0.067, "2K": 0.101, "4K": 0.151},
    "gemini-2.5-flash-image": {"1K": 0.0336},
}
COST_RETRIEVAL_DATE: str = "2026-07-01"
