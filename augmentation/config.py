from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class AugmentationConfig:
    # How many distinct text variants to generate per original episode.
    # Each text variant is paired with one visual variant → total new episodes
    # = n_text_variants * n_visual_variants (+ 1 if include_original=True).
    n_text_variants: int = 5
    n_visual_variants: int = 3

    # --- Visual augmentation bounds ---

    # Multiplicative brightness factor
    brightness_range: Tuple[float, float] = (0.75, 1.25)
    # Multiplicative contrast factor
    contrast_range: Tuple[float, float] = (0.80, 1.20)
    # Multiplicative saturation factor (PIL "Color" enhancer)
    saturation_range: Tuple[float, float] = (0.80, 1.20)
    # Hue shift in degrees — kept tight so bowl colors stay distinguishable
    hue_range: Tuple[int, int] = (-8, 8)
    # Gamma correction exponent (applied as pixel^(1/gamma))
    gamma_range: Tuple[float, float] = (0.85, 1.15)
    # Gaussian noise std on [0, 255] scale
    noise_std_max: float = 6.0
    # Probability of applying a mild Gaussian blur (kernel radius 1–2)
    blur_prob: float = 0.3
    # Probability of applying a mild shadow overlay
    shadow_prob: float = 0.2
    # Translation as fraction of image dimension (applied identically to all frames)
    translate_range: Tuple[float, float] = (-0.03, 0.03)

    # --- Output ---

    # Keep the original (unmodified) episode in the output dataset
    include_original: bool = True
    # Episodes per chunk folder (mirrors LeRobot convention)
    chunks_size: int = 1000

    # --- Reproducibility ---
    seed: int = 42

    # --- Hugging Face Hub ---
    push_to_hub: bool = False
    hub_repo_id: str = ""
    hub_private: bool = True
