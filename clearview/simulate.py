"""
simulate.py — Color blindness simulation via LMS color space transforms.

Pipeline: RGB → LMS → apply deficiency matrix → RGB
Uses scientifically-derived 3×3 matrices from Brettel, Viénot & Mollon (1997)
and Machado, Oliveira & Fernandes (2009).
"""

import numpy as np
from numpy.typing import NDArray


# ──────────────────────────────────────────────
# Step 1: RGB ↔ LMS conversion matrices
# Based on sRGB → Hunt-Pointer-Estévez LMS
# ──────────────────────────────────────────────

_RGB_TO_LMS = np.array([
    [0.31399022, 0.63951294, 0.04649755],
    [0.15537241, 0.75789446, 0.08670142],
    [0.01775239, 0.10944209, 0.87256922],
], dtype=np.float64)

_LMS_TO_RGB = np.linalg.inv(_RGB_TO_LMS)


# ──────────────────────────────────────────────
# Step 2: LMS-domain deficiency simulation matrices
# These zero-out or collapse the missing cone response
# onto the remaining two cone axes.
# ──────────────────────────────────────────────

_LMS_DEFICIENCY = {
    # Protanopia — missing L cones (red)
    "protanopia": np.array([
        [0.0, 1.05118294, -0.05116099],
        [0.0, 1.0,         0.0        ],
        [0.0, 0.0,         1.0        ],
    ], dtype=np.float64),

    # Deuteranopia — missing M cones (green)
    "deuteranopia": np.array([
        [1.0,         0.0, 0.0        ],
        [0.9513092,   0.0, 0.04866992 ],
        [0.0,         0.0, 1.0        ],
    ], dtype=np.float64),

    # Tritanopia — missing S cones (blue)
    "tritanopia": np.array([
        [1.0,          0.0,         0.0],
        [0.0,          1.0,         0.0],
        [-0.86744736,  1.86727089,  0.0],
    ], dtype=np.float64),
}

# Pre-compute full RGB→simulate→RGB matrices for speed
SIMULATION_MATRICES: dict[str, NDArray[np.float64]] = {}
for _name, _lms_mat in _LMS_DEFICIENCY.items():
    SIMULATION_MATRICES[_name] = _LMS_TO_RGB @ _lms_mat @ _RGB_TO_LMS

# Human-readable labels
DEFICIENCY_LABELS = {
    "protanopia":   "Protanopia (no red cones)",
    "deuteranopia": "Deuteranopia (no green cones)",
    "tritanopia":   "Tritanopia (no blue cones)",
}


def simulate(
    image_rgb: NDArray[np.uint8],
    deficiency: str,
) -> NDArray[np.uint8]:
    """
    Simulate color-blind vision for a given deficiency type.

    Parameters
    ----------
    image_rgb : ndarray, shape (H, W, 3), dtype uint8
        Input image in sRGB color space.
    deficiency : str
        One of 'protanopia', 'deuteranopia', 'tritanopia'.

    Returns
    -------
    ndarray, shape (H, W, 3), dtype uint8
        Simulated image.
    """
    if deficiency not in SIMULATION_MATRICES:
        raise ValueError(
            f"Unknown deficiency '{deficiency}'. "
            f"Choose from: {list(SIMULATION_MATRICES.keys())}"
        )

    h, w, _ = image_rgb.shape
    mat = SIMULATION_MATRICES[deficiency]

    # Flatten to (N, 3), normalize to [0, 1]
    pixels = image_rgb.reshape(-1, 3).astype(np.float64) / 255.0

    # Apply the combined RGB→sim→RGB matrix
    simulated = pixels @ mat.T

    # Clip and convert back
    simulated = np.clip(simulated, 0.0, 1.0)
    return (simulated * 255.0).astype(np.uint8).reshape(h, w, 3)


def contrast_score(
    original: NDArray[np.uint8],
    simulated: NDArray[np.uint8],
) -> float:
    """
    Compute a simple perceptual contrast preservation score (0–100).

    Compares luminance variance retention between the original
    and simulated images. 100 = identical contrast, 0 = total loss.
    """
    # BT.601 luminance weights
    lum_weights = np.array([0.299, 0.587, 0.114], dtype=np.float64)

    lum_orig = (original.astype(np.float64) / 255.0) @ lum_weights
    lum_sim  = (simulated.astype(np.float64) / 255.0) @ lum_weights

    var_orig = np.var(lum_orig)
    var_sim  = np.var(lum_sim)

    if var_orig < 1e-10:
        return 100.0  # Uniform image — no contrast to lose

    score = min(var_sim / var_orig, 1.0) * 100.0
    return round(score, 1)
