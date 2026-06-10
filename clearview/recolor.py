"""
recolor.py — Daltonize algorithm for color-blind accessibility.

Shifts color information from deficient channels into channels that
color-blind viewers can still perceive. Based on the method by
Fidaner, Ozguven & El-Nasr (2006).

Pipeline:
  1. Simulate the deficiency to get sim_image
  2. Compute error = original - sim_image  (lost information)
  3. Redistribute error into visible channels via a correction matrix
  4. Add corrected error back to original
"""

import numpy as np
from numpy.typing import NDArray

from clearview.simulate import simulate


# ──────────────────────────────────────────────
# Error redistribution matrices
# These shift the "lost" color error from the deficient
# channel into the two remaining perceptible channels.
# ──────────────────────────────────────────────

_CORRECTION_MATRICES = {
    # Protanopia — lost red info → shift into green & blue
    "protanopia": np.array([
        [0.0, 0.0, 0.0],
        [0.7, 1.0, 0.0],
        [0.7, 0.0, 1.0],
    ], dtype=np.float64),

    # Deuteranopia — lost green info → shift into red & blue
    "deuteranopia": np.array([
        [1.0, 0.7, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.7, 1.0],
    ], dtype=np.float64),

    # Tritanopia — lost blue info → shift into red & green
    "tritanopia": np.array([
        [1.0, 0.0, 0.7],
        [0.0, 1.0, 0.7],
        [0.0, 0.0, 0.0],
    ], dtype=np.float64),
}


def daltonize(
    image_rgb: NDArray[np.uint8],
    deficiency: str,
) -> NDArray[np.uint8]:
    """
    Apply Daltonize correction to make an image more distinguishable
    for viewers with the specified color vision deficiency.

    Parameters
    ----------
    image_rgb : ndarray, shape (H, W, 3), dtype uint8
        Input image in sRGB.
    deficiency : str
        One of 'protanopia', 'deuteranopia', 'tritanopia'.

    Returns
    -------
    ndarray, shape (H, W, 3), dtype uint8
        Corrected image with boosted distinguishability.
    """
    if deficiency not in _CORRECTION_MATRICES:
        raise ValueError(
            f"Unknown deficiency '{deficiency}'. "
            f"Choose from: {list(_CORRECTION_MATRICES.keys())}"
        )

    h, w, _ = image_rgb.shape
    correction_mat = _CORRECTION_MATRICES[deficiency]

    # Step 1: Simulate the deficiency
    sim = simulate(image_rgb, deficiency)

    # Step 2: Compute perceptual error (what's lost)
    original_f = image_rgb.astype(np.float64)
    error = original_f.reshape(-1, 3) - sim.astype(np.float64).reshape(-1, 3)

    # Step 3: Redistribute error into visible channels
    corrected_error = error @ correction_mat.T

    # Step 4: Add correction to original
    result = original_f.reshape(-1, 3) + corrected_error
    result = np.clip(result, 0.0, 255.0)

    return result.astype(np.uint8).reshape(h, w, 3)
