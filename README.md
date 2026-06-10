# ClearView — Color Vision Accessibility Tool

A Streamlit web app that simulates how images appear to people with three types of color vision deficiency (color blindness) and provides Daltonize correction to improve accessibility.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![License](https://img.shields.io/badge/License-MIT-green)

**Author:** Alfons Patrick M

## Features

- **Color Blindness Simulation** — Accurate LMS color space transforms for:
  - Protanopia (no red/L cones)
  - Deuteranopia (no green/M cones)
  - Tritanopia (no blue/S cones)
- **Daltonize Correction** — Shifts lost color information into visible channels
- **Download** — Export any simulated or corrected image as PNG
- **Sample Images** — Built-in test images for quick demos

## App Flow

1. **Homepage** — Choose which color vision deficiency to explore
2. **Simulation Page** — Upload an image and see three views side-by-side:
   - **Original** — The uploaded image
   - **Simulated** — How a color-blind person sees it
   - **Daltonized** — Corrected for accessibility

Each simulation page includes download buttons and educational info about the deficiency.

## How It Works

```
RGB Image → flatten to (N, 3) → matrix multiply by deficiency transform → reshape → display
```

The simulation pipeline:
1. Convert RGB to LMS (Long, Medium, Short cone response) color space
2. Apply a deficiency matrix that collapses the missing cone axis
3. Convert back to RGB

Daltonize does the inverse: computes the perceptual error (what's lost in simulation) and redistributes it into the remaining visible channels.

## File Structure

```
clearview/
├── __init__.py     # Package init
├── app.py          # Streamlit UI — multi-page (Home + 3 simulation pages)
├── simulate.py     # LMS color transform matrices + simulate()
├── recolor.py      # Daltonize channel-shift correction
├── home-icon.png   # Sidebar home navigation icon
└── samples/        # Fallback test images
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| Image Processing | NumPy, Pillow |
| Color Science | LMS color space (Hunt-Pointer-Estévez) |
| Deployment | Streamlit Cloud |

## References

- Brettel, Viénot & Mollon (1997) — *Computerized simulation of color appearance for dichromats*
- Machado, Oliveira & Fernandes (2009) — *A physiologically-based model for simulation of color vision deficiency*
- Fidaner, Ozguven & El-Nasr (2006) — *Analysis of Color Blindness (Daltonize)*

## License

MIT — Alfons Patrick M
