"""
app.py — ClearView Streamlit UI

Multi-page layout:
  - Homepage  (no sidebar) — pick a color blindness type
  - Per-type  (with sidebar) — upload image, see Original | Simulated | Daltonized
"""

import io
import sys
from pathlib import Path

import streamlit.components.v1 as components

# Ensure the parent directory is on sys.path so that
# `clearview` is importable when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import streamlit as st
from PIL import Image

from clearview.simulate import simulate, contrast_score, DEFICIENCY_LABELS
from clearview.recolor import daltonize

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────

# Load logo for page config
LOGO_PATH = Path(__file__).parent / "logo.png"
LOGO_WHITE_PATH = Path(__file__).parent / "logo_white.png"
logo_image = Image.open(LOGO_PATH) if LOGO_PATH.exists() else "👁️"
logo_white_image = Image.open(LOGO_WHITE_PATH) if LOGO_WHITE_PATH.exists() else logo_image

st.set_page_config(
    page_title="ClearView — Color Vision Accessibility Tool",
    page_icon=logo_image,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# Session-state navigation
# ──────────────────────────────────────────────

if "page" not in st.session_state:
    st.session_state.page = "home"


def go_home():
    st.session_state.page = "home"


def go_to(deficiency: str):
    st.session_state.page = deficiency


page = st.session_state.page
is_home = page == "home"

# ──────────────────────────────────────────────
# Custom CSS — dynamic based on page
# ──────────────────────────────────────────────

SIDEBAR_HIDE_CSS = """
section[data-testid="stSidebar"] {
    display: none !important;
}
div[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}
"""

SIDEBAR_SHOW_CSS = """
section[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
    min-width: 280px !important;
    width: 280px !important;
    transform: translateX(0px) !important;
    margin-left: 0px !important;
    display: flex !important;
    visibility: visible !important;
}
section[data-testid="stSidebar"] > div:first-child {
    width: 280px !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li {
    color: #c4c9d4;
}
/* Hide the collapse/expand toggle buttons */
button[data-testid="stSidebarCollapseButton"],
button[kind="headerNoPadding"],
div[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}
"""

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Global */
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

/* Main background */
.stApp {{
    background: linear-gradient(160deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
}}

/* Sidebar: conditionally show/hide */
{SIDEBAR_HIDE_CSS if is_home else SIDEBAR_SHOW_CSS}

/* Contrast badge */
.contrast-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 99px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.08);
    margin-top: 6px;
    margin-bottom: 2px;
}}
.badge-excellent {{
    background: linear-gradient(135deg, rgba(16,185,129,0.18), rgba(16,185,129,0.08));
    color: #34d399;
    border-color: rgba(16,185,129,0.25);
}}
.badge-good {{
    background: linear-gradient(135deg, rgba(59,130,246,0.18), rgba(59,130,246,0.08));
    color: #60a5fa;
    border-color: rgba(59,130,246,0.25);
}}
.badge-moderate {{
    background: linear-gradient(135deg, rgba(251,191,36,0.18), rgba(251,191,36,0.08));
    color: #fbbf24;
    border-color: rgba(251,191,36,0.25);
}}
.badge-poor {{
    background: linear-gradient(135deg, rgba(239,68,68,0.18), rgba(239,68,68,0.08));
    color: #f87171;
    border-color: rgba(239,68,68,0.25);
}}

/* Section header */
.section-header {{
    font-size: 1.35rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #a78bfa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
    letter-spacing: -0.01em;
}}

/* Hero header */
.hero-title {{
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.15;
    letter-spacing: -0.03em;
    margin-bottom: 0;
}}
.hero-subtitle {{
    font-size: 1.1rem;
    color: #94a3b8;
    font-weight: 400;
    margin-top: 2px;
    margin-bottom: 24px;
}}

/* Card */
.sim-card {{
    background: rgba(30, 27, 75, 0.45);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(12px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}}
.sim-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.12);
    border-color: rgba(99, 102, 241, 0.2);
}}
.sim-card-title {{
    font-size: 1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 10px;
}}

/* Info banner */
.info-banner {{
    background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(168,85,247,0.08));
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 16px 20px;
    color: #c4b5fd;
    font-size: 0.9rem;
    line-height: 1.6;
}}

/* Divider */
.gradient-divider {{
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.3), rgba(168,85,247,0.3), transparent);
    border: none;
    margin: 28px 0;
    border-radius: 1px;
}}

/* Column labels */
.col-label {{
    text-align: center;
    font-size: 0.95rem;
    font-weight: 600;
    color: #cbd5e1;
    padding: 8px 0 4px 0;
    letter-spacing: 0.01em;
}}

/* Download button */
.stDownloadButton > button {{
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}}
.stDownloadButton > button:hover {{
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(102, 126, 234, 0.35) !important;
}}

/* File uploader */
div[data-testid="stFileUploader"] section {{
    border: 2px dashed rgba(99, 102, 241, 0.35);
    border-radius: 16px;
    background: rgba(30, 27, 75, 0.3);
    padding: 32px;
    transition: border-color 0.3s ease, background 0.3s ease;
}}
div[data-testid="stFileUploader"] section:hover {{
    border-color: rgba(99, 102, 241, 0.6);
    background: rgba(30, 27, 75, 0.45);
}}

/* Watermark */
.watermark {{
    text-align: center;
    color: rgba(148, 163, 184, 0.5);
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    padding: 24px 0 12px 0;
    font-weight: 500;
}}

/* Hide Streamlit branding */
#MainMenu {{visibility: hidden;}}
header {{visibility: hidden;}}
footer {{visibility: hidden;}}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

SAMPLE_DIR = Path(__file__).parent / "samples"

DEFICIENCY_INFO = {
    "protanopia": {
        "label": "Protanopia",
        "short": "No red (L) cones",
        "color_dot": "#ef4444",
        "desc": (
            "Protanopia is a type of red-green color blindness where the "
            "long-wavelength (L) cones are absent. People with protanopia "
            "cannot perceive red light and often confuse reds with greens, "
            "browns, and dark shades."
        ),
        "prevalence": "~1.3% of males",
    },
    "deuteranopia": {
        "label": "Deuteranopia",
        "short": "No green (M) cones",
        "color_dot": "#22c55e",
        "desc": (
            "Deuteranopia is the most common form of color blindness. "
            "The medium-wavelength (M) cones are missing, making it hard "
            "to distinguish between red and green hues. It affects daily "
            "tasks like reading traffic lights and choosing ripe fruit."
        ),
        "prevalence": "~1.2% of males",
    },
    "tritanopia": {
        "label": "Tritanopia",
        "short": "No blue (S) cones",
        "color_dot": "#3b82f6",
        "desc": (
            "Tritanopia is a rare form of color blindness where the "
            "short-wavelength (S) cones are absent. Blues appear greenish, "
            "and yellows appear violet or light grey. It affects both "
            "males and females equally."
        ),
        "prevalence": "~0.01% of population",
    },
}




def _image_to_bytes(img_array: np.ndarray, fmt: str = "PNG") -> bytes:
    """Convert numpy image to downloadable bytes."""
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _load_image(source) -> np.ndarray | None:
    """Load image from upload or file path, return RGB numpy array."""
    try:
        img = Image.open(source).convert("RGB")
        return np.array(img)
    except Exception:
        return None


def _render_watermark():
    """Render the author watermark."""
    st.markdown(
        '<div class="watermark">Alfons Patrick M - 2026</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# Sidebar (only rendered on simulation pages)
# ──────────────────────────────────────────────

if not is_home:
    # JS: click the sidebar expand button if it is currently collapsed
    components.html(
        """
        <script>
        (function tryExpand() {
            var p = window.parent.document;
            var btn = p.querySelector('[data-testid="stSidebarCollapsedControl"] button');
            if (btn) {
                btn.click();
            } else {
                setTimeout(tryExpand, 150);
            }
        })();
        </script>
        """,
        height=0,
        scrolling=False,
    )

    with st.sidebar:
        st.markdown(
            '<p class="hero-title" style="font-size:1.8rem;">ClearView</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="hero-subtitle" style="font-size:0.9rem;">'
            "Color Vision Accessibility Tool</p>",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        st.markdown("**Simulations**")
        for key, info in DEFICIENCY_INFO.items():
            is_active = (page == key)
            label = f"{'> ' if is_active else ''}{info['label']}"
            if st.button(
                label,
                key=f"nav_{key}",
                use_container_width=True,
            ):
                go_to(key)
                st.rerun()

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="info-banner">'
            "<strong>How it works</strong><br>"
            "ClearView uses LMS color space transforms to "
            "simulate how images appear to people with color "
            "vision deficiencies. The Daltonize feature shifts "
            "lost color information into visible channels."
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        _render_watermark()


# ──────────────────────────────────────────────
# Page: HOME (no sidebar, centered layout)
# ──────────────────────────────────────────────

def render_home():
    """Render the homepage with deficiency type selection cards."""

    st.markdown("")

    col_l, col_c, col_r = st.columns([1, 3, 1])
    with col_c:
        if LOGO_WHITE_PATH.exists() or LOGO_PATH.exists():
            logo_cols = st.columns([1, 1, 1])
            with logo_cols[1]:
                st.image(logo_white_image, use_container_width=True)
                
        st.markdown(
            '<p class="hero-title" style="text-align:center; font-size:3.2rem;">'
            "ClearView"
            "</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="hero-subtitle" style="text-align:center; font-size:1.2rem;">'
            "See the world through different eyes"
            "</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="info-banner" style="text-align:center; font-size:1rem; '
            'max-width:600px; margin:0 auto;">'
            "Choose a color vision deficiency below to simulate how your "
            "images appear, and see the Daltonize correction side-by-side."
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown("")

    # ── Three selection cards ──
    card_cols = st.columns(3)

    for col, (key, info) in zip(card_cols, DEFICIENCY_INFO.items()):
        with col:
            st.markdown(
                f'<div class="sim-card" style="text-align:center; min-height:220px;">'
                f'<div style="width:40px; height:40px; border-radius:50%; '
                f'background:{info["color_dot"]}; margin:0 auto 14px auto; '
                f'box-shadow: 0 0 16px {info["color_dot"]}44;"></div>'
                f'<div class="sim-card-title" style="font-size:1.15rem;">'
                f'{info["label"]}</div>'
                f'<div style="color:#94a3b8; font-size:0.85rem; margin-bottom:4px;">'
                f'{info["short"]}</div>'
                f'<div style="color:#64748b; font-size:0.78rem;">'
                f'Prevalence: {info["prevalence"]}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
            if st.button(
                f"Explore {info['label']}",
                key=f"card_{key}",
                use_container_width=True,
            ):
                go_to(key)
                st.rerun()

    st.markdown("")

    # ── Features row ──
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    feat_cols = st.columns(3)
    features = [
        ("Accurate Simulation", "Scientifically-derived LMS color space transforms"),
        ("Daltonize Correction", "Shifts lost color info into visible channels"),
        ("Contrast Scoring", "Luminance-based contrast preservation metric"),
    ]
    for col, (title, desc) in zip(feat_cols, features):
        with col:
            st.markdown(
                f'<div class="sim-card" style="text-align:center; min-height:110px;">'
                f'<div class="sim-card-title">{title}</div>'
                f'<div style="color:#94a3b8; font-size:0.82rem;">{desc}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

    _render_watermark()


# ──────────────────────────────────────────────
# Page: SIMULATION (one per deficiency type)
# ──────────────────────────────────────────────

def render_simulation(deficiency: str):
    """Render the simulation page for a specific deficiency type."""

    info = DEFICIENCY_INFO[deficiency]

    # ── Page header ──
    st.markdown(
        f'<p class="section-header" style="font-size:1.8rem;">'
        f'{info["label"]} — {info["short"]}'
        f"</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="info-banner" style="margin-bottom:20px;">'
        f'{info["desc"]}<br><br>'
        f'<strong>Prevalence:</strong> {info["prevalence"]}'
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # ── Image upload ──
    uploaded_file = st.file_uploader(
        "Upload an image to analyze",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        help="Supported formats: PNG, JPG, JPEG, WebP, BMP",
        key=f"upload_{deficiency}",
    )

    # Sample image fallback
    sample_images = sorted(SAMPLE_DIR.glob("*")) if SAMPLE_DIR.exists() else []
    image_rgb = None

    if uploaded_file is not None:
        image_rgb = _load_image(uploaded_file)
    elif sample_images:
        st.markdown("**Or try a sample image:**")
        sample_cols = st.columns(len(sample_images))
        for sc, sample_path in zip(sample_cols, sample_images):
            with sc:
                if st.button(
                    sample_path.stem.replace("_", " ").title(),
                    key=f"sample_{deficiency}_{sample_path.stem}",
                    use_container_width=True,
                ):
                    st.session_state[f"selected_sample_{deficiency}"] = sample_path

        if f"selected_sample_{deficiency}" in st.session_state:
            image_rgb = _load_image(st.session_state[f"selected_sample_{deficiency}"])

    if image_rgb is None:
        st.markdown("")
        st.markdown(
            '<div class="sim-card" style="text-align:center; padding:48px 20px;">'
            '<div style="color:#94a3b8; font-size:1rem;">'
            "Upload an image above to see the simulation"
            "</div></div>",
            unsafe_allow_html=True,
        )
        _render_watermark()
        return

    # ── Process ──
    sim_img = simulate(image_rgb, deficiency)
    dal_img = daltonize(image_rgb, deficiency)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # ── Three-column display: Original | Simulated | Daltonized ──
    st.markdown(
        f'<p class="section-header">Comparison — {info["label"]}</p>',
        unsafe_allow_html=True,
    )

    col_orig, col_sim, col_dal = st.columns(3)

    with col_orig:
        st.markdown(
            '<div class="col-label">Original</div>',
            unsafe_allow_html=True,
        )
        st.image(image_rgb, width="stretch")
        h, w, _ = image_rgb.shape
        st.markdown(
            f'<div style="color:#64748b; font-size:0.78rem; text-align:center;">'
            f"{w} x {h} px</div>",
            unsafe_allow_html=True,
        )

    with col_sim:
        st.markdown(
            f'<div class="col-label">{info["label"]} Simulation</div>',
            unsafe_allow_html=True,
        )
        st.image(sim_img, width="stretch")
        st.download_button(
            "Download Simulation",
            data=_image_to_bytes(sim_img),
            file_name=f"clearview_{deficiency}_simulation.png",
            mime="image/png",
            key=f"dl_sim_{deficiency}",
            use_container_width=True,
        )

    with col_dal:
        st.markdown(
            '<div class="col-label">Daltonized (Corrected)</div>',
            unsafe_allow_html=True,
        )
        st.image(dal_img, width="stretch")
        st.download_button(
            "Download Daltonized",
            data=_image_to_bytes(dal_img),
            file_name=f"clearview_{deficiency}_daltonized.png",
            mime="image/png",
            key=f"dl_dal_{deficiency}",
            use_container_width=True,
        )

    _render_watermark()


# ──────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────

if page == "home":
    render_home()
elif page in DEFICIENCY_INFO:
    render_simulation(page)
else:
    st.session_state.page = "home"
    render_home()
