"""ClinKey design system — centralized color tokens + light/dark CSS.

Brand colors (extracted from the logo) are the single source of truth:

  navy   #003060   deep primary
  deep   #002040
  blue   #0060A0
  accent #0070B0   primary action color (light)
  cyan   #0080C0
  teal   #35A7C8   primary accent (dark)

Light and dark themes derive ALL semantic tokens from these same hues, so
both themes read as the same Clinkey product. Nothing is ever styled with
an ad-hoc hex — every component references a token below.
"""
from __future__ import annotations

# ---- brand palette (do NOT change these) ----
NAVY = "#003060"
DEEP = "#002040"
BLUE = "#0060A0"
ACCENT = "#0070B0"
CYAN = "#0080C0"
TEAL = "#35A7C8"
MINT = "#6FD1D8"

# ============================================================ LIGHT THEME
LIGHT = {
    # brand actions
    "primary": "#0070B0",          # ACCENT
    "primary_hover": "#005F96",
    "primary_active": "#004E7C",
    "primary_disabled": "#9CC3DB",
    "primary_text": "#FFFFFF",
    "accent": "#0070B0",
    "accent_hover": "#0080C0",
    "link": "#0060A0",

    # surfaces
    "bg": "#F4FAFD",
    "surface": "#FFFFFF",           # cards
    "surface_secondary": "#EAF4FA", # chips / subtle fills
    "sidebar_bg": "#EAF4FA",
    "hover_bg": "#DDEBF4",

    # text
    "text": "#0F1B2D",              # primary text
    "text_secondary": "#3B4A5E",
    "muted": "#5A6B7F",
    "placeholder": "#6B7C90",
    "icon": "#3B4A5E",              # icons/chevrons (light)

    # borders
    "border": "#D6E6F0",
    "border_strong": "#AEC7D8",

    # inputs
    "input_bg": "#FFFFFF",
    "input_border": "#C4D7E3",
    "input_border_focus": "#0070B0",
    "input_text": "#0F1B2D",
    "input_disabled_bg": "#EFF4F8",

    # chat
    "bubble_user": "#0070B0",
    "bubble_user_text": "#FFFFFF",
    "bubble_assistant": "#FFFFFF",
    "bubble_assistant_text": "#0F1B2D",
    "bubble_assistant_border": "#E1ECF3",

    # status
    "success": "#0E7A52",
    "success_bg": "#E6F4EC",
    "warning": "#9A5B00",
    "warning_bg": "#FBF0DE",
    "error": "#B3261E",
    "error_bg": "#FBE9E7",
    "info": "#0060A0",
    "info_bg": "#E7F2F9",

    # misc
    "code_bg": "#EEF3F7",
    "code_text": "#0F1B2D",
    "focus_ring": "rgba(0,112,176,0.35)",
    "chip": "#EAF4FA",
    "overlay": "rgba(15,27,45,0.06)",
    "good": "#0E7A52",
    "warn": "#9A5B00",
    "bad": "#B3261E",
}

# ============================================================ DARK THEME
DARK = {
    # brand actions
    "primary": "#35A7C8",           # TEAL accent — readable on dark navy
    "primary_hover": "#4BB7D6",
    "primary_active": "#2E93B1",
    "primary_disabled": "#2A5A6B",
    "primary_text": "#04222E",      # dark text on light-teal button
    "accent": "#35A7C8",
    "accent_hover": "#4BB7D6",
    "link": "#6FD1D8",

    # surfaces
    "bg": "#0A1622",
    "surface": "#11222F",           # cards
    "surface_secondary": "#1A3346", # chips / subtle fills
    "sidebar_bg": "#0E1D2A",
    "hover_bg": "#1B3448",

    # text
    "text": "#E6EFF5",
    "text_secondary": "#C4D3E0",
    "muted": "#9DB2C4",
    "placeholder": "#7A93A8",
    "icon": "#FFFFFF",              # icons/chevrons (white in dark mode)

    # borders
    "border": "#1E3A4E",
    "border_strong": "#33526A",

    # inputs
    "input_bg": "#0F2030",
    "input_border": "#2A4559",
    "input_border_focus": "#35A7C8",
    "input_text": "#E6EFF5",
    "input_disabled_bg": "#12242F",

    # chat
    "bubble_user": "#0070B0",
    "bubble_user_text": "#FFFFFF",
    "bubble_assistant": "#142B3A",
    "bubble_assistant_text": "#E6EFF5",
    "bubble_assistant_border": "#22435A",

    # status
    "success": "#4BD79A",
    "success_bg": "#123528",
    "warning": "#F0A64B",
    "warning_bg": "#3A2A14",
    "error": "#F0716B",
    "error_bg": "#3A1B1B",
    "info": "#6FBDE0",
    "info_bg": "#122A3A",

    # misc
    "code_bg": "#0C1A26",
    "code_text": "#E6EFF5",
    "focus_ring": "rgba(53,167,200,0.4)",
    "chip": "#1A3346",
    "overlay": "rgba(0,0,0,0.25)",
    "good": "#4BD79A",
    "warn": "#F0A64B",
    "bad": "#F0716B",
}


def theme_colors(dark: bool) -> dict:
    return DARK if dark else LIGHT


def inject_theme_css(dark: bool) -> str:
    """Return a <style> block restyling Streamlit for the selected theme."""
    c = theme_colors(dark)
    return f"""
<style>
:root {{
  --ck-primary: {c['primary']};
  --ck-bg: {c['bg']};
  --ck-surface: {c['surface']};
  --ck-text: {c['text']};
  --ck-muted: {c['muted']};
  --ck-border: {c['border']};
  --ck-accent: {c['accent']};
  --ck-focus: {c['focus_ring']};
}}

/* ============================================================ base */
.stApp {{ background: {c['bg']}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.stApp {{ color: {c['text']}; }}

/* Links */
a {{ color: {c['link']}; }}
a:hover {{ color: {c['accent']}; text-decoration: underline; }}

/* Headings / text — scoped to the app content ONLY. Streamlit's own chrome
   (the top-right "⋮" menu, Deploy dropdown, tooltips, etc.) renders in
   portals OUTSIDE .stApp with light backgrounds, so we must NOT force their
   text white (that made it invisible in dark mode). */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stApp p, .stApp li, .stApp label, .stApp span {{ color: {c['text']}; }}
hr {{ border-color: {c['border']}; }}
code, pre {{ background: {c['code_bg']}; color: {c['code_text']}; }}
pre {{ border: 1px solid {c['border']}; border-radius: 8px; }}

/* ============================================================ sidebar */
section[data-testid="stSidebar"] {{
  background: {c['sidebar_bg']};
  border-right: 1px solid {c['border']};
}}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {{ color: {c['text']}; }}

/* Header toolbar (Deploy / Running status) — visible in dark mode. These
   labels live in the app header (inside .stApp) which is transparent over
   the dark background, so they must be light. */
[data-testid="stToolbarActionButtonLabel"],
[data-testid="stStatusWidget"],
[data-testid="stMainMenuButton"] {{
  color: {c['text']} !important;
}}

/* Sidebar collapse/expand chevrons — keep visible in dark mode.
   These icons are font-based Material glyphs (stIconMaterial) recolored with
   `color`/`-webkit-text-fill-color` (not `fill`). The wrapper elements are
   NOT <button> nodes, so selectors must not require a button tag. */
[data-testid="stSidebarCollapseButton"],
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {{
  color: {c['icon']} !important;
  background: transparent;
}}
[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"],
[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"],
[data-testid="stSidebarCollapsedControl"] [data-testid="stIconMaterial"],
[data-testid="collapsedControl"] [data-testid="stIconMaterial"] {{
  color: {c['icon']} !important;
  -webkit-text-fill-color: {c['icon']} !important;
}}
[data-testid="stSidebarCollapseButton"] svg,
[data-testid="stSidebarCollapseButton"] svg *,
[data-testid="stExpandSidebarButton"] svg,
[data-testid="stExpandSidebarButton"] svg *,
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="stSidebarCollapsedControl"] svg *,
[data-testid="collapsedControl"] svg,
[data-testid="collapsedControl"] svg * {{
  fill: {c['icon']} !important;
  color: {c['icon']} !important;
  stroke: {c['icon']} !important;
}}

/* Expander chevrons — font glyph + any svg fallback */
[data-testid="stExpanderIcon"],
[data-testid="stExpanderIconCheck"],
[data-testid="stExpanderIconError"],
[data-testid="stExpanderIconSpinner"],
[data-testid="stExpander"] summary [data-testid="stIconMaterial"] {{
  color: {c['icon']} !important;
  -webkit-text-fill-color: {c['icon']} !important;
}}
[data-testid="stExpander"] summary svg,
[data-testid="stExpander"] summary svg * {{
  fill: {c['icon']} !important;
  color: {c['icon']} !important;
  stroke: {c['icon']} !important;
}}

/* ============================================================ buttons */
.stButton > button {{
  background: transparent;
  color: {c['text']};
  border: 1px solid {c['border_strong']};
  border-radius: 10px;
  transition: background .15s ease, border-color .15s ease, color .15s ease, box-shadow .15s ease;
}}
.stButton > button:hover {{
  background: {c['hover_bg']};
  color: {c['text']};
  border-color: {c['accent']};
}}
.stButton > button:active {{
  background: {c['surface_secondary']};
  border-color: {c['accent']};
}}
.stButton > button:focus,
.stButton > button:focus-visible {{
  outline: none;
  box-shadow: 0 0 0 3px {c['focus_ring']};
  color: {c['text']};
}}
.stButton > button:disabled {{
  background: {c['input_disabled_bg']};
  color: {c['placeholder']};
  border-color: {c['border']};
  cursor: not-allowed;
}}

/* Primary buttons */
.stButton > button[kind="primary"],
.stButton > button[kind="primaryFormSubmit"] {{
  background: {c['primary']};
  color: {c['primary_text']};
  border: 1px solid transparent;
}}
.stButton > button[kind="primary"]:hover,
.stButton > button[kind="primaryFormSubmit"]:hover {{
  background: {c['primary_hover']};
  color: {c['primary_text']};
}}
.stButton > button[kind="primary"]:active,
.stButton > button[kind="primaryFormSubmit"]:active {{
  background: {c['primary_active']};
}}
.stButton > button[kind="primary"]:focus,
.stButton > button[kind="primaryFormSubmit"]:focus,
.stButton > button[kind="primary"]:focus-visible,
.stButton > button[kind="primaryFormSubmit"]:focus-visible {{
  box-shadow: 0 0 0 3px {c['focus_ring']};
  color: {c['primary_text']};
}}
.stButton > button[kind="primary"]:disabled,
.stButton > button[kind="primaryFormSubmit"]:disabled {{
  background: {c['primary_disabled']};
  color: {c['primary_text']};
  opacity: 0.7;
}}

/* ============================================================ inputs */
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div,
div[data-baseweb="select"] > div {{
  background: {c['input_bg']} !important;
  border-color: {c['input_border']} !important;
  border-radius: 12px;
}}
div[data-baseweb="input"]:focus-within > div,
div[data-baseweb="textarea"]:focus-within > div,
div[data-baseweb="select"]:focus-within > div {{
  border-color: {c['input_border_focus']} !important;
  box-shadow: 0 0 0 3px {c['focus_ring']};
}}
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea,
div[data-baseweb="select"] input,
div[data-baseweb="select"] div[role="listbox"] {{
  color: {c['input_text']} !important;
}}
div[data-baseweb="input"] input::placeholder,
div[data-baseweb="textarea"] textarea::placeholder {{
  color: {c['placeholder']} !important;
  opacity: 1;
}}
div[data-baseweb="select"] span {{
  color: {c['text']} !important;
}}
/* select dropdown list */
ul[data-baseweb="menu"], ul[role="listbox"] {{
  background: {c['surface']} !important;
}}
li[data-baseweb="menu"] div,
li[role="option"] {{
  color: {c['text']} !important;
  background: {c['surface']} !important;
}}
li[data-baseweb="menu"]:hover div,
li[role="option"]:hover {{
  background: {c['hover_bg']} !important;
}}

/* ============================================================ chat input (form) */
div[data-testid="stForm"] {{
  background: transparent;
  border: none;
  padding: 0;
}}
[data-testid="stChatInput"] {{ background: transparent; border: none; }}
[data-testid="stChatInput"] > div {{
  background: {c['input_bg']};
  border: 1px solid {c['input_border']};
  border-radius: 12px;
}}
[data-testid="stChatInput"] > div:focus-within {{
  border-color: {c['input_border_focus']};
  box-shadow: 0 0 0 3px {c['focus_ring']};
}}
[data-testid="stChatInput"] textarea {{
  color: {c['input_text']} !important;
  caret-color: {c['accent']};
}}
[data-testid="stChatInput"] textarea::placeholder {{
  color: {c['placeholder']} !important;
  opacity: 1;
}}

/* ============================================================ chat messages */
[data-testid="stChatMessage"] {{
  background: transparent;
  border: none;
  padding: 0.3rem 0;
}}
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"],
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] {{
  background: {c['accent']};
}}
/* User message bubble */
.ck-msg-user {{
  background: {c['bubble_user']};
  color: {c['bubble_user_text']};
  border-radius: 16px;
  padding: 0.6rem 0.95rem;
  display: inline-block;
  max-width: 86%;
  line-height: 1.5;
}}
/* Assistant message body — strong contrast, subtle card */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {{
  color: {c['bubble_assistant_text']};
}}

/* ============================================================ expanders */
[data-testid="stExpander"] {{
  background: {c['surface']};
  border: 1px solid {c['border']};
  border-radius: 10px;
}}
[data-testid="stExpander"] summary {{
  color: {c['text']};
}}
[data-testid="stExpander"] summary:hover {{ color: {c['accent']}; }}
[data-testid="stExpander"] summary:focus-visible {{
  box-shadow: 0 0 0 3px {c['focus_ring']};
  border-radius: 8px;
}}
[data-testid="stExpander"] p, [data-testid="stExpander"] span,
[data-testid="stExpander"] div, [data-testid="stExpander"] li {{ color: {c['text']}; }}
[data-testid="stExpander"] svg {{ fill: {c['text']}; }}

/* ============================================================ popover (the "⋮" menu)
   Streamlit popovers open as a light overlay by default. In dark mode that
   light square contains light text (Rename / Delete) which becomes invisible.
   Give the popover body a themed dark surface so its content is readable. */
[data-testid="stPopoverBody"] {{
  background: {c['surface']} !important;
  color: {c['text']} !important;
  border: 1px solid {c['border_strong']} !important;
  border-radius: 12px;
  box-shadow: 0 6px 24px {c['overlay']};
}}
[data-testid="stPopoverBody"] p,
[data-testid="stPopoverBody"] span,
[data-testid="stPopoverBody"] label,
[data-testid="stPopoverBody"] div {{
  color: {c['text']} !important;
}}
[data-testid="stPopoverButton"] {{
  color: {c['text']} !important;
  background: transparent !important;
  border: 1px solid {c['border_strong']} !important;
  border-radius: 8px;
}}
[data-testid="stPopoverButton"]:hover {{
  background: {c['hover_bg']} !important;
}}
[data-testid="stPopoverButton"] [data-testid="stIconMaterial"] {{
  color: {c['icon']} !important;
  -webkit-text-fill-color: {c['icon']} !important;
}}

/* ============================================================ tabs */
button[data-baseweb="tab"] {{
  color: {c['muted']};
  background: transparent;
}}
button[data-baseweb="tab"]:hover {{ color: {c['text']}; }}
button[data-baseweb="tab"][aria-selected="true"] {{
  color: {c['accent']};
  border-bottom: 2px solid {c['accent']};
}}
div[data-baseweb="tab-highlight"],
div[data-baseweb="tab-border"] {{ background: {c['accent']}; }}

/* ============================================================ checkbox / radio / toggle */
div[data-baseweb="checkbox"] label span,
div[data-baseweb="radio"] label span {{
  color: {c['text']};
}}
div[data-baseweb="checkbox"] label div[role="checkbox"],
div[data-baseweb="radio"] label div[role="radio"] {{
  border-color: {c['border_strong']} !important;
  background: {c['input_bg']} !important;
}}
div[data-baseweb="checkbox"] label div[role="checkbox"][aria-checked="true"],
div[data-baseweb="radio"] label div[role="radio"][aria-checked="true"] {{
  background: {c['accent']} !important;
  border-color: {c['accent']} !important;
}}
div[data-baseweb="checkbox"] label div[role="checkbox"]:focus-visible,
div[data-baseweb="radio"] label div[role="radio"]:focus-visible {{
  box-shadow: 0 0 0 3px {c['focus_ring']};
}}

/* ============================================================ metrics / chips */
[data-testid="stMetric"] {{
  background: {c['surface']};
  border: 1px solid {c['border']};
  border-radius: 10px;
  padding: 0.6rem 0.8rem;
}}
[data-testid="stMetricLabel"] {{ color: {c['muted']}; }}
[data-testid="stMetricValue"] {{ color: {c['accent']}; }}

.ck-chip {{
  display: inline-block;
  background: {c['chip']};
  color: {c['text']};
  border: 1px solid {c['border']};
  border-radius: 999px;
  padding: 0.15rem 0.6rem;
  margin: 0.1rem 0.25rem 0.1rem 0;
  font-size: 0.78rem;
}}
.ck-disclaimer {{
  color: {c['muted']};
  font-size: 0.78rem;
  line-height: 1.4;
  padding: 0.4rem 0.2rem;
}}
.ck-cite {{ color: {c['accent']}; font-weight: 600; }}

/* ============================================================ file uploader */
[data-testid="stFileUploaderDropzone"] {{
  background: {c['input_bg']};
  border: 1.5px dashed {c['input_border']};
  border-radius: 10px;
  padding: 0.3rem 0.35rem;
  min-height: 2.4rem;
}}
[data-testid="stFileUploaderDropzone"]:hover {{
  border-color: {c['accent']};
}}
[data-testid="stFileUploaderDropzoneInstructions"] {{ color: {c['placeholder']}; font-size: 0.75rem; }}
[data-testid="stFileUploaderDropzoneInstructions"] button,
[data-testid="stFileUploaderDropzone"] button {{
  background: {c['primary']};
  color: {c['primary_text']};
  border: none;
  border-radius: 6px;
  font-size: 0.78rem;
  padding: 0.25rem 0.5rem;
}}
[data-testid="stFileUploaderDropzoneInstructions"] button:hover,
[data-testid="stFileUploaderDropzone"] button:hover {{
  background: {c['primary_hover']};
  color: {c['primary_text']};
}}
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small {{ color: {c['muted']}; }}
[data-testid="stFileUploader"] label p {{ font-size: 0.85rem; color: {c['text']}; }}
[data-testid="stFileUploader"] svg {{ fill: {c['muted']}; }}

/* ============================================================ status messages
   Streamlit alerts use baseweb's own per-kind colors (light pastel
   backgrounds with dark text) which remain readable in BOTH themes.
   We only round the corners and add a border — we do NOT override the
   background/text, to avoid dark-on-dark or light-on-light mismatches. */
div[data-testid="stAlert"] {{
  border-radius: 10px;
  border: 1px solid {c['border']} !important;
}}
div[data-testid="stAlertTitle"],
div[data-testid="stAlert"] a {{
  font-weight: 600;
}}

/* ============================================================ spinner / progress */
[data-testid="stSpinner"] {{ color: {c['accent']}; }}
[data-testid="stProgress"] > div > div > div > div {{ background: {c['accent']}; }}
[data-testid="stProgress"] > div > div {{ background: {c['surface_secondary']}; }}

/* ============================================================ welcome / hero */
.ck-welcome-title {{
  font-size: 2.6rem;
  font-weight: 800;
  text-align: center;
  letter-spacing: -0.015em;
  line-height: 1.1;
  color: {c['text']};
  margin-top: 0.4rem;
}}
.ck-welcome-sub {{
  text-align: center;
  font-size: 1.18rem;
  font-weight: 500;
  letter-spacing: 0.01em;
  color: {c['text_secondary']};
  margin-top: 0.45rem;
}}
.ck-welcome-desc {{
  text-align: center;
  color: {c['muted']};
  font-size: 0.98rem;
  line-height: 1.7;
  max-width: 520px;
  margin: 0.85rem auto 0.25rem;
}}
.ck-examples-label {{
  text-align: center;
  color: {c['muted']};
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin: 1.75rem 0 0.8rem;
}}
.ck-logo-wrap {{
  display: flex;
  justify-content: center;
  margin: 0 0 0.5rem;
}}

/* ============================================================ conversation header */
.ck-conv-header {{
  display: flex;
  align-items: baseline;
  gap: 0.7rem;
  margin-bottom: 0.4rem;
}}
.ck-conv-title {{
  font-size: 1.1rem;
  font-weight: 700;
  color: {c['text']};
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}
.ck-conv-note {{ color: {c['muted']}; font-size: 0.8rem; }}

/* ============================================================ empty / misc */
.ck-empty {{
  color: {c['muted']};
  font-size: 0.85rem;
  line-height: 1.5;
  text-align: center;
  padding: 1.2rem 0.4rem;
}}
.ck-sources {{ margin-top: 0.6rem; }}
</style>
"""