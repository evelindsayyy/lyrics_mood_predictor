"""
LyricMood — paste song lyrics, get mood + explanation + 5 similar songs.

This file is a PURE API client: it holds zero ML code. All inference lives
behind the LyricMood HTTP API (FastAPI), and this Streamlit app is only a UI
container that renders whatever the API returns.

    Classification + explanation → POST /v1/predict
        (mood, confidence, probabilities, explanation tokens, model_version)
    Retrieval (similar songs)    → POST /v1/similar
        (results with title, artist, score)

The single backend dependency is HTTP: the API base URL comes from the
LYRICMOOD_API_URL env var (default http://localhost:8000). No joblib / sklearn /
shap / sentence-transformers / pandas / numpy / src.* imports remain here — the
model artifacts are owned by the API service, not the UI.

UI follows DESIGN_HANDOFF.md: design tokens + component styles live in
app/static/lyricmood.css; this file just wires Streamlit widgets to those
class hooks (.brand, .prompt, .paper, .mood-word, .conf, .probs, .shap, .similar).

Run with: `streamlit run app/streamlit_app.py` from the project root, with the
API reachable at LYRICMOOD_API_URL — or `docker compose up` to run both.

AI attribution: this file is the most AI-assisted piece of the project.
I created the visual design upfront — LyricMood Minimal.html, lyricmood.css,
DESIGN_HANDOFF.md, the design tokens — plus the data-flow spec, the pivot to a
client/server split, and the @st.cache_resource strategy (now caching an httpx
client instead of models). Claude wrote the bulk of the Streamlit Python and the
CSS overrides that re-skin Streamlit's built-in widgets to match my design
(chipbar layout, raw-HTML SHAP chart, song-list grid, set_mood_accent helper),
plus the API-client wiring and error handling. I integrated, tested, and
iterated on the result. See ../ATTRIBUTION.md for the full breakdown.
"""

import math
import os
from pathlib import Path

import httpx
import streamlit as st


# ------------------------------------------------------------
# design system — load CSS + Google Fonts once at app start
# (swapped st.markdown → st.html for <style>/<link> so Streamlit's
# markdown sanitizer doesn't strip them)
# ------------------------------------------------------------

CSS_PATH = Path(__file__).parent / "static" / "lyricmood.css"

FONTS_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?'
    'family=Fraunces:ital,wght@0,300;0,400;1,300;1,400'
    '&family=IBM+Plex+Mono:wght@400;500'
    '&family=Inter:wght@300;400;500'
    '&family=Noto+Serif+Hebrew:wght@400'
    '&family=Noto+Naskh+Arabic:wght@400'
    '&display=swap" rel="stylesheet">'
)


@st.cache_resource
def _load_css() -> str:
    """Load the CSS and strip the header /* ... */ comment, which contains
    literal '<style>' and '</style>' example snippets — those would close
    my wrapping <style> tag early and cause the rest of the CSS to render
    as body text."""
    raw = CSS_PATH.read_text(encoding="utf-8")
    # drop the leading /* ... */ block
    if raw.lstrip().startswith("/*"):
        end = raw.find("*/")
        if end != -1:
            raw = raw[end + 2:]
    return raw


# Streamlit-specific overrides that layer on top of the design-system CSS.
# These re-skin Streamlit's built-in widgets (button, text_area, block container)
# to match the look defined in lyricmood.css.
STREAMLIT_OVERRIDES = """
/* force light rendering even if the OS is in dark mode */
:root { color-scheme: light; }

/* hide Streamlit chrome */
header[data-testid="stHeader"] { display: none; }
footer { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
#MainMenu { display: none; }

/* page frame = the design system's .shell */
.stApp { background: var(--paper); }
.block-container {
  max-width: 720px !important;
  padding: 72px 32px 140px !important;
}
[data-testid="stVerticalBlock"] { gap: 0.4rem; }

body, .stApp {
  font-family: var(--sans) !important;
  color: var(--ink) !important;
  font-size: 15px;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}

/* --- textarea: match .paper + textarea.lyrics --- */
.stTextArea label { display: none; }
.stTextArea [data-baseweb="textarea"],
.stTextArea [data-baseweb="base-input"],
.stTextArea > div > div {
  background: var(--paper-2) !important;
  border: 1px solid var(--rule) !important;
  border-radius: 6px 6px 0 0 !important;
  border-bottom: 0 !important;
  padding: 0 !important;
  transition: border-color .2s ease, box-shadow .2s ease;
}
.stTextArea [data-baseweb="textarea"]:focus-within,
.stTextArea > div > div:focus-within {
  border-color: var(--ink-2) !important;
  box-shadow: 0 1px 0 var(--ink-2);
}
.stTextArea textarea {
  font-family: var(--serif) !important;
  font-weight: 300 !important;
  font-size: 18px !important;
  line-height: 1.6 !important;
  color: var(--ink) !important;
  background: var(--paper-2) !important;
  padding: 22px 24px !important;
  min-height: 220px !important;
  border: 0 !important;
  outline: 0 !important;
  caret-color: var(--ink) !important;
  resize: vertical !important;
}
.stTextArea textarea::placeholder {
  color: var(--ink-3) !important;
  font-style: italic;
  opacity: 1 !important;
}

/* --- chipbar: the .foot strip under the textarea --- */
/* st.container(key="chipbar") produces a wrapper div with class "st-key-chipbar" */
.st-key-chipbar {
  background: var(--paper-2);
  border: 1px solid var(--rule);
  border-radius: 0 0 6px 6px;
  padding: 10px 16px !important;
  margin-top: -8px;
}
.st-key-chipbar [data-testid="stHorizontalBlock"] {
  align-items: center !important;
  gap: 14px !important;
}
.st-key-chipbar .lm-try,
.st-key-chipbar .lm-count {
  font-family: var(--mono);
  font-size: 10.5px;
  letter-spacing: 0.08em;
  color: var(--ink-3);
  line-height: 1.6;
}
.st-key-chipbar .lm-try { opacity: 0.7; }
.st-key-chipbar .lm-count {
  text-align: right;
  font-variant-numeric: tabular-nums;
}

/* --- buttons: primary pill + all non-primary as muted mono links --- */
.stButton > button[kind="primary"] {
  background: var(--ink) !important;
  color: var(--paper) !important;
  border: 1px solid var(--ink) !important;
  border-radius: 999px !important;
  padding: 12px 52px 12px 22px !important;
  font-family: var(--sans) !important;
  font-weight: 400 !important;
  font-size: 13.5px !important;
  letter-spacing: 0.02em !important;
  box-shadow: none !important;
  position: relative;
  transition: background .15s ease, transform .12s ease;
}
.stButton > button[kind="primary"]::before {
  content: '';
  position: absolute;
  right: 22px; top: 50%;
  width: 14px; height: 1px;
  background: currentColor;
  transform: translateY(-0.5px);
}
.stButton > button[kind="primary"]::after {
  content: '';
  position: absolute;
  right: 22px; top: 50%;
  width: 6px; height: 6px;
  border-right: 1px solid currentColor;
  border-top: 1px solid currentColor;
  transform: translateY(-50%) rotate(45deg);
  margin-top: -3px;
}
.stButton > button[kind="primary"]:hover {
  background: var(--ink-2) !important;
  border-color: var(--ink-2) !important;
  color: var(--paper) !important;
}
.stButton > button[kind="primary"]:active { transform: translateY(1px); }

.stButton > button:not([kind="primary"]) {
  background: transparent !important;
  border: 0 !important;
  border-bottom: 1px dotted var(--ink-3) !important;
  border-radius: 0 !important;
  color: var(--ink-3) !important;
  font-family: var(--mono) !important;
  font-size: 10.5px !important;
  letter-spacing: 0.08em !important;
  font-weight: 400 !important;
  padding: 2px 0 !important;
  min-height: 0 !important;
  box-shadow: none !important;
  transition: color .15s ease, border-color .15s ease;
  text-transform: lowercase;
}
.stButton > button:not([kind="primary"]):hover {
  color: var(--ink) !important;
  border-color: var(--ink) !important;
  background: transparent !important;
}

/* CLEAR button = the mock's .ghost style */
.st-key-clear_wrap .stButton > button {
  border-bottom: 0 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.14em !important;
}

/* action row spacing + vertical centering */
.st-key-action_row { margin-top: 18px; }
.st-key-action_row [data-testid="stHorizontalBlock"] { align-items: center !important; }
.st-key-action_row [data-testid="column"] { display: flex; align-items: center; }

/* keyboard shortcut hint */
.lm-shortcut {
  font-family: var(--mono);
  font-size: 10.5px;
  letter-spacing: 0.1em;
  color: var(--ink-3);
  text-align: right;
  line-height: 1;
}
.lm-shortcut kbd {
  border: 1px solid var(--rule);
  border-radius: 4px;
  padding: 2px 6px;
  background: var(--paper-2);
  font-family: inherit; font-size: 10px;
  color: var(--ink-2);
  margin: 0 1px;
}

/* fall back to Noto fonts for scripts Fraunces can't render */
.row .t, .mood-word, .shap-row .w, .brand, .brand em {
  font-family: 'Fraunces', 'Noto Serif Hebrew', 'Noto Naskh Arabic', 'Times New Roman', serif;
}
"""


def inject_design_system() -> None:
    """Call once near the top of the app. Loads fonts + base CSS + Streamlit overrides."""
    st.markdown(FONTS_LINK, unsafe_allow_html=True)
    st.markdown(f"<style>{_load_css()}</style>", unsafe_allow_html=True)
    st.markdown(f"<style>{STREAMLIT_OVERRIDES}</style>", unsafe_allow_html=True)


def set_mood_accent(mood: str) -> None:
    """Swap the active accent after a prediction. mood: Hype|Romantic|Calm|Sad|Angry."""
    var = f"--{mood.lower()}"
    st.markdown(
        f"""<style>
        :root {{
          --accent: var({var});
          --accent-soft: color-mix(in oklab, var({var}) 14%, var(--paper));
        }}
        </style>""",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# page config (must come before any Streamlit command that writes output)
# ------------------------------------------------------------

st.set_page_config(page_title="LyricMood", page_icon="🎵", layout="centered")
inject_design_system()


# ------------------------------------------------------------
# constants
# ------------------------------------------------------------

MOOD_ORDER = ["Hype", "Romantic", "Calm", "Sad", "Angry"]
MOOD_BLURB = {
    "Hype":     "imperatives · crowd nouns · short declaratives",
    "Romantic": "tender pronouns · soft verbs · warm nouns",
    "Calm":     "stillness · domestic scenes · long vowels",
    "Sad":      "absence · weather · negation",
    "Angry":    "direct address · charged verbs · second person",
}

# original short samples from LyricMood Minimal.html — safe to reuse
SAMPLES = {
    "hype": (
        "Stadium lights, I'm the main event\n"
        "Every seat up, every phone bent\n"
        "Bass is kicking, feet off the floor\n"
        "Give me loud and give me more"
    ),
    "romantic": (
        "Your hand in mine by the kitchen door\n"
        "Soft radio song we've heard before\n"
        "I could stay here for a lifetime more\n"
        "Just your breathing and the hallway floor"
    ),
    "calm": (
        "Light moves slow across the rug\n"
        "Tea has gone cool in the cup\n"
        "There is nowhere I have to be\n"
        "Only the window and the tree"
    ),
    "sad": (
        "The rain again on the empty street\n"
        "I counted every car I did not meet\n"
        "Your coat is still on the kitchen chair\n"
        "And I'm still talking to the empty air"
    ),
    "angry": (
        "Say it again, say it to my face\n"
        "Tell me how I'm the one out of place\n"
        "I built this house and you burned the door\n"
        "Don't tell me I'm asking for more"
    ),
}


# ------------------------------------------------------------
# API client — the only backend dependency is HTTP
# ------------------------------------------------------------

API_URL = os.environ.get("LYRICMOOD_API_URL", "http://localhost:8000")


@st.cache_resource
def api_client() -> httpx.Client:
    return httpx.Client(base_url=API_URL, timeout=30.0)


def _api_error_message(response: httpx.Response) -> str:
    try:
        return response.json()["error"]["message"]
    except Exception:
        return f"API error (HTTP {response.status_code})"


# ------------------------------------------------------------
# session state + callbacks
# ------------------------------------------------------------

if "lyrics" not in st.session_state:
    st.session_state["lyrics"] = ""
if "result" not in st.session_state:
    st.session_state["result"] = None


def set_sample(sample_key: str) -> None:
    st.session_state["lyrics"] = SAMPLES[sample_key]
    st.session_state["result"] = None


def clear_all() -> None:
    st.session_state["lyrics"] = ""
    st.session_state["result"] = None


# ------------------------------------------------------------
# UI — brand row + prompt
# ------------------------------------------------------------

# accent starts as --romantic (CSS default). Swap it if there's already a prediction.
if st.session_state["result"]:
    set_mood_accent(st.session_state["result"]["pred"])

st.markdown(
    """
    <div class="head">
      <div class="brand">
        <span class="mark"></span>
        <span>Lyric<em>Mood</em></span>
      </div>
      <div class="meta">a quiet reader for feeling</div>
    </div>
    <div class="prompt">
      Paste a verse.<br>
      I'll tell you what it feels like.
    </div>
    """,
    unsafe_allow_html=True,
)

# textarea
st.text_area(
    "lyrics",
    key="lyrics",
    height=240,
    placeholder="Drop a chorus, a verse, or a whole song here…",
    label_visibility="collapsed",
)

# chipbar: [try] [hype] [romantic] [calm] [sad] [angry] ........ [NN words]
word_count = len((st.session_state["lyrics"] or "").split())

with st.container(key="chipbar"):
    cols = st.columns([0.06, 0.1, 0.15, 0.1, 0.08, 0.1, 0.41])
    with cols[0]:
        st.markdown('<span class="lm-try">try</span>', unsafe_allow_html=True)
    for col, key in zip(cols[1:6], ["hype", "romantic", "calm", "sad", "angry"]):
        with col:
            st.button(key, key=f"chip_{key}", on_click=set_sample, args=(key,))
    with cols[6]:
        st.markdown(
            f'<span class="lm-count">{word_count} word{"" if word_count == 1 else "s"}</span>',
            unsafe_allow_html=True,
        )

# action row: Read-the-mood primary + CLEAR ghost + ⌘↵ hint
with st.container(key="action_row"):
    action_cols = st.columns([0.32, 0.18, 0.5])
    with action_cols[0]:
        go_clicked = st.button("Read the mood", type="primary", use_container_width=True)
    with action_cols[1]:
        with st.container(key="clear_wrap"):
            st.button("clear", on_click=clear_all, key="clear_btn")
    with action_cols[2]:
        st.markdown(
            '<div class="lm-shortcut"><kbd>⌘</kbd> <kbd>↵</kbd></div>',
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------
# run the pipeline on click
# ------------------------------------------------------------

if go_clicked:
    text = (st.session_state["lyrics"] or "").strip()
    if not text:
        st.warning("paste some lyrics first 🙂")
        st.stop()

    with st.spinner("reading the room…"):
        client = api_client()
        try:
            pred_resp = client.post("/v1/predict", json={"lyrics": text})
        except httpx.HTTPError:
            st.error(f"can't reach the LyricMood API at {API_URL} — is it running? (docker compose up)")
            st.stop()
        if pred_resp.status_code != 200:
            st.error(_api_error_message(pred_resp))
            st.stop()
        pred = pred_resp.json()

        top10 = [(e["token"], e["weight"]) for e in (pred["explanation"] or [])]

        recs = []
        sim_resp = client.post(
            "/v1/similar", json={"lyrics": text, "mood": pred["mood"], "limit": 5}
        )
        if sim_resp.status_code == 200:
            recs = [
                {"title": r["title"], "artist": r["artist"], "similarity": r["score"]}
                for r in sim_resp.json()["results"]
            ]

        st.session_state["result"] = {
            "pred": pred["mood"],
            "confidence": pred["confidence"],
            "prob_map": pred["probabilities"],
            "top10": top10,
            "recs": recs,
        }


# ------------------------------------------------------------
# reading block — only if we have a result
# ------------------------------------------------------------

result = st.session_state["result"]
if result:
    pred = result["pred"]
    conf = result["confidence"]
    prob_map = result["prob_map"]
    top10 = result["top10"]
    recs = result["recs"]

    set_mood_accent(pred)  # re-assert the accent now that we have a prediction

    # reading header + mood headline + confidence
    st.markdown(
        f"""
        <section class="reading on">
          <div class="reading-head">
            <span class="lab">reading</span>
            <span class="lab">{pred.lower()} · {conf*100:.0f}%</span>
          </div>

          <div class="mood">
            <div class="mood-word">{pred}<span class="mark"></span></div>
            <div class="conf">
              <span class="num">{conf*100:.0f}%</span>
              confidence
              <div class="bar"><span style="width: {conf*100:.1f}%;"></span></div>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    # stacked probability bar + legend
    stack_segs = []
    legend_items = []
    for m in MOOD_ORDER:
        p = prob_map.get(m, 0.0)
        opacity = 1.0 if m == pred else 0.35
        stack_segs.append(
            f'<span style="width: {p*100:.2f}%; background: var(--{m.lower()}); opacity: {opacity};"></span>'
        )
        li_class = "li pred" if m == pred else "li"
        sw_opacity = 1.0 if m == pred else 0.45
        legend_items.append(
            f"""<div class="{li_class}">
              <div class="nm"><span class="sw" style="background: var(--{m.lower()}); opacity: {sw_opacity};"></span>{m}</div>
              <div class="vl">{p*100:.0f}%</div>
            </div>"""
        )

    st.markdown(
        f"""
        <div class="probs">
          <div class="lab">
            <span>probability across moods</span>
            <span>{MOOD_BLURB[pred]}</span>
          </div>
          <div class="stack">{''.join(stack_segs)}</div>
          <div class="legend">{''.join(legend_items)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # SHAP chart — proper horizontal bar chart using the .shap / .shap-row classes
    if top10:
        max_abs = max(abs(v) for _, v in top10) or 0.01
        scale_max = float(math.ceil(max_abs * 10) / 10)

        shap_rows = []
        for word, v in top10:
            sign_class = "pos" if v >= 0 else "neg"
            pct_half = abs(v) / scale_max * 50  # % of full track width
            left = 50 if v >= 0 else 50 - pct_half
            sign = "+" if v >= 0 else "−"
            shap_rows.append(
                f"""<div class="shap-row {sign_class}">
                  <div class="w">{word}</div>
                  <div class="shap-track"><div class="shap-bar" style="left: {left:.2f}%; width: {pct_half:.2f}%;"></div></div>
                  <div class="v">{sign}{abs(v):.2f}</div>
                </div>"""
            )

        ticks = [-scale_max, -scale_max / 2, 0, scale_max / 2, scale_max]
        tick_html = "".join(
            f'<span class="tick">{("0" if t == 0 else ("+" if t > 0 else "−") + f"{abs(t):.2f}")}</span>'
            for t in ticks
        )

        st.markdown(
            f"""
            <div class="why">
              <div class="section-head">
                <h3>why <em>{pred.lower()}</em></h3>
                <div class="lab">top 10 words · SHAP</div>
              </div>
              <div class="shap">
                <div class="shap-axis-top">
                  <span>pushes away</span>
                  <span class="center">0</span>
                  <span>pushes toward</span>
                </div>
                <div class="shap-list">{''.join(shap_rows)}</div>
                <div class="shap-scale">{tick_html}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="why">
              <div class="section-head">
                <h3>why <em>{pred.lower()}</em></h3>
                <div class="lab">top 10 words · SHAP</div>
              </div>
              <div class="shap" style="padding: 22px; color: var(--ink-3); font-family: var(--mono); font-size: 11px;">
                no vocabulary overlap — SHAP has nothing to explain on this input.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # similar songs
    if recs:
        song_rows = []
        for i, r in enumerate(recs):
            song_rows.append(
                f"""<div class="row">
                  <div class="n">{i+1:02d}</div>
                  <div>
                    <div class="t">{r['title']}</div>
                    <div class="a">{r['artist']}</div>
                  </div>
                  <div class="s">{r['similarity']:.3f}</div>
                </div>"""
            )
        list_html = "".join(song_rows)
    else:
        list_html = (
            '<div class="lab" style="padding: 18px;">'
            "retrieval offline — similar songs unavailable"
            "</div>"
        )

    st.markdown(
        f"""
        <div class="similar">
          <div class="section-head">
            <h3>in a <em>similar</em> key</h3>
            <div class="lab">cosine · mood-filtered</div>
          </div>
          <div class="list">
            {list_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
