import time

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain

# Requires streamlit >= 1.39 (keyed containers are used for styling)


# ── Page icon (drawn in code, so no extra file is needed) ────────────────────
def make_icon(size: int = 192) -> Image.Image:
    """Gradient rounded square with a bold X and a small spark."""
    n = size * 2  # draw at 2x, then downsample for smooth edges
    ys, xs = np.mgrid[0:n, 0:n]
    t = (xs + ys) / (2 * (n - 1))
    stops = [(0.0, (99, 102, 241)), (0.55, (217, 70, 239)), (1.0, (249, 115, 22))]
    rgb = np.zeros((n, n, 3))
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        m = (t >= t0) & (t <= t1)
        f = ((t - t0) / (t1 - t0))[..., None]
        seg = np.array(c0) * (1 - f) + np.array(c1) * f
        rgb = np.where(m[..., None], seg, rgb)

    img = Image.fromarray(rgb.astype("uint8"), "RGB").convert("RGBA")
    mask = Image.new("L", (n, n), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, n - 1, n - 1), radius=int(n * 0.24), fill=255)
    img.putalpha(mask)

    d = ImageDraw.Draw(img)
    w = int(n * 0.14)
    a, b = int(n * 0.29), int(n * 0.71)
    r = w // 2
    strokes = (((a, a, b, b), (255, 255, 255)), ((a, b, b, a), (255, 236, 179)))
    for (x0, y0, x1, y1), col in strokes:
        d.line((x0, y0, x1, y1), fill=col, width=w)
        for x, y in ((x0, y0), (x1, y1)):
            d.ellipse((x - r, y - r, x + r, y + r), fill=col)

    cx, cy, s = int(n * 0.78), int(n * 0.22), int(n * 0.07)
    d.polygon(
        [(cx, cy - s), (cx + s // 3, cy - s // 3), (cx + s, cy), (cx + s // 3, cy + s // 3),
         (cx, cy + s), (cx - s // 3, cy + s // 3), (cx - s, cy), (cx - s // 3, cy - s // 3)],
        fill=(125, 211, 252),
    )
    return img.resize((size, size), Image.LANCZOS)


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchX · AI Research Agent",
    page_icon=make_icon(),
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design system ────────────────────────────────────────────────────────────
# Deep indigo night with an aurora of colour. Each agent owns a colour that
# follows it through the whole page:
#   Search = sky, Reader = violet, Writer = pink, Critic = amber.
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #090818;
    --surface: #14122f;
    --surface-2: #1d1a42;
    --line: rgba(255,255,255,0.12);
    --text: #f4f2ff;
    --body: #d6d3ef;
    --muted: #a09cc7;
    --sky: #38bdf8;
    --violet: #a78bfa;
    --pink: #f472b6;
    --amber: #fbbf24;
    --grad: linear-gradient(90deg, #38bdf8, #a78bfa 40%, #f472b6 75%, #fbbf24);
    --grad-btn: linear-gradient(100deg, #6366f1, #d946ef 55%, #f43f5e);
    --serif: 'Instrument Serif', Georgia, serif;
    --sans: 'Manrope', system-ui, sans-serif;
}

/* ── Base: aurora background ── */
html, body, [class*="css"], .stApp {
    font-family: var(--sans);
    color: var(--body);
}
.stApp {
    background-color: var(--bg);
    background-image:
        radial-gradient(ellipse 55% 40% at 6% -5%,  rgba(99,102,241,0.40), transparent 65%),
        radial-gradient(ellipse 45% 35% at 96% 6%,  rgba(244,114,182,0.26), transparent 65%),
        radial-gradient(ellipse 50% 40% at 72% 106%, rgba(56,189,248,0.20), transparent 65%),
        radial-gradient(ellipse 40% 30% at 0% 92%,  rgba(251,191,36,0.12), transparent 65%);
    background-attachment: fixed;
}
::selection { background: rgba(217,70,239,0.45); color: #fff; }
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2.2rem 3rem 4rem;
    max-width: 1180px;
}
@media (max-width: 640px) {
    .block-container { padding: 1.2rem 1rem 3rem; }
}
:focus-visible {
    outline: 2px solid var(--violet) !important;
    outline-offset: 2px !important;
}

/* ── Hero ── */
.brand {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-weight: 800;
    font-size: 1.15rem;
    color: var(--text);
    letter-spacing: 0.01em;
}
.brand-logo {
    width: 34px;
    height: 34px;
    border-radius: 10px;
    display: grid;
    place-items: center;
    background: linear-gradient(135deg, #6366f1, #d946ef 55%, #f97316);
    color: #fff;
    font-weight: 800;
    font-size: 1.1rem;
    box-shadow: 0 8px 24px -6px rgba(217,70,239,0.6);
}
.hero { padding: 2.4rem 0 2.2rem; }
.hero h1 {
    display: inline-block;
    font-family: var(--serif);
    font-weight: 400;
    font-size: clamp(2.8rem, 6.4vw, 5.4rem);
    line-height: 1.05;
    letter-spacing: -0.02em;
    margin: 1.5rem 0 1.1rem;
    padding: 0 0.05em 0.12em 0;
    background: linear-gradient(90deg, #7dd3fc 0%, #c4b5fd 38%, #f9a8d4 72%, #fcd34d 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
}
.hero p {
    font-size: 1.08rem;
    line-height: 1.7;
    color: var(--muted);
    max-width: 52ch;
    margin: 0;
}
.agents {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-top: 1.6rem;
}
.agent {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.42rem 0.95rem;
    border-radius: 999px;
    font-size: 0.86rem;
    font-weight: 700;
    color: var(--text);
    background: color-mix(in srgb, var(--c) 15%, transparent);
    border: 1px solid color-mix(in srgb, var(--c) 50%, transparent);
}
.agent i {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--c);
    box-shadow: 0 0 10px var(--c);
}

/* ── Cards ── */
.st-key-input_card {
    background:
        linear-gradient(var(--surface), var(--surface)) padding-box,
        linear-gradient(135deg, #38bdf8, #a78bfa 45%, #f472b6) border-box;
    border: 1.5px solid transparent;
    border-radius: 20px;
    padding: 1.8rem 1.9rem;
    box-shadow: 0 24px 70px -28px rgba(167,139,250,0.55);
}
.st-key-pipeline_card {
    background: rgba(255,255,255,0.045);
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 1.8rem 1.9rem;
    backdrop-filter: blur(12px);
}
.card-title {
    font-family: var(--serif);
    font-size: 1.65rem;
    color: var(--text);
    line-height: 1.2;
    margin: 0 0 1.3rem;
}
.hint {
    font-size: 0.84rem;
    color: var(--muted);
    margin: 1.2rem 0 0.5rem;
}

/* ── Text input ── */
.stTextInput label p {
    font-size: 0.92rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
}
.stTextInput [data-baseweb="input"] {
    background: var(--bg) !important;
    border: 1px solid var(--line) !important;
    border-radius: 12px !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.stTextInput [data-baseweb="input"]:focus-within {
    border-color: var(--violet) !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.25) !important;
}
.stTextInput input {
    background: transparent !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
    font-size: 1rem !important;
    padding: 0.85rem 1rem !important;
}
.stTextInput input::placeholder { color: #6f6a9c !important; }

/* ── Primary button ── */
.st-key-run_btn button {
    background: var(--grad-btn) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 1.4rem !important;
    min-height: 0 !important;
    box-shadow: 0 12px 30px -10px rgba(217,70,239,0.7) !important;
    transition: transform 0.15s, filter 0.15s, box-shadow 0.15s;
}
.st-key-run_btn button p {
    color: #fff !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
}
.st-key-run_btn button:hover {
    filter: brightness(1.1);
    transform: translateY(-1px);
    box-shadow: 0 16px 36px -10px rgba(217,70,239,0.85) !important;
}
.st-key-run_btn button:active { transform: translateY(0); }
.st-key-run_btn button:disabled { filter: grayscale(0.6) brightness(0.8); }

/* ── Example chips (each picks up an agent colour) ── */
.st-key-chips button {
    background: transparent !important;
    border: 1px solid var(--line) !important;
    border-radius: 999px !important;
    min-height: 0 !important;
    padding: 0.32rem 0.9rem !important;
    transition: border-color 0.15s, background 0.15s;
}
.st-key-chips button p {
    color: var(--muted) !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
}
.st-key-chip_0 button p { color: #7dd3fc !important; }
.st-key-chip_1 button p { color: #c4b5fd !important; }
.st-key-chip_2 button p { color: #f9a8d4 !important; }
.st-key-chip_0 button:hover { border-color: var(--sky) !important;    background: rgba(56,189,248,0.12) !important; }
.st-key-chip_1 button:hover { border-color: var(--violet) !important; background: rgba(167,139,250,0.12) !important; }
.st-key-chip_2 button:hover { border-color: var(--pink) !important;   background: rgba(244,114,182,0.12) !important; }

/* ── Pipeline relay (each step carries its own colour in --c) ── */
.relay { display: flex; flex-direction: column; }
.relay-step {
    position: relative;
    display: grid;
    grid-template-columns: 34px 1fr auto;
    gap: 14px;
    align-items: start;
    padding-bottom: 1.6rem;
}
.relay-step:last-child { padding-bottom: 0; }
.relay-step::after {
    content: '';
    position: absolute;
    left: 16px;
    top: 38px;
    bottom: 4px;
    width: 2px;
    background: var(--line);
    border-radius: 2px;
}
.relay-step:last-child::after { display: none; }
.relay-step.done::after { background: var(--c); opacity: 0.6; }

.relay-node {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    display: grid;
    place-items: center;
    font-weight: 800;
    font-size: 0.85rem;
    background: var(--bg);
    border: 2px solid color-mix(in srgb, var(--c) 35%, transparent);
    color: var(--muted);
    transition: border-color 0.3s, color 0.3s, background 0.3s;
}
.relay-step.running .relay-node {
    border-color: var(--c);
    color: var(--c);
    animation: pulse 1.8s ease-out infinite;
}
.relay-step.done .relay-node {
    border-color: var(--c);
    background: var(--c);
    color: #090818;
    box-shadow: 0 0 18px -2px var(--c);
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 color-mix(in srgb, var(--c) 55%, transparent); }
    80%  { box-shadow: 0 0 0 12px color-mix(in srgb, var(--c) 0%, transparent); }
    100% { box-shadow: 0 0 0 0 color-mix(in srgb, var(--c) 0%, transparent); }
}
@media (prefers-reduced-motion: reduce) {
    .relay-step.running .relay-node { animation: none; }
}

.relay-title { font-weight: 800; font-size: 0.98rem; color: var(--text); line-height: 1.3; padding-top: 5px; }
.relay-step.running .relay-title,
.relay-step.done .relay-title { color: var(--c); }
.relay-desc  { font-size: 0.84rem; color: var(--muted); line-height: 1.5; margin-top: 2px; }
.relay-meta  { font-size: 0.8rem; font-weight: 700; color: #6f6a9c; padding-top: 8px; }
.relay-step.running .relay-meta,
.relay-step.done .relay-meta { color: var(--c); }

/* ── Results ── */
.results-heading {
    font-family: var(--serif);
    font-size: 2.4rem;
    color: var(--text);
    margin: 3rem 0 0.8rem;
    line-height: 1.1;
}

.stTabs [data-baseweb="tab-list"] { gap: 0.4rem; border-bottom: 1px solid var(--line); }
.stTabs button[data-baseweb="tab"] { background: transparent; padding: 0.7rem 1rem; }
.stTabs button[data-baseweb="tab"] p { font-weight: 700; color: var(--muted); font-size: 0.95rem; }
.stTabs button[data-baseweb="tab"][aria-selected="true"] p { color: var(--text); }
.stTabs [data-baseweb="tab-highlight"] { background: var(--grad) !important; height: 3px !important; }
.stTabs [data-baseweb="tab-border"] { display: none; }

.st-key-report_card {
    background:
        linear-gradient(var(--surface), var(--surface)) padding-box,
        linear-gradient(135deg, #f472b6, #a78bfa 50%, #38bdf8) border-box;
    border: 1.5px solid transparent;
    border-radius: 20px;
    padding: 2.2rem 2.6rem;
    margin-top: 1.2rem;
    box-shadow: 0 24px 70px -30px rgba(244,114,182,0.45);
}
.st-key-critic_card {
    background: linear-gradient(180deg, rgba(251,191,36,0.08), rgba(251,191,36,0.02));
    border: 1px solid rgba(251,191,36,0.35);
    border-left: 4px solid var(--amber);
    border-radius: 20px;
    padding: 2.2rem 2.6rem;
    margin-top: 1.2rem;
}
.st-key-trail_card {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 2.2rem 2.6rem;
    margin-top: 1.2rem;
}
@media (max-width: 640px) {
    :is(.st-key-report_card, .st-key-critic_card, .st-key-trail_card) { padding: 1.4rem 1.2rem; }
}

:is(.st-key-report_card, .st-key-critic_card, .st-key-trail_card) [data-testid="stMarkdownContainer"] {
    max-width: 74ch;
}
:is(.st-key-report_card, .st-key-critic_card, .st-key-trail_card) :is(h1, h2, h3, h4) {
    font-family: var(--serif) !important;
    font-weight: 400 !important;
    letter-spacing: -0.01em;
    line-height: 1.15 !important;
}
:is(.st-key-report_card, .st-key-critic_card, .st-key-trail_card) h1 { color: var(--text) !important; font-size: 2.6rem !important; }
:is(.st-key-report_card, .st-key-critic_card, .st-key-trail_card) h2 { color: #c4b5fd !important; font-size: 1.95rem !important; margin-top: 1.6rem !important; }
:is(.st-key-report_card, .st-key-critic_card, .st-key-trail_card) h3 { color: #f9a8d4 !important; font-size: 1.5rem !important; }
:is(.st-key-report_card, .st-key-critic_card, .st-key-trail_card) :is(p, li) {
    font-size: 1rem;
    line-height: 1.8;
    color: var(--body);
}
:is(.st-key-report_card, .st-key-critic_card, .st-key-trail_card) strong { color: var(--text); }
:is(.st-key-report_card, .st-key-critic_card, .st-key-trail_card) a { color: #7dd3fc; }
:is(.st-key-report_card, .st-key-critic_card, .st-key-trail_card) blockquote {
    border-left: 3px solid var(--pink);
    padding-left: 1rem;
    color: var(--muted);
}
:is(.st-key-report_card, .st-key-critic_card, .st-key-trail_card) code {
    background: var(--surface-2);
    color: #fcd34d;
    border-radius: 6px;
    padding: 0.1rem 0.4rem;
}

/* ── Download button ── */
.stDownloadButton button {
    background:
        linear-gradient(var(--bg), var(--bg)) padding-box,
        var(--grad) border-box !important;
    border: 1.5px solid transparent !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.3rem !important;
    transition: filter 0.15s, transform 0.15s;
}
.stDownloadButton button p { color: var(--text) !important; font-weight: 700 !important; }
.stDownloadButton button:hover { filter: brightness(1.2); transform: translateY(-1px); }

/* ── Footer ── */
.footer {
    font-size: 0.82rem;
    color: #6f6a9c;
    margin-top: 4rem;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ── Pipeline definition ──────────────────────────────────────────────────────
STEPS = [
    ("search", "Search agent", "Finds recent, reliable sources on your topic", "#38bdf8"),
    ("reader", "Reader agent", "Opens the best source and pulls out the details", "#a78bfa"),
    ("writer", "Writer", "Drafts the full research report", "#f472b6"),
    ("critic", "Critic", "Reviews the draft and scores it", "#fbbf24"),
]

EXAMPLES = ["Agentic AI frameworks", "RAG techniques in 2026", "Small language models"]


# ── Session state ────────────────────────────────────────────────────────────
defaults = {
    "results": {},
    "timings": {},
    "running": False,
    "done": False,
    "error": None,
    "topic_val": "",
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ── Helpers ──────────────────────────────────────────────────────────────────
def use_example(text: str):
    st.session_state.topic_input = text


def pipeline_html() -> str:
    """Render the four pipeline steps as a connected vertical relay."""
    results = st.session_state.results
    timings = st.session_state.timings
    running = st.session_state.running
    first_missing = next((s[0] for s in STEPS if s[0] not in results), None)

    rows = []
    for i, (key, title, desc, color) in enumerate(STEPS, start=1):
        if key in results:
            state, node = "done", "✓"
            meta = f"{timings.get(key, 0):.1f}s"
        elif running and key == first_missing:
            state, node, meta = "running", str(i), "Working…"
        else:
            state, node, meta = "waiting", str(i), "Waiting"
        rows.append(
            f'<div class="relay-step {state}" style="--c:{color}">'
            f'<div class="relay-node">{node}</div>'
            f'<div><div class="relay-title">{title}</div>'
            f'<div class="relay-desc">{desc}</div></div>'
            f'<div class="relay-meta">{meta}</div>'
            f"</div>"
        )
    return '<div class="relay">' + "".join(rows) + "</div>"


# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
<div class="brand"><span class="brand-logo">X</span>ResearchX</div>
<h1>Four agents.<br>One researched answer.</h1>
<p>Describe a topic. A search agent, a reader, a writer and a critic work through it
in turn and hand you a finished report.</p>
<div class="agents">
<span class="agent" style="--c:#38bdf8"><i></i>Search</span>
<span class="agent" style="--c:#a78bfa"><i></i>Read</span>
<span class="agent" style="--c:#f472b6"><i></i>Write</span>
<span class="agent" style="--c:#fbbf24"><i></i>Critique</span>
</div>
</div>
""",
    unsafe_allow_html=True,
)


# ── Layout: input left, pipeline right ───────────────────────────────────────
col_input, col_pipeline = st.columns([6, 4], gap="large")

with col_input:
    with st.container(key="input_card"):
        st.markdown('<div class="card-title">What should we research?</div>', unsafe_allow_html=True)
        topic = st.text_input(
            "Research topic",
            placeholder="e.g. Quantum computing breakthroughs in 2025",
            key="topic_input",
        )
        run_btn = st.button(
            "Run research pipeline",
            use_container_width=True,
            key="run_btn",
            disabled=st.session_state.running,
        )

        st.markdown('<div class="hint">Or start from an example</div>', unsafe_allow_html=True)
        with st.container(key="chips"):
            chip_cols = st.columns(len(EXAMPLES))
            for i, (col, ex) in enumerate(zip(chip_cols, EXAMPLES)):
                with col:
                    st.button(
                        ex,
                        key=f"chip_{i}",
                        on_click=use_example,
                        args=(ex,),
                        use_container_width=True,
                        disabled=st.session_state.running,
                    )

    if st.session_state.error:
        st.error(f"The pipeline stopped: {st.session_state.error}")

with col_pipeline:
    with st.container(key="pipeline_card"):
        st.markdown('<div class="card-title">Pipeline</div>', unsafe_allow_html=True)
        pipeline_slot = st.empty()
        pipeline_slot.markdown(pipeline_html(), unsafe_allow_html=True)


# ── Start a run ──────────────────────────────────────────────────────────────
if run_btn:
    if not topic.strip():
        st.warning("Enter a research topic to get started.")
    else:
        st.session_state.results = {}
        st.session_state.timings = {}
        st.session_state.error = None
        st.session_state.topic_val = topic.strip()
        st.session_state.running = True
        st.session_state.done = False
        st.rerun()


# ── Run pipeline (updates the cards live after every step) ───────────────────
def commit(results: dict, key: str, started: float):
    st.session_state.results = dict(results)
    st.session_state.timings[key] = time.time() - started
    pipeline_slot.markdown(pipeline_html(), unsafe_allow_html=True)


if st.session_state.running and not st.session_state.done:
    results = {}
    topic_val = st.session_state.topic_val

    try:
        # Step 1: Search
        t = time.time()
        search_agent = build_search_agent()
        sr = search_agent.invoke({
            "messages": [("user", f"Find recent, reliable and detailed information about: {topic_val}")]
        })
        results["search"] = sr["messages"][-1].content
        commit(results, "search", t)

        # Step 2: Reader
        t = time.time()
        reader_agent = build_reader_agent()
        rr = reader_agent.invoke({
            "messages": [("user",
                f"Based on the following search results about '{topic_val}', "
                f"pick the most relevant URL and scrape it for deeper content.\n\n"
                f"Search Results:\n{results['search'][:800]}"
            )]
        })
        results["reader"] = rr["messages"][-1].content
        commit(results, "reader", t)

        # Step 3: Writer
        t = time.time()
        research_combined = (
            f"SEARCH RESULTS:\n{results['search']}\n\n"
            f"DETAILED SCRAPED CONTENT:\n{results['reader']}"
        )
        results["writer"] = writer_chain.invoke({
            "topic": topic_val,
            "research": research_combined,
        })
        commit(results, "writer", t)

        # Step 4: Critic
        t = time.time()
        results["critic"] = critic_chain.invoke({"report": results["writer"]})
        commit(results, "critic", t)

        st.session_state.done = True
    except Exception as exc:  # surface the failure instead of leaving a spinner
        st.session_state.error = str(exc)
    finally:
        st.session_state.running = False

    st.rerun()


# ── Results ──────────────────────────────────────────────────────────────────
r = st.session_state.results

if r:
    st.markdown('<div class="results-heading">Results</div>', unsafe_allow_html=True)

    tab_names = []
    if "writer" in r:
        tab_names.append("Report")
    if "critic" in r:
        tab_names.append("Critic review")
    tab_names.append("Research trail")
    tabs = dict(zip(tab_names, st.tabs(tab_names)))

    if "Report" in tabs:
        with tabs["Report"]:
            with st.container(key="report_card"):
                st.markdown(r["writer"])
            st.download_button(
                label="Download report (.md)",
                data=r["writer"],
                file_name=f"research_report_{int(time.time())}.md",
                mime="text/markdown",
            )

    if "Critic review" in tabs:
        with tabs["Critic review"]:
            with st.container(key="critic_card"):
                st.markdown(r["critic"])

    with tabs["Research trail"]:
        with st.container(key="trail_card"):
            st.markdown("### Search agent output")
            st.markdown(r["search"])
            if "reader" in r:
                st.markdown("### Reader agent output")
                st.markdown(r["reader"])


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">ResearchX is a LangChain multi-agent pipeline built with Streamlit.</div>',
    unsafe_allow_html=True,
)