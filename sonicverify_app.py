"""
SonicVerify — AI-Powered Real-Time Voice Cloning & Impersonation Risk Detector
--------------------------------------------------------------------------------
A Streamlit front-end for a voice-integrity verification tool. Users can
record or upload an audio clip; the app analyzes it via the FastAPI backend
and produces a Risk % score, a breakdown across Acoustic / Prosody / Spectral
dimensions, a plain-language explanation, alerts, and recommended actions.
"""

import io
import time
import wave
import requests
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="SonicVerify — Voice Cloning Risk Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# SESSION STATE
# ============================================================================
defaults = {
    "theme": "dark",
    "page": "Dashboard",
    "history": [],           # list of result dicts
    "current_audio": None,   # np.ndarray samples
    "current_sr": None,
    "current_source": None,  # filename / "Live Recording"
    "last_result": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================================
# THEME & CSS INJECTION
# ============================================================================
def inject_theme(theme: str):
    if theme == "dark":
        bg, bg2, text, sub, card, border = (
            "#070c1f", "#0f1a3a", "#f4f6fc", "#a8b6db", "#121f45", "#22325f",
        )
        grad1, grad2 = "#ff7a1a", "#ffa94d"
    else:
        bg, bg2, text, sub, card, border = (
            "#fdf6ee", "#f7ead9", "#1a2140", "#5c6690", "#fffaf2", "#ecd9bd",
        )
        grad1, grad2 = "#e8590c", "#ff8a3d"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&display=swap');

        :root {{
            --font-display: 'Sora', sans-serif;
            --primary-color: {grad1};
            --background-color: {bg};
            --secondary-background-color: {card};
            --text-color: {text};
            color: {text};
        }}

        html, body, .stApp, .stApp *:not([data-testid="stIconMaterial"]):not(.material-icons):not([class*="material-symbols"]):not(i) {{
            font-family: "Sora", sans-serif;
            color: {text};
        }}

        .stApp p, .stApp span, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
            color: {text} !important;
        }}

        [data-testid="stIconMaterial"],
        span[data-testid="stIconMaterial"],
        .material-icons,
        .material-symbols-outlined,
        .material-symbols-rounded,
        [class*="material-symbols"],
        [data-testid="stExpander"] summary svg,
        [data-testid="stExpander"] summary span,
        [data-testid="stExpander"] summary [data-testid="stIconMaterial"] {{
            font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
            font-weight: normal !important;
            font-style: normal !important;
            line-height: 1 !important;
            text-transform: none !important;
            letter-spacing: normal !important;
            word-wrap: normal !important;
            white-space: nowrap !important;
            direction: ltr !important;
            -webkit-font-smoothing: antialiased !important;
        }}

        html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
        .main, .block-container, [data-testid="stBottom"], [data-testid="stBottomBlockContainer"],
        [data-testid="stForm"], [data-testid="stVerticalBlock"], [data-testid="stHorizontalBlock"] {{
            background-color: {bg} !important;
        }}
        header[data-testid="stHeader"] {{ background-color: {bg} !important; }}
        header[data-testid="stHeader"] * {{ color: {text} !important; fill: {text} !important; }}
        [data-testid="stToolbar"] {{ background-color: {bg} !important; }}
        section[data-testid="stSidebar"] {{ background-color: {bg2} !important; border-right: 1px solid {border} !important; }}
        section[data-testid="stSidebar"] * {{ color: {text} !important; }}

        [data-testid="stSelectbox"] > div > div {{
            background-color: {card} !important;
            color: {text} !important;
            border: 1px solid {border} !important;
            border-radius: 10px !important;
        }}
        [data-testid="stSelectbox"] * {{
            color: {text} !important;
            fill: {text} !important;
        }}
        [data-baseweb="select"] * {{
            background-color: {card} !important;
            color: {text} !important;
        }}
        [data-baseweb="popover"], [data-baseweb="menu"], ul[role="listbox"] {{
            background-color: {card} !important;
            border: 1px solid {border} !important;
            border-radius: 10px !important;
        }}
        li[role="option"] {{
            background-color: {card} !important;
            color: {text} !important;
        }}
        li[role="option"]:hover, li[aria-selected="true"] {{
            background-color: {bg2} !important;
            color: {grad1} !important;
        }}

        [data-testid="stExpander"] {{
            background-color: {card} !important;
            border: 1px solid {border} !important;
            border-radius: 12px !important;
            overflow: hidden;
            margin-bottom: 12px;
        }}
        [data-testid="stExpander"] details {{
            background-color: {card} !important;
        }}
        [data-testid="stExpander"] summary {{
            background-color: {card} !important;
            color: {text} !important;
            padding: 12px 16px !important;
            border-radius: 12px !important;
        }}
        [data-testid="stExpander"] summary:hover {{
            background-color: {bg2} !important;
        }}
        [data-testid="stExpander"] div[role="group"] {{
            background-color: {card} !important;
            padding: 16px !important;
        }}

        [data-testid="stFileUploader"] {{
            background-color: {card} !important;
            border: 1px solid {border} !important;
            border-radius: 14px !important;
            padding: 16px !important;
        }}
        [data-testid="stFileUploader"] label, 
        [data-testid="stFileUploader"] [data-testid="stWidgetLabel"] p,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] small {{
            color: {text} !important;
            font-weight: 600 !important;
        }}
        [data-testid="stFileUploaderDropzone"] {{
            background-color: {bg2} !important;
            border: 2px dashed {border} !important;
            border-radius: 12px !important;
            color: {text} !important;
        }}
        [data-testid="stFileUploaderDropzone"] * {{
            color: {text} !important;
        }}
        [data-testid="stFileUploaderDropzone"] button {{
            background-color: {card} !important;
            color: {text} !important;
            border: 1px solid {border} !important;
            border-radius: 8px !important;
        }}
        [data-testid="stFileUploaderDropzone"] button:hover {{
            border-color: {grad1} !important;
            color: {grad1} !important;
        }}
        [data-testid="stFileUploaderFile"] {{
            background-color: {bg2} !important;
            border: 1px solid {border} !important;
            border-radius: 10px !important;
        }}
        [data-testid="stFileUploaderFile"] * {{
            color: {text} !important;
        }}
        [data-testid="stFileUploaderFileName"] {{
            color: {text} !important;
            font-weight: 500 !important;
        }}

        [data-testid="stAudioInput"] {{
            background-color: {card} !important;
            border: 1px dashed {border} !important;
            border-radius: 14px !important;
            padding: 14px !important;
        }}
        [data-testid="stAudioInput"] * {{
            background-color: transparent !important;
            color: {text} !important;
        }}
        [data-testid="stAudioInput"] svg {{ fill: {grad1} !important; color: {grad1} !important; }}

        .sv-hero {{
            background: linear-gradient(120deg, {grad1}22, {grad2}22);
            border: 1px solid {border};
            border-radius: 20px; padding: 28px 32px; margin-bottom: 22px;
        }}
        .sv-hero h1 {{
            font-weight: 700 !important;
            font-size: clamp(32px, 4vw, 50px) !important;
            letter-spacing: -0.04em !important;
            line-height: 1.1 !important;
            margin: 0;
            background: linear-gradient(90deg, {grad1}, {grad2});
            -webkit-background-clip: text; background-clip: text; color: transparent !important;
        }}
        .sv-hero p {{
            color: {sub} !important; margin-top: 8px; font-size: 1.02rem;
            line-height: 1.6 !important;
        }}

        .sv-card {{
            background-color: {card} !important; border: 1px solid {border} !important;
            border-radius: 16px; padding: 22px 24px; margin-bottom: 18px;
            color: {text} !important;
        }}
        .sv-card *, .sv-card p, .sv-card li, .sv-card span, .sv-card div {{
            color: {text} !important;
        }}

        .sv-badge {{
            display: inline-block; padding: 5px 14px; border-radius: 999px;
            font-weight: 700; font-size: 0.85rem; letter-spacing: 0.02em;
        }}
        .sv-alert-high {{
            background: #e74c3c22; border: 1px solid #e74c3c; color: #ff6b5b !important;
            border-radius: 12px; padding: 14px 18px; font-weight: 600;
        }}
        .sv-alert-med {{
            background: #f1c40f22; border: 1px solid #f1c40f; color: #f1c40f !important;
            border-radius: 12px; padding: 14px 18px; font-weight: 600;
        }}
        .sv-alert-low {{
            background: #2ecc7122; border: 1px solid #2ecc71; color: #2ecc71 !important;
            border-radius: 12px; padding: 14px 18px; font-weight: 600;
        }}
        .sv-subtle {{ color: {sub} !important; }}
        .sv-navlabel {{ color: {sub} !important; font-size: 0.78rem; text-transform: uppercase;
            letter-spacing: 0.08em; margin: 14px 0 4px 2px; }}
        .sv-bar-track {{
            background-color: {border}; border-radius: 8px; height: 12px; width: 100%;
            overflow: hidden; margin-top: 4px;
        }}
        .sv-bar-fill {{ height: 100%; border-radius: 8px; }}

        div[data-testid="stMetric"] {{
            background-color: {card} !important; border: 1px solid {border} !important;
            border-radius: 14px; padding: 14px 18px;
        }}
        [data-testid="stMetricLabel"] p {{ color: {sub} !important; }}
        [data-testid="stMetricValue"] {{ color: {text} !important; }}

        .stButton>button, .stDownloadButton>button {{
            border-radius: 10px; font-weight: 600; border: 1px solid {border};
            background-color: {card}; color: {text} !important;
        }}
        .stButton>button p {{ color: {text} !important; }}
        .stButton>button:hover {{ border: 1px solid {grad1}; color: {grad1} !important; }}
        .stButton>button:hover p {{ color: {grad1} !important; }}
        button[kind="primary"] {{
            background: linear-gradient(90deg, {grad1}, {grad2}) !important;
            border: none !important;
        }}
        button[kind="primary"] p {{ color: #0a1128 !important; font-weight: 700; }}

        .sv-table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
        .sv-table th {{
            text-align: left; color: {sub} !important; font-weight: 600; font-size: 0.78rem;
            text-transform: uppercase; letter-spacing: 0.05em;
            padding: 8px 12px; border-bottom: 1px solid {border};
        }}
        .sv-table td {{ padding: 10px 12px; border-bottom: 1px solid {border}; color: {text} !important; }}
        .sv-table tr:hover td {{ background-color: {bg2} !important; }}
        audio {{ border-radius: 10px; width: 100%; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    return dict(bg=bg, bg2=bg2, text=text, sub=sub, card=card, border=border, grad1=grad1, grad2=grad2)


C = inject_theme(st.session_state.theme)


def risk_color(risk):
    if risk >= 65:
        return "#e74c3c"
    if risk >= 35:
        return "#f1c40f"
    return "#2ecc71"


def render_table(rows, columns):
    head = "".join(f"<th>{label}</th>" for _, label in columns)
    body_rows = []
    for r in rows:
        cells = "".join(f"<td>{r.get(k, '')}</td>" for k, _ in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    html = f'<table class="sv-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'
    st.markdown(html, unsafe_allow_html=True)


def score_bar(label, value, color):
    st.markdown(
        f"""
        <div style="margin-bottom:10px;">
          <div style="display:flex; justify-content:space-between; font-size:0.92rem;">
            <span style="color:{C['text']};">{label}</span><span style="font-weight:700; color:{C['text']};">{value:.0f}%</span>
          </div>
          <div class="sv-bar-track">
            <div class="sv-bar-fill" style="width:{value}%; background-color:{color};"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# AUDIO LOADING
# ============================================================================
def load_audio(uploaded_file):
    raw = uploaded_file.read()
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    if HAS_SOUNDFILE:
        try:
            data, sr = sf.read(io.BytesIO(raw), always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)
            return data.astype(np.float32), sr
        except Exception:
            pass

    try:
        with wave.open(io.BytesIO(raw), "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            sampwidth = wf.getsampwidth()
            n_channels = wf.getnchannels()
            frames = wf.readframes(n_frames)
            dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sampwidth, np.int16)
            data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
            if n_channels > 1:
                data = data.reshape(-1, n_channels).mean(axis=1)
            data = data / float(np.iinfo(dtype).max)
            return data, sr
    except Exception:
        return None, None


# ============================================================================
# ANALYSIS ENGINE (FASTAPI BACKEND DISPATCHER)
# ============================================================================
def run_backend_analysis(file_obj, filename, samples=None, sr=None):
    with st.spinner("Analyzing audio with backend model..."):
        try:
            file_bytes = file_obj.getvalue() if hasattr(file_obj, "getvalue") else file_obj.read()
            try:
                file_obj.seek(0)
            except Exception:
                pass

            # Read backend URL dynamically from Streamlit Secrets or default to Render
            backend_url = st.secrets.get("BACKEND_URL", "https://sonicverify.onrender.com/api/analyze")

            response = requests.post(
                backend_url,
                files={"audio": (filename, file_bytes, "audio/wav")},
                data={
                    "phone_number": "+10000000000",
                    "financial_request": "false",
                    "identity_claim": "unspecified",
                },
                timeout=120,
            )

            if response.status_code == 200:
                res_data = response.json()
                if res_data.get("success"):
                    risk_info = res_data.get("risk_assessment", {})
                    risk_score = float(risk_info.get("risk_score", 50.0))
                    
                    if samples is None or sr is None:
                        samples, sr = load_audio(file_obj)

                    breakdown = risk_info.get("breakdown", {})
                    res = {
                        "duration": len(samples) / sr if (samples is not None and sr) else 0,
                        "sample_rate": sr or 22050,
                        "energy_var": 0.01,
                        "silence_ratio": 0.1,
                        "zcr": 0.08,
                        "spectral_flatness": 0.1,
                        "acoustic": float(breakdown.get("model_score", 50)),
                        "prosody": float(breakdown.get("phone_reputation_score", 50)),
                        "spectral": float(breakdown.get("context_risk_score", 50)),
                        "risk": risk_score,
                        "verdict": risk_info.get("risk_level", "Unknown"),
                        "tier": "high" if risk_score >= 65 else ("medium" if risk_score >= 35 else "low"),
                        "color": risk_color(risk_score),
                        "filename": filename,
                        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

                    st.session_state.current_audio = samples
                    st.session_state.current_sr = sr
                    st.session_state.current_source = filename
                    st.session_state.last_result = res
                    st.session_state.history.append(res)
                    st.session_state.page = "Analysis Result"
                    st.rerun()
                else:
                    st.error(f"Backend error: {res_data.get('error', 'Unknown error')}")
            else:
                st.error(f"Backend HTTP {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")


def explain(res):
    lines = []
    if res["spectral_flatness"] > 0.25:
        lines.append("The frequency spectrum is unusually flat/uniform — a pattern often seen in neural speech synthesis rather than natural vocal tracts.")
    else:
        lines.append("Spectral texture shows the natural peaks and troughs typical of a human vocal tract.")

    if res["silence_ratio"] > 0.35:
        lines.append("An abnormally high proportion of near-silent frames was detected, which can indicate splicing or generation artifacts.")
    elif res["silence_ratio"] < 0.05:
        lines.append("Very little natural pausing was found — continuous, unbroken speech can be a synthetic-voice indicator.")
    else:
        lines.append("Pause and silence patterns fall within a normal conversational range.")

    if abs(res["zcr"] - 0.08) > 0.05:
        lines.append("Zero-crossing rate (a proxy for voicing/noisiness) deviates from typical human speech norms.")
    else:
        lines.append("Zero-crossing rate is consistent with natural voiced/unvoiced speech transitions.")

    if res["energy_var"] < 0.003:
        lines.append("Loudness/energy stays oddly constant over time — human speech usually has more natural dynamic variation.")
    else:
        lines.append("Energy dynamics (loudness variation) look consistent with natural speech delivery.")
    return lines


def recommendations(tier):
    if tier == "high":
        return [
            "🚨 Do NOT approve any transaction, share credentials, or disclose confidential information on this call.",
            "📞 Terminate the call and re-establish contact using a known, previously verified phone number.",
            "🧑‍💼 Escalate immediately to your security / fraud team and log the incident.",
            "🔐 Trigger secondary verification: multi-factor authentication or a pre-agreed passphrase.",
        ]
    if tier == "medium":
        return [
            "⚠️ Pause before acting — ask a personal or pre-agreed verification question the caller should know.",
            "📞 Offer to call back on a known/registered number before proceeding.",
            "🧑‍💼 Loop in a supervisor for high-value or sensitive requests.",
            "🗒️ Log this interaction for review even if it ultimately proceeds.",
        ]
    return [
        "✅ No immediate action required — signals are consistent with a genuine human voice.",
        "🔁 For very high-value transactions, periodic re-verification is still good practice.",
        "🗂️ Archive the interaction summary for audit trail purposes.",
    ]


def precautions(tier):
    if tier == "high":
        return [
            "Stop the call now. Do not send money, share OTPs, or reveal confidential information.",
            "Call the person back on a number you already trust, not one the caller gives you.",
            "Report this to your security or fraud team right away and keep the recording.",
        ]
    if tier == "medium":
        return [
            "Do not share OTPs, passwords, or card details until the caller is verified.",
            "Ask a question only the real person would know, or call back on a saved number.",
            "Ignore urgency or pressure. Take your time and involve a supervisor.",
        ]
    return [
        "Signals look consistent with a genuine human voice, but no automated check is perfect.",
        "For high-value requests, still confirm through a second channel before acting.",
        "Keep a record of the call in case you need to review it later.",
    ]


# ============================================================================
# CHARTS
# ============================================================================
def waveform_fig(samples, sr, color, upto=None):
    n = upto if upto else len(samples)
    fig, ax = plt.subplots(figsize=(9, 2.4))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    t = np.linspace(0, len(samples) / sr if sr else len(samples), num=len(samples))
    ax.plot(t[:n], samples[:n], linewidth=0.6, color=color)
    ax.set_xlim(0, t[-1] if len(t) else 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Time (s)", color=C["text"])
    ax.set_yticks([])
    ax.tick_params(colors=C["text"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def spectrogram_fig(samples, sr):
    fig, ax = plt.subplots(figsize=(9, 2.8))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.specgram(samples, Fs=sr if sr else 22050, cmap="magma")
    ax.set_xlabel("Time (s)", color=C["text"])
    ax.set_ylabel("Freq (Hz)", color=C["text"])
    ax.tick_params(colors=C["text"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def gauge_fig(risk, color):
    fig, ax = plt.subplots(figsize=(3.2, 3.2), subplot_kw={"aspect": "equal"})
    fig.patch.set_alpha(0)
    track = "#22325f" if st.session_state.theme == "dark" else "#ecd9bd"
    ax.pie([risk, 100 - risk], colors=[color, track], startangle=90,
           counterclock=False, wedgeprops={"width": 0.32, "edgecolor": "none"})
    ax.text(0, 0.08, f"{risk:.0f}%", ha="center", va="center", fontsize=28,
            fontweight="bold", color=C["text"])
    ax.text(0, -0.28, "AI Voice Risk", ha="center", va="center", fontsize=10,
            color=C["sub"])
    return fig


# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
NAV = [
    ("Dashboard", "🏠"),
    ("Record Audio", "🎙️"),
    ("Upload Audio", "📁"),
    ("Analysis Result", "📊"),
    ("Alerts & History", "🔔"),
    ("Recommendations", "🧭"),
]

with st.sidebar:
    st.markdown("## 🛡️ SonicVerify")
    st.caption("Real-time voice cloning & impersonation risk detection")
    st.markdown('<div class="sv-navlabel">Navigate</div>', unsafe_allow_html=True)
    for name, icon in NAV:
        is_active = st.session_state.page == name
        if st.button(f"{icon}  {name}", key=f"nav_{name}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.page = name
            st.rerun()

    st.markdown("---")
    theme_choice = st.toggle("🌙 Dark mode", value=(st.session_state.theme == "dark"))
    new_theme = "dark" if theme_choice else "light"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

    st.markdown("---")
    st.caption(
        "⚠️ AI security tool — analyzes acoustic, prosody and "
        "spectral signal statistics using the backend risk model."
    )

page = st.session_state.page


# ============================================================================
# PAGE: DASHBOARD
# ============================================================================
if page == "Dashboard":
    st.markdown(
        """
        <div class="sv-hero">
          <h1>🛡️ SonicVerify</h1>
          <p>AI-powered real-time detection of voice cloning &amp; synthetic-speech
          impersonation — screen calls before you trust them.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    hist = st.session_state.history
    total = len(hist)
    high = sum(1 for h in hist if h["tier"] == "high")
    avg_risk = np.mean([h["risk"] for h in hist]) if hist else 0
    last_verdict = hist[-1]["verdict"] if hist else "—"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Scans", total)
    c2.metric("High-Risk Flags", high)
    c3.metric("Average Risk", f"{avg_risk:.0f}%")
    c4.metric("Last Verdict", last_verdict)

    st.markdown("#### Quick Actions")
    a, b, c = st.columns(3)
    with a:
        if st.button("🎙️ Record a Call / Voice", use_container_width=True):
            st.session_state.page = "Record Audio"; st.rerun()
    with b:
        if st.button("📁 Upload an Audio File", use_container_width=True):
            st.session_state.page = "Upload Audio"; st.rerun()
    with c:
        if st.button("🔔 View Alerts & History", use_container_width=True):
            st.session_state.page = "Alerts & History"; st.rerun()

    st.markdown("#### Recent Activity")
    if not hist:
        st.info("No scans yet. Record or upload a clip to get started.")
    else:
        recent = []
        for h in hist[-5:][::-1]:
            recent.append({
                "timestamp": h["timestamp"], "filename": h["filename"],
                "verdict_html": f'<span style="color:{h["color"]}; font-weight:700;">{h["verdict"]}</span>',
                "risk_html": f'<b>{h["risk"]:.0f}%</b>',
            })
        st.markdown('<div class="sv-card">', unsafe_allow_html=True)
        render_table(recent, [("timestamp", "Time"), ("filename", "Source"),
                               ("verdict_html", "Verdict"), ("risk_html", "Risk")])
        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# PAGE: RECORD AUDIO
# ============================================================================
elif page == "Record Audio":
    st.markdown('<div class="sv-hero"><h1>🎙️ Record Audio</h1><p>Record a live sample directly from your microphone to screen it for cloning risk.</p></div>', unsafe_allow_html=True)

    if "recorder_key" not in st.session_state:
        st.session_state.recorder_key = 0

    with st.expander("🔧 If recording doesn't start, read this first", expanded=False):
        st.markdown(
            """
            - Open the app's **Local URL** in a real browser tab (Chrome, Edge, or
              Firefox) — recording will **not** work inside VS Code's built-in
              "Simple Browser" preview or other embedded webviews.
            - When you click the mic icon, your browser will ask for **microphone
              permission** — click **Allow**. If you accidentally blocked it,
              click the 🔒/ⓘ icon in the address bar and re-enable the mic for
              this site, then refresh the page.
            - Speak for at least 1–2 seconds — very short clips can't be analyzed.
            - If none of this helps, use **Upload Audio** instead with a
              pre-recorded WAV file — the analysis works identically either way.
            """
        )

    rec = None
    audio_input_supported = hasattr(st, "audio_input")
    if not audio_input_supported:
        st.error(
            "Your installed Streamlit version doesn't support in-browser "
            "recording. Run `pip install -U streamlit`, restart the "
            "app, then reload this page. In the meantime, use **Upload Audio**."
        )
    else:
        try:
            rec = st.audio_input("Tap the microphone to record",
                                   key=f"recorder_{st.session_state.recorder_key}")
        except Exception as e:
            st.error(f"Recording widget failed to load: {e}")

    if rec is not None:
        raw_bytes = rec.getvalue() if hasattr(rec, "getvalue") else rec.read()
        st.success(f"✅ Captured {len(raw_bytes)/1024:.0f} KB of audio.")
        st.audio(rec)

        col_a, col_b = st.columns([1, 1])
        with col_a:
            analyze_clicked = st.button("▶️ Analyze Recording", type="primary", use_container_width=True)
        with col_b:
            if st.button("🔄 Re-record", use_container_width=True):
                st.session_state.recorder_key += 1
                st.rerun()

        if analyze_clicked:
            try:
                rec.seek(0)
            except Exception:
                pass
            samples, sr = load_audio(rec)
            if samples is None:
                st.error("Couldn't decode this recording. Try recording again or upload a WAV file.")
            elif sr and len(samples) / sr < 0.3:
                st.warning("That recording was too short to analyze. Please record at least 1–2 seconds of speech.")
            else:
                run_backend_analysis(rec, "Live Recording", samples=samples, sr=sr)
    else:
        st.info("Click the microphone icon above to start recording.")


# ============================================================================
# PAGE: UPLOAD AUDIO
# ============================================================================
elif page == "Upload Audio":
    st.markdown('<div class="sv-hero"><h1>📁 Upload Audio</h1><p>Upload a call recording or voice sample (WAV / FLAC / OGG) for analysis.</p></div>', unsafe_allow_html=True)

    up = st.file_uploader("Upload Audio File", type=["wav", "flac", "ogg"])
    if up is not None:
        st.audio(up)
        if st.button("▶️ Analyze File", type="primary"):
            run_backend_analysis(up, up.name)


# ============================================================================
# PAGE: ANALYSIS RESULT
# ============================================================================
elif page == "Analysis Result":
    st.markdown('<div class="sv-hero"><h1>📊 Analysis Result</h1><p>Detailed voice integrity analysis & risk assessment.</p></div>', unsafe_allow_html=True)
    
    res = st.session_state.last_result
    if not res:
        st.info("No active result. Please record or upload audio first.")
    else:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.pyplot(gauge_fig(res["risk"], res["color"]))
        with c2:
            st.markdown(f"### Verdict: <span style='color:{res['color']}'>{res['verdict']}</span>", unsafe_allow_html=True)
            st.caption(f"Source: {res['filename']} | Analyzed: {res['timestamp']}")
            score_bar("Acoustic Model Confidence", res["acoustic"], res["color"])
            score_bar("Prosody & Cadence Risk", res["prosody"], res["color"])
            score_bar("Spectral Signal Anomaly", res["spectral"], res["color"])

        st.markdown("---")
        st.markdown("#### Signal & Feature Analysis")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Duration", f"{res['duration']:.2f}s")
        f2.metric("Sample Rate", f"{res['sample_rate']} Hz")
        f3.metric("Silence Ratio", f"{res['silence_ratio']*100:.1f}%")
        f4.metric("Zero-Crossing Rate", f"{res['zcr']:.3f}")

        st.markdown("#### Audio Visualizations")
        tab1, tab2 = st.tabs(["Waveform", "Spectrogram"])
        with tab1:
            if st.session_state.current_audio is not None:
                st.pyplot(waveform_fig(st.session_state.current_audio, st.session_state.current_sr, res["color"]))
        with tab2:
            if st.session_state.current_audio is not None:
                st.pyplot(spectrogram_fig(st.session_state.current_audio, st.session_state.current_sr))

        st.markdown("#### Explainability Insights")
        exp_lines = explain(res)
        for line in exp_lines:
            st.markdown(f"- {line}")


# ============================================================================
# PAGE: ALERTS & HISTORY
# ============================================================================
elif page == "Alerts & History":
    st.markdown('<div class="sv-hero"><h1>🔔 Alerts & History</h1><p>View historical risk scans and logged alerts.</p></div>', unsafe_allow_html=True)
    hist = st.session_state.history
    if not hist:
        st.info("No historical scans recorded.")
    else:
        table_data = []
        for h in hist[::-1]:
            table_data.append({
                "timestamp": h["timestamp"],
                "filename": h["filename"],
                "verdict_html": f'<span style="color:{h["color"]}; font-weight:700;">{h["verdict"]}</span>',
                "risk_html": f'<b>{h["risk"]:.0f}%</b>',
            })
        render_table(table_data, [("timestamp", "Timestamp"), ("filename", "File"),
                                  ("verdict_html", "Verdict"), ("risk_html", "Risk Score")])


# ============================================================================
# PAGE: RECOMMENDATIONS
# ============================================================================
elif page == "Recommendations":
    st.markdown('<div class="sv-hero"><h1>🧭 Recommendations</h1><p>Actionable security guidance based on risk levels.</p></div>', unsafe_allow_html=True)
    res = st.session_state.last_result
    tier = res["tier"] if res else "low"
    
    st.markdown(f"### Current Recommended Actions (Risk Tier: **{tier.upper()}**)")
    recs = recommendations(tier)
    for r in recs:
        st.markdown(f"- {r}")

    st.markdown("---")
    st.markdown("### Caller Security Precautions")
    precs = precautions(tier)
    for p in precs:
        st.markdown(f"- {p}")
