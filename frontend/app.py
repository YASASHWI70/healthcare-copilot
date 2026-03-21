"""
frontend/app.py - Streamlit Healthcare Copilot UI

A rich, production-style chat interface featuring:
  - Multi-turn conversation with memory
  - Structured output display (symptoms, conditions, risk, recommendations)
  - PDF upload support
  - Query history dashboard
  - Animated risk-level indicators
  - Reasoning chain viewer
"""

import streamlit as st
import requests
import uuid
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

# ─── Page Config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="MediCopilot — AI Healthcare Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Configuration ────────────────────────────────────────────────────────────
import os
from dotenv import load_dotenv
load_dotenv()

BACKEND_URL = os.getenv("FRONTEND_BACKEND_URL", "http://localhost:8000")
API_BASE = f"{BACKEND_URL}/api/v1"

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Global ── */
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  /* ── Hide Streamlit defaults ── */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  header {visibility: hidden;}

  /* ── Risk badges ── */
  .risk-low {
    background: linear-gradient(135deg, #d1fae5, #a7f3d0);
    color: #065f46;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.9rem;
    display: inline-block;
    border: 1px solid #6ee7b7;
  }
  .risk-moderate {
    background: linear-gradient(135deg, #fef3c7, #fde68a);
    color: #92400e;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.9rem;
    display: inline-block;
    border: 1px solid #fbbf24;
  }
  .risk-high {
    background: linear-gradient(135deg, #fee2e2, #fca5a5);
    color: #991b1b;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.9rem;
    display: inline-block;
    border: 1px solid #f87171;
    animation: pulse 1.5s infinite;
  }
  .risk-unknown {
    background: #f3f4f6;
    color: #6b7280;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.9rem;
    display: inline-block;
  }

  @keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.4); }
    50% { box-shadow: 0 0 0 8px rgba(239,68,68,0); }
  }

  /* ── Cards ── */
  .info-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }
  .info-card-header {
    font-weight: 600;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6b7280;
    margin-bottom: 10px;
  }

  /* ── Symptom tags ── */
  .symptom-tag {
    display: inline-block;
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    border-radius: 999px;
    padding: 3px 12px;
    font-size: 0.82rem;
    font-weight: 500;
    margin: 3px;
  }
  .symptom-tag.severe { background: #fef2f2; color: #dc2626; border-color: #fca5a5; }
  .symptom-tag.moderate { background: #fffbeb; color: #d97706; border-color: #fcd34d; }

  /* ── Condition card ── */
  .condition-item {
    border-left: 3px solid #6366f1;
    padding: 8px 14px;
    margin-bottom: 8px;
    background: #fafafa;
    border-radius: 0 8px 8px 0;
  }
  .condition-item.high { border-left-color: #ef4444; }
  .condition-item.moderate { border-left-color: #f59e0b; }
  .condition-item.low { border-left-color: #10b981; }

  /* ── Reasoning steps ── */
  .reasoning-step {
    position: relative;
    padding: 12px 16px 12px 44px;
    margin-bottom: 10px;
    background: #f8fafc;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
  }
  .step-number {
    position: absolute;
    left: 12px;
    top: 12px;
    width: 24px;
    height: 24px;
    background: #6366f1;
    color: white;
    border-radius: 50%;
    font-size: 0.75rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  /* ── Chat messages ── */
  .chat-user {
    background: #eff6ff;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 80%;
    margin-left: auto;
    color: #1e3a5f;
    font-size: 0.93rem;
  }
  .chat-assistant {
    background: #f0fdf4;
    border-radius: 16px 16px 16px 4px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 80%;
    color: #14532d;
    font-size: 0.93rem;
    border: 1px solid #bbf7d0;
  }

  /* ── Disclaimer ── */
  .disclaimer-box {
    background: #fffbeb;
    border: 1px solid #fbbf24;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.78rem;
    color: #92400e;
    margin-top: 16px;
  }

  /* ── Recommendation list ── */
  .rec-item {
    display: flex;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid #f3f4f6;
    font-size: 0.88rem;
    align-items: flex-start;
  }
  .rec-num {
    flex-shrink: 0;
    width: 22px;
    height: 22px;
    background: #6366f1;
    color: white;
    border-radius: 50%;
    font-size: 0.7rem;
    font-weight: 700;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-top: 1px;
  }

  /* ── Follow-up questions ── */
  .followup-q {
    background: #f5f3ff;
    border: 1px solid #ddd6fe;
    border-radius: 8px;
    padding: 8px 14px;
    margin: 6px 0;
    font-size: 0.85rem;
    color: #5b21b6;
    cursor: pointer;
  }

  /* ── Header banner ── */
  .app-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f4c81 100%);
    color: white;
    padding: 20px 28px;
    border-radius: 16px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
  }
  .app-header h1 {
    margin: 0;
    font-size: 1.6rem;
    font-weight: 700;
    color: white !important;
  }
  .app-header p {
    margin: 4px 0 0 0;
    font-size: 0.85rem;
    opacity: 0.75;
    color: white !important;
  }

  /* ── Section headers ── */
  .section-title {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748b;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 2px solid #e2e8f0;
  }
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ────────────────────────────────────────────
def init_session():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "last_response" not in st.session_state:
        st.session_state.last_response = None
    if "pdf_text" not in st.session_state:
        st.session_state.pdf_text = None
    if "pdf_filename" not in st.session_state:
        st.session_state.pdf_filename = None
    if "query_log" not in st.session_state:
        st.session_state.query_log = []

init_session()


# ─── Helper Functions ─────────────────────────────────────────────────────────

def get_risk_badge(risk_level: str) -> str:
    level = risk_level.lower()
    icons = {"low": "🟢", "moderate": "🟡", "high": "🔴", "unknown": "⚪"}
    icon = icons.get(level, "⚪")
    return f'<span class="risk-{level}">{icon} {level.upper()}</span>'


def get_confidence_color(confidence: str) -> str:
    mapping = {"high": "#10b981", "moderate": "#f59e0b", "low": "#6b7280"}
    return mapping.get(confidence.lower(), "#6b7280")


def call_chat_api(message: str, pdf_text: Optional[str] = None) -> Optional[Dict]:
    """Call the backend chat API."""
    try:
        payload = {
            "session_id": st.session_state.session_id,
            "message": message,
            "conversation_history": [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.conversation_history[-10:]
            ],
            "pdf_text": pdf_text,
        }
        response = requests.post(
            f"{API_BASE}/chat",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend. Is the FastAPI server running on port 8000?")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. The AI is taking too long — please try again.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API Error {e.response.status_code}: {e.response.text[:200]}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return None


def upload_pdf_api(uploaded_file) -> Optional[Dict]:
    """Upload PDF to backend for text extraction."""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        response = requests.post(
            f"{API_BASE}/upload-pdf",
            files=files,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"❌ PDF upload failed: {str(e)}")
        return None


def render_response(data: Dict):
    """Render the structured healthcare response."""

    # ─── Assistant Message ─────────────────────────────────────────────────
    st.markdown(
        f'<div class="chat-assistant">🤖 {data.get("assistant_message", "")}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ─── Three-column layout ───────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        # Extracted Symptoms
        st.markdown('<div class="section-title">🔍 Extracted Symptoms</div>', unsafe_allow_html=True)
        symptoms = data.get("extracted_symptoms", [])
        if symptoms:
            tags_html = ""
            for s in symptoms:
                sev = s.get("severity", "") or ""
                css_class = f"symptom-tag {sev}" if sev in ("severe", "moderate") else "symptom-tag"
                label = s.get("name", "")
                if sev:
                    label += f" ({sev})"
                if s.get("duration"):
                    label += f" · {s['duration']}"
                tags_html += f'<span class="{css_class}">{label}</span>'
            st.markdown(tags_html, unsafe_allow_html=True)
        else:
            st.caption("No specific symptoms extracted.")

    with col2:
        # Risk Level
        st.markdown('<div class="section-title">⚠️ Risk Assessment</div>', unsafe_allow_html=True)
        risk = data.get("risk_level", "unknown")
        st.markdown(get_risk_badge(risk), unsafe_allow_html=True)
        st.markdown(f"<small style='color:#64748b'>{data.get('risk_rationale', '')[:200]}</small>", unsafe_allow_html=True)

    with col3:
        # Possible Conditions
        st.markdown('<div class="section-title">🩺 Possible Conditions</div>', unsafe_allow_html=True)
        conditions = data.get("possible_conditions", [])
        if conditions:
            for c in conditions[:4]:
                conf = c.get("confidence", "low")
                color = get_confidence_color(conf)
                st.markdown(
                    f"""<div class="condition-item {conf}">
                        <strong style="font-size:0.88rem">{c.get('name','')}</strong>
                        <span style="float:right;font-size:0.75rem;color:{color};font-weight:600">{conf.upper()}</span>
                        <div style="font-size:0.76rem;color:#64748b;margin-top:2px">
                            {', '.join(c.get('matching_symptoms',[])[:3])}
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Insufficient symptoms for differential.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Explanation + Reasoning (tabs) ───────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📝 Explanation", "🧠 Reasoning Chain", "📋 Recommendations"])

    with tab1:
        explanation = data.get("explanation", "")
        if explanation:
            st.markdown(
                f'<div class="info-card"><div class="info-card-header">Plain-Language Explanation</div>{explanation}</div>',
                unsafe_allow_html=True,
            )

        # Follow-up questions
        follow_ups = data.get("follow_up_questions", [])
        if follow_ups:
            st.markdown('<div class="section-title" style="margin-top:16px">💬 Follow-Up Questions to Consider</div>', unsafe_allow_html=True)
            for q in follow_ups:
                st.markdown(f'<div class="followup-q">💡 {q}</div>', unsafe_allow_html=True)

    with tab2:
        steps = data.get("reasoning_steps", [])
        if steps:
            for step in steps:
                st.markdown(
                    f"""<div class="reasoning-step">
                        <div class="step-number">{step.get('step_number','')}</div>
                        <strong style="font-size:0.88rem;color:#1e293b">{step.get('title','')}</strong>
                        <div style="font-size:0.85rem;color:#475569;margin-top:6px">{step.get('content','')}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Reasoning steps not available for this query.")

        # RAG context
        rag_snippets = data.get("rag_context_used", [])
        if rag_snippets:
            with st.expander("📚 Retrieved Medical Knowledge (RAG Context)"):
                for i, snippet in enumerate(rag_snippets, 1):
                    st.markdown(f"**Snippet {i}:**")
                    st.text(snippet[:300] + "..." if len(snippet) > 300 else snippet)

    with tab3:
        recs = data.get("recommendations", [])
        if recs:
            for i, rec in enumerate(recs, 1):
                st.markdown(
                    f'<div class="rec-item"><span class="rec-num">{i}</span><span>{rec}</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No recommendations generated.")

    # ─── Disclaimer ───────────────────────────────────────────────────────
    disclaimer = data.get("disclaimer", "")
    if disclaimer:
        st.markdown(
            f'<div class="disclaimer-box">⚠️ {disclaimer}</div>',
            unsafe_allow_html=True,
        )

    # ─── Agent Trace (debug) ───────────────────────────────────────────────
    trace = data.get("agent_trace", {})
    if trace:
        with st.expander("🔧 Agent Execution Trace"):
            cols = st.columns(len(trace))
            status_icons = {"success": "✅", "error": "❌", "skipped": "⏭️"}
            for i, (agent, status) in enumerate(trace.items()):
                with cols[i]:
                    icon = status_icons.get(status, "❓")
                    st.markdown(f"**{agent.replace('_', ' ').title()}**")
                    st.markdown(f"{icon} {status}")


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏥 MediCopilot")
    st.markdown("*AI Healthcare Assistant*")
    st.markdown("---")

    # Session info
    st.markdown(f"**Session ID:** `{st.session_state.session_id}`")
    st.markdown(f"**Turns:** {len(st.session_state.conversation_history) // 2}")

    st.markdown("---")

    # PDF Upload
    st.markdown("### 📎 Upload Medical Report")
    st.caption("Upload a PDF report to include in analysis")
    uploaded_pdf = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed",
        key="pdf_uploader",
    )

    if uploaded_pdf:
        if uploaded_pdf.name != st.session_state.pdf_filename:
            with st.spinner("Extracting PDF text..."):
                result = upload_pdf_api(uploaded_pdf)
            if result and result.get("success"):
                st.session_state.pdf_text = result["extracted_text"]
                st.session_state.pdf_filename = uploaded_pdf.name
                st.success(f"✅ {uploaded_pdf.name} ({result['page_count']} pages)")
                st.caption(f"{len(result['extracted_text'])} characters extracted")
            else:
                error = result.get("error", "Unknown error") if result else "Upload failed"
                st.error(f"❌ {error}")
    elif st.session_state.pdf_filename:
        st.info(f"📄 Active: {st.session_state.pdf_filename}")
        if st.button("Clear PDF"):
            st.session_state.pdf_text = None
            st.session_state.pdf_filename = None
            st.rerun()

    st.markdown("---")

    # Example prompts
    st.markdown("### 💡 Example Prompts")

    EXAMPLE_PROMPTS = [
        "I have a high fever of 39.5°C, severe body aches, and I've been extremely tired for 2 days",
        "I've been having sharp chest pain on the left side, shortness of breath, and my arm feels numb",
        "I have a headache, runny nose, mild sore throat and sneezing since yesterday",
        "I feel very dizzy, nauseous, and have been vomiting. I also have severe abdominal pain",
        "I've had a persistent dry cough for 3 weeks, mild fever, and I lost my sense of smell",
    ]

    for prompt in EXAMPLE_PROMPTS:
        if st.button(f"📌 {prompt[:45]}...", key=prompt, use_container_width=True):
            st.session_state["prefill_message"] = prompt

    st.markdown("---")

    # Controls
    st.markdown("### ⚙️ Controls")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.conversation_history = []
        st.session_state.last_response = None
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()

    # Backend health check
    if st.button("🔌 Check Backend", use_container_width=True):
        try:
            r = requests.get(f"{API_BASE}/health", timeout=5)
            if r.status_code == 200:
                data = r.json()
                st.success(f"✅ Backend online\nModel: {data.get('model','?')}\nRAG: {'✅' if data.get('rag_index_loaded') else '⚠️'}")
            else:
                st.error(f"❌ Backend error: {r.status_code}")
        except Exception:
            st.error("❌ Backend unreachable")

    st.markdown("---")
    st.markdown(
        "<small style='color:#94a3b8'>⚠️ For educational purposes only.<br>Not medical advice.</small>",
        unsafe_allow_html=True,
    )


# ─── Main Content ─────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="app-header">
  <div>🏥</div>
  <div>
    <h1>MediCopilot — AI Healthcare Assistant</h1>
    <p>Multi-Agent Clinical Decision Support · Powered by GPT-4 · RAG-Grounded · For Educational Use Only</p>
  </div>
</div>
""", unsafe_allow_html=True)


# Main tabs
main_tab, history_tab = st.tabs(["💬 Chat", "📊 Query History"])

with main_tab:
    # ── Conversation History ───────────────────────────────────────────────
    if st.session_state.conversation_history:
        st.markdown("### Conversation")
        for msg in st.session_state.conversation_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">👤 {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            # Assistant messages are rendered as structured cards from last_response

    # ── Last Response ──────────────────────────────────────────────────────
    if st.session_state.last_response:
        st.markdown("### 📊 Analysis Report")
        render_response(st.session_state.last_response)

    # ── Input Area ─────────────────────────────────────────────────────────
    st.markdown("---")

    # Pre-fill from sidebar example
    default_msg = st.session_state.pop("prefill_message", "")

    with st.form("chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])

        with col_input:
            user_input = st.text_area(
                "Describe your symptoms",
                value=default_msg,
                placeholder="e.g. I've had a fever of 38.5°C, sore throat, and body aches for 2 days...",
                height=80,
                label_visibility="collapsed",
            )

        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button(
                "Analyze →",
                use_container_width=True,
                type="primary",
            )

    # ── Handle Submission ──────────────────────────────────────────────────
    if submitted and user_input.strip():
        # Add user message to history
        st.session_state.conversation_history.append({
            "role": "user",
            "content": user_input.strip(),
        })

        with st.spinner("🤖 Analyzing your symptoms through multi-agent pipeline..."):
            response_data = call_chat_api(
                user_input.strip(),
                pdf_text=st.session_state.pdf_text,
            )

        if response_data:
            # Add assistant reply to history
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": response_data.get("assistant_message", "Analysis complete."),
            })

            st.session_state.last_response = response_data

            # Log to history
            st.session_state.query_log.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "query": user_input.strip()[:80],
                "risk": response_data.get("risk_level", "unknown"),
                "conditions": len(response_data.get("possible_conditions", [])),
                "symptoms": len(response_data.get("extracted_symptoms", [])),
            })

            st.rerun()

    elif submitted and not user_input.strip():
        st.warning("Please describe your symptoms before submitting.")


with history_tab:
    st.markdown("### 📊 Query History (This Session)")

    if st.session_state.query_log:
        for i, entry in enumerate(reversed(st.session_state.query_log)):
            risk = entry.get("risk", "unknown").lower()
            risk_colors = {"low": "#10b981", "moderate": "#f59e0b", "high": "#ef4444", "unknown": "#6b7280"}
            color = risk_colors.get(risk, "#6b7280")

            with st.container():
                c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
                with c1:
                    st.markdown(f"**{entry['query']}**")
                    st.caption(f"🕐 {entry['timestamp']}")
                with c2:
                    st.markdown(f"<span style='color:{color};font-weight:700'>{risk.upper()}</span>", unsafe_allow_html=True)
                with c3:
                    st.caption(f"🩺 {entry['conditions']} conditions")
                with c4:
                    st.caption(f"🔍 {entry['symptoms']} symptoms")
                st.divider()
    else:
        st.info("No queries yet in this session. Start a conversation!")

    # Fetch server-side history
    if st.button("🔄 Fetch Server History"):
        try:
            r = requests.get(f"{API_BASE}/history?limit=20", timeout=10)
            if r.status_code == 200:
                server_history = r.json()
                if server_history:
                    st.markdown("### 🖥️ Server-Side Query Log")
                    for entry in server_history:
                        risk = entry.get("risk_level", "unknown").lower()
                        risk_colors = {"low": "#10b981", "moderate": "#f59e0b", "high": "#ef4444", "unknown": "#6b7280"}
                        color = risk_colors.get(risk, "#6b7280")
                        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
                        with c1:
                            st.markdown(f"**{entry.get('query','')[:80]}**")
                            st.caption(f"🕐 {entry.get('timestamp','')[:19]}")
                        with c2:
                            st.markdown(f"<span style='color:{color};font-weight:700'>{risk.upper()}</span>", unsafe_allow_html=True)
                        with c3:
                            st.caption(f"🩺 {entry.get('conditions_count',0)} conditions")
                        with c4:
                            st.caption(f"🔍 {entry.get('symptoms_count',0)} symptoms")
                        st.divider()
                else:
                    st.info("No server history found.")
        except Exception as e:
            st.error(f"Failed to fetch server history: {e}")
