"""
DocChat AI — Masterpiece UI

This module serves the Streamlit frontend. It includes a custom glassmorphism design
system, dynamic CSS injection, two main modes (Document Q&A and Toxicity Checker),
and asynchronous connections to the FastAPI backend.
"""

import time
import html
import re
import base64
from pathlib import Path
import httpx
import streamlit as st

# -----------------------------------------------------------------------------
# Configuration & CSS
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="DocChat AI",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="expanded"
)


def inject_css():
    """Injects the custom CSS for the glassmorphism design system."""
    # Background: load image and overlay a dark gradient directly — no pseudo-elements
    bg_path = Path(__file__).parent / "background.png"
    if bg_path.exists():
        bg_b64 = base64.b64encode(bg_path.read_bytes()).decode()
        bg_layer = (
            f"background: linear-gradient(135deg,rgba(10,10,15,0.92),rgba(13,13,26,0.88),rgba(10,15,26,0.92)),"
            f"url('data:image/png;base64,{bg_b64}') center/cover no-repeat fixed;"
        )
    else:
        bg_layer = "background: linear-gradient(135deg,#0a0a0f 0%,#0d0d1a 50%,#0a0f1a 100%);"

    st.markdown(f"<style>.stApp{{{bg_layer}min-height:100vh;}}</style>", unsafe_allow_html=True)

    # Main design system CSS
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

* { font-family: 'Inter', sans-serif; box-sizing: border-box; }

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Custom scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #12121a; }
::-webkit-scrollbar-thumb { background: #4f8ef7; border-radius: 2px; }

/* Sidebar glassmorphism - Forced Dark */
section[data-testid="stSidebar"] {
  background: #0a0a14 !important;
  border-right: 1px solid rgba(79,142,247,0.2) !important;
  backdrop-filter: blur(20px);
}

/* Force dark theme for all inputs to prevent HF "wash out" */
.stTextInput input, .stTextArea textarea, [data-testid="stFileUploadDropzone"], div[data-baseweb="select"] > div {
    background-color: rgba(20, 20, 35, 0.8) !important;
    color: white !important;
    border: 1px solid rgba(79, 142, 247, 0.3) !important;
    border-radius: 12px !important;
}

div[data-testid="stExpander"] {
    background: rgba(20, 20, 35, 0.6) !important;
    border: 1px solid rgba(79, 142, 247, 0.2) !important;
    border-radius: 12px !important;
}

/* Title styling */
.main-title {
  font-size: 2.5rem;
  font-weight: 700;
  background: linear-gradient(135deg, #4f8ef7, #9b59f7, #00d4ff);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  text-align: center;
  padding: 1.5rem 0 0.5rem 0;
  letter-spacing: -0.02em;
}

.subtitle {
  text-align: center;
  color: #8888aa;
  font-size: 0.9rem;
  margin-bottom: 2rem;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

/* Mode badge */
.mode-badge {
  display: inline-block;
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}
.mode-qa { background: rgba(79,142,247,0.15); color: #4f8ef7; 
           border: 1px solid rgba(79,142,247,0.3); }
.mode-toxic { background: rgba(255,71,87,0.15); color: #ff4757; 
              border: 1px solid rgba(255,71,87,0.3); }

/* Chat container */
.chat-container {
  max-width: 800px;
  margin: 0 auto;
  padding: 1rem;
}

/* Chat bubbles */
.chat-bubble {
  display: flex;
  gap: 12px;
  margin-bottom: 1.5rem;
  animation: slideIn 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

@keyframes slideIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

.bubble-user {
  justify-content: flex-end;
}

.bubble-content {
  max-width: 85%;
  padding: 14px 18px;
  border-radius: 18px;
  font-size: 0.95rem;
  line-height: 1.6;
}

.bubble-content-user {
  background: linear-gradient(135deg, #4f8ef7, #9b59f7) !important;
  color: white !important;
  border-bottom-right-radius: 4px;
}

.bubble-content-ai {
  background: rgba(30, 30, 45, 0.7) !important;
  border: 1px solid rgba(79, 142, 247, 0.3) !important;
  color: #eef2ff !important;
  border-bottom-left-radius: 4px;
  backdrop-filter: blur(10px);
  box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}

/* Avatar */
.avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1rem;
  flex-shrink: 0;
}
.avatar-ai {
  background: linear-gradient(135deg, #4f8ef7, #9b59f7);
}
.avatar-user {
  background: rgba(255,255,255,0.1);
  border: 1px solid rgba(255,255,255,0.15);
}

/* Toxicity result card */
.tox-card {
  padding: 16px;
  border-radius: 12px;
  margin-top: 8px;
}
.tox-safe {
  background: rgba(0,255,136,0.12) !important;
  border: 1px solid rgba(0,255,136,0.3) !important;
}
.tox-toxic {
  background: rgba(255,71,87,0.12) !important;
  border: 1px solid rgba(255,71,87,0.3) !important;
}
.tox-warning {
  background: rgba(255,165,2,0.12) !important;
  border: 1px solid rgba(255,165,2,0.3) !important;
}

.tox-label {
  font-size: 1rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
}
.label-safe { color: #00ff88 !important; }
.label-toxic { color: #ff4757 !important; }
.label-warning { color: #ffa502 !important; }

.confidence-bar-bg {
  height: 6px;
  background: rgba(255,255,255,0.1);
  border-radius: 3px;
  margin: 10px 0;
  overflow: hidden;
}
.confidence-bar-fill {
  height: 100%;
  border-radius: 3px;
  transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Thinking animation */
.thinking-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #4f8ef7;
  margin: 0 3px;
  animation: pulse 1.4s infinite ease-in-out;
}
.thinking-dot:nth-child(2) { animation-delay: 0.2s; }
.thinking-dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes pulse {
  0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
  40% { transform: scale(1.1); opacity: 1; }
}

/* Document area */
.doc-area {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  border-radius: 12px;
  padding: 16px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.82rem;
  color: #a0a0cc;
  max-height: 180px;
  overflow-y: auto;
  margin-bottom: 12px;
}

/* Stats Row */
.stats-row {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 24px;
}
.stat-card {
  flex: 1;
  background: linear-gradient(145deg, rgba(30,30,45,0.6), rgba(15,15,25,0.8)) !important;
  border: 1px solid rgba(79, 142, 247, 0.2) !important;
  border-radius: 16px;
  padding: 20px;
  text-align: center;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  box-shadow: inset 0 0 10px rgba(79, 142, 247, 0.05), 0 8px 32px rgba(0,0,0,0.3);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
}
.stat-card:hover {
  transform: translateY(-5px);
  border-color: rgba(79, 142, 247, 0.6) !important;
  box-shadow: inset 0 0 15px rgba(79, 142, 247, 0.1), 0 12px 40px rgba(79, 142, 247, 0.2);
}
.stat-number {
  font-size: 2.2rem;
  font-weight: 700;
  background: linear-gradient(135deg, #ffffff, #88b0ff);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 8px;
  font-family: 'JetBrains Mono', monospace;
}
.stat-label {
  font-size: 0.85rem;
  color: #a0a0cc;
  text-transform: uppercase;
  letter-spacing: 0.15em;
  font-weight: 600;
}

/* Chat Input Glassmorphism Fix */
[data-testid="stBottomBlock"], [data-testid="stBottom"] > div {
    background: transparent !important;
}
[data-testid="stChatInput"] {
    background: rgba(20, 20, 35, 0.6) !important;
    backdrop-filter: blur(20px);
    border: 1px solid rgba(79, 142, 247, 0.3) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* Final Button overrides */
.stButton > button {
  background: linear-gradient(135deg, #4f8ef7, #9b59f7) !important;
  color: white !important;
  border: none !important;
  box-shadow: 0 4px 15px rgba(79,142,247,0.3) !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# State Initialization
# -----------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_content" not in st.session_state:
    st.session_state.doc_content = ""
if "total_messages" not in st.session_state:
    st.session_state.total_messages = 0
if "docs_analyzed" not in st.session_state:
    st.session_state.docs_analyzed = 0
if "comments_checked" not in st.session_state:
    st.session_state.comments_checked = 0
if "server_status" not in st.session_state:
    st.session_state.server_status = False


def sanitize_legacy_messages() -> None:
    """Clean malformed historical chat messages created by old renderer logic."""
    cleaned: list[dict] = []
    for msg in st.session_state.messages:
        normalized = dict(msg)
        content = str(normalized.get("content", ""))
        is_user = bool(normalized.get("is_user", False))
        is_html_msg = bool(normalized.get("is_html", False))
        if is_user:
            # Decode potential escaped payload first, then remove any leaked tags.
            decoded = html.unescape(content)
            if ("bubble-content" in decoded) or ("avatar-user" in decoded) or ("<" in decoded and ">" in decoded):
                decoded = re.sub(r"<[^>]+>", " ", decoded)
                normalized["content"] = " ".join(decoded.split())
                normalized["is_html"] = False
            else:
                normalized["content"] = content
        elif not is_html_msg:
            normalized["content"] = content
        cleaned.append(normalized)
    st.session_state.messages = cleaned

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def render_message(is_user: bool, content: str, is_html: bool = False):
    """Renders a single chat bubble."""
    bubble_class = "bubble-user" if is_user else ""
    content_class = "bubble-content-user" if is_user else "bubble-content-ai"
    avatar_class = "avatar-user" if is_user else "avatar-ai"

    if is_user:
        ai_avatar = ''
        user_avatar = '<div class="avatar avatar-user" style="font-size:1rem;">&#128100;</div>'
    else:
        if _logo_b64:
            ai_avatar = f'<div class="avatar avatar-ai" style="padding:0;overflow:hidden;"><img src="data:image/png;base64,{_logo_b64}" style="width:100%;height:100%;object-fit:cover;border-radius:50%;"/></div>'
        else:
            ai_avatar = '<div class="avatar avatar-ai">⚡</div>'
        user_avatar = ''

    if is_user:
        # Hard guard: never allow HTML-like payload in user bubble content.
        content = re.sub(r"<[^>]+>", " ", html.unescape(str(content)))
        content = " ".join(content.split())
    safe_content = content if is_html else html.escape(content).replace("\n", "<br>")
    bubble_html = f"""<div class="chat-bubble {bubble_class}">{ai_avatar}<div class="bubble-content {content_class}">{safe_content}</div>{user_avatar}</div>"""
    st.markdown(bubble_html, unsafe_allow_html=True)

def render_thinking():
    """Renders the animated thinking dots."""
    if _logo_b64:
        logo_tag = f'<div class="avatar avatar-ai" style="padding:0;overflow:hidden;"><img src="data:image/png;base64,{_logo_b64}" style="width:100%;height:100%;object-fit:cover;border-radius:50%;"/></div>'
    else:
        logo_tag = '<div class="avatar avatar-ai">⚡</div>'
    thinking_html = (f'<div class="chat-bubble">'
        f'{logo_tag}'
        f'<div class="bubble-content bubble-content-ai" style="padding:16px;">'
        f'<span class="thinking-dot"></span><span class="thinking-dot"></span><span class="thinking-dot"></span>'
        f'</div></div>')
    return st.empty().markdown(thinking_html, unsafe_allow_html=True)

def check_server_health(url: str):
    """Checks if the FastAPI server is running."""
    try:
        response = httpx.get(f"{url}/health", timeout=30.0)
        return response.status_code == 200
    except Exception:
        return False

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        # Sidebar logo + title
        if _logo_b64:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;padding:0.5rem 0;">'
                f'<img src="data:image/png;base64,{_logo_b64}" style="width:36px;height:36px;border-radius:50%;box-shadow:0 0 12px rgba(79,142,247,0.5);"/>'
                f'<span class="main-title" style="font-size:1.3rem;padding:0;margin:0;">DocChat AI</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown('<div class="main-title" style="font-size: 1.5rem;">⚡ DocChat AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-label">Server Configuration</div>', unsafe_allow_html=True)
        server_url = st.text_input("Server URL", value="http://localhost:8000")
        api_key = st.text_input("API Key", value="docchat-secret-key", type="password")
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-label">Mode Selection</div>', unsafe_allow_html=True)
        mode = st.radio(
            "Select Mode", 
            ["📄 Document Q&A", "🛡️ Toxicity Checker"],
            label_visibility="collapsed"
        )
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.0, max_value=1.0, value=0.5, step=0.05
        )
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        if st.button("Check Connection"):
            st.session_state.server_status = check_server_health(server_url)
            
        status_color = "#00ff88" if st.session_state.server_status else "#ff4757"
        status_text = "Online" if st.session_state.server_status else "Offline"
        st.markdown(
            f'<div style="display: flex; align-items: center; gap: 8px; margin-top: 10px;">'
            f'<div style="width: 10px; height: 10px; border-radius: 50%; background: {status_color};"></div>'
            f'<span style="color: {status_color}; font-size: 0.9rem;">Server {status_text}</span>'
            f'</div>', 
            unsafe_allow_html=True
        )
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-label">Session Stats Summary</div>', unsafe_allow_html=True)
        st.caption(f"Messages: {st.session_state.total_messages}")
        st.caption(f"Docs analyzed: {st.session_state.docs_analyzed}")
        st.caption(f"Comments checked: {st.session_state.comments_checked}")
            
    return server_url, api_key, mode, confidence_threshold

# -----------------------------------------------------------------------------
# Main App Layout
# -----------------------------------------------------------------------------

inject_css()
sanitize_legacy_messages()

# Load logo as base64 so it works in all environments
_logo_path = Path(__file__).parent / "logo.png"
_logo_b64 = ""
if _logo_path.exists():
    _logo_b64 = base64.b64encode(_logo_path.read_bytes()).decode()

# Render Header with logo
if _logo_b64:
    st.markdown(
        f'<div style="display:flex;align-items:center;justify-content:center;gap:16px;padding:1.5rem 0 0.2rem 0;">'
        f'<img src="data:image/png;base64,{_logo_b64}" style="width:72px;height:72px;border-radius:50%;box-shadow:0 0 30px rgba(79,142,247,0.6);"/>'
        f'<span class="main-title" style="padding:0;margin:0;">DocChat AI</span>'
        f'</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown('<div class="main-title">⚡ DocChat AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Fine-Tuned LLM · Document Q&amp;A · Toxicity Detection</div>', unsafe_allow_html=True)

# Render Stats Row
stats_html = f"""
<div class="stats-row">
    <div class="stat-card">
        <div class="stat-number">{st.session_state.total_messages}</div>
        <div class="stat-label">Messages Sent</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">{st.session_state.docs_analyzed}</div>
        <div class="stat-label">Docs Analyzed</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">{st.session_state.comments_checked}</div>
        <div class="stat-label">Comments Checked</div>
    </div>
</div>
"""
st.markdown(stats_html, unsafe_allow_html=True)

server_url, api_key, mode, confidence_threshold = render_sidebar()

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Mode Specific Logic
if "Document Q&A" in mode:
    st.markdown('<span class="mode-badge mode-qa">Document Q&A</span>', unsafe_allow_html=True)
    with st.expander("📄 Upload or Paste Document", expanded=True if not st.session_state.doc_content else False):
        uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])
        pasted_text = st.text_area("Or paste document text here:")
        
        if st.button("Load Document"):
            if uploaded_file is not None:
                ext = uploaded_file.name.split('.')[-1].lower()
                content = ""
                try:
                    if ext == "txt":
                        content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
                    elif ext == "pdf":
                        import pypdf
                        import io
                        pdf = pypdf.PdfReader(io.BytesIO(uploaded_file.getvalue()))
                        content = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                    elif ext == "docx":
                        import docx
                        import io
                        doc = docx.Document(io.BytesIO(uploaded_file.getvalue()))
                        content = "\n".join([para.text for para in doc.paragraphs])
                    st.session_state.doc_content = content
                    st.session_state.docs_analyzed += 1
                except Exception as e:
                    st.error(f"Failed to read {ext.upper()} file: {e}")
            elif pasted_text.strip():
                st.session_state.doc_content = pasted_text
                st.session_state.docs_analyzed += 1
                
            if st.session_state.doc_content:
                st.rerun()
                
    if st.session_state.doc_content:
        char_count = len(st.session_state.doc_content)
        token_est = char_count // 4
        st.markdown(f'<div class="success-box">✓ Document loaded ({char_count} chars, ~{token_est} tokens)</div>', unsafe_allow_html=True)
        
        preview = st.session_state.doc_content[:200] + ("..." if char_count > 200 else "")
        st.markdown(f'<div class="doc-area">{preview}</div>', unsafe_allow_html=True)
        
    chat_prompt = "Ask anything about your document..."

else:
    st.markdown('<span class="mode-badge mode-toxic">Toxicity Checker</span>', unsafe_allow_html=True)
    chat_prompt = "Type a comment to analyze..."

# Render Chat History
for msg in st.session_state.messages:
    render_message(msg["is_user"], msg["content"], is_html=msg.get("is_html", False))

# Handle Input
if user_input := st.chat_input(chat_prompt):
    sanitized_input = re.sub(r"<[^>]+>", " ", html.unescape(user_input))
    sanitized_input = " ".join(sanitized_input.split())
    # Add user message
    st.session_state.messages.append({"is_user": True, "content": sanitized_input})
    st.session_state.total_messages += 1
    
    render_message(True, sanitized_input, is_html=False)
    
    # Placeholder for thinking animation
    thinking_placeholder = st.empty()
    if _logo_b64:
        _ai_logo_tag = f'<div class="avatar avatar-ai" style="padding:0;overflow:hidden;"><img src="data:image/png;base64,{_logo_b64}" style="width:100%;height:100%;object-fit:cover;border-radius:50%;"/></div>'
    else:
        _ai_logo_tag = '<div class="avatar avatar-ai">⚡</div>'
    thinking_html = (f'<div class="chat-bubble">'
        f'{_ai_logo_tag}'
        f'<div class="bubble-content bubble-content-ai" style="padding:16px;">'
        f'<span class="thinking-dot"></span><span class="thinking-dot"></span><span class="thinking-dot"></span>'
        f'</div></div>')
    thinking_placeholder.markdown(thinking_html, unsafe_allow_html=True)
    
    headers = {"X-API-Key": api_key}
    client = httpx.Client(timeout=600.0)
    
    try:
        if "Document Q&A" in mode:
            if not st.session_state.doc_content:
                response_html = '<div class="error-box">Please load a document first.</div>'
            else:
                resp = client.post(
                    f"{server_url}/ask",
                    json={"doc": st.session_state.doc_content, "question": sanitized_input},
                    headers=headers
                )
                if resp.status_code >= 500:
                    time.sleep(1.0)
                    resp = client.post(
                        f"{server_url}/ask",
                        json={"doc": st.session_state.doc_content, "question": sanitized_input},
                        headers=headers
                    )
                
                if resp.status_code == 200:
                    data = resp.json()
                    response_html = data.get("answer", "No answer generated.")
                elif resp.status_code == 401:
                    response_html = '<div class="error-box">Invalid API key</div>'
                elif resp.status_code == 503:
                    response_html = '<div class="error-box">Model not loaded on server</div>'
                else:
                    response_html = f'<div class="error-box">Server error: {resp.status_code}</div>'
                    
        else: # Toxicity Mode
            st.session_state.comments_checked += 1
            resp = client.post(
                f"{server_url}/predict",
                json={"comment": sanitized_input},
                headers=headers
            )
            if resp.status_code >= 500:
                time.sleep(1.0)
                resp = client.post(
                    f"{server_url}/predict",
                    json={"comment": sanitized_input},
                    headers=headers
                )
            
            if resp.status_code == 200:
                data = resp.json()
                label = data.get("label", "unknown")
                confidence = data.get("confidence", 0.0)
                explanation = data.get("explanation", "")
                
                if confidence < confidence_threshold:
                    response_html = (f'<div class="tox-card tox-warning">'
                        f'<div class="tox-label label-warning">⚠️ LOW CONFIDENCE</div>'
                        f'<div style="color:#8888aa;font-size:0.9rem;margin-top:8px;">'
                        f'Result hidden. Model confidence ({confidence:.2f}) is below threshold ({confidence_threshold}).'
                        f'</div></div>')
                else:
                    if label in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]:
                        card_class = "tox-toxic"
                        label_class = "label-toxic"
                        bar_color = "#ff4757"
                    elif label == "safe":
                        card_class = "tox-safe"
                        label_class = "label-safe"
                        bar_color = "#00ff88"
                    else:
                        card_class = "tox-warning"
                        label_class = "label-warning"
                        bar_color = "#ffa502"
                        
                    bar_width = int(confidence * 100)
                    response_html = (f'<div class="tox-card {card_class}">'
                        f'<div class="tox-label {label_class}">{label.replace("_", " ").upper()}</div>'
                        f'<div class="confidence-bar-bg">'
                        f'<div class="confidence-bar-fill" style="width:{bar_width}%;background:{bar_color};"></div>'
                        f'</div>'
                        f'<div style="color:#f0f0ff;font-size:0.9rem;">{html.escape(explanation)}</div>'
                        f'<div style="color:#8888aa;font-size:0.8rem;margin-top:4px;">Confidence: {confidence:.2f}</div>'
                        f'</div>')
            elif resp.status_code == 401:
                response_html = '<div class="error-box">Invalid API key</div>'
            elif resp.status_code == 503:
                response_html = '<div class="error-box">Model not loaded on server</div>'
            else:
                response_html = f'<div class="error-box">Server error: {resp.status_code}</div>'
                
    except httpx.ConnectError:
        response_html = '<div class="error-box">Cannot connect to DocChat server</div>'
    except httpx.TimeoutException:
        response_html = '<div class="error-box">Request timed out. Model may be processing.</div>'
    except Exception as e:
        response_html = f'<div class="error-box">An unexpected error occurred: {str(e)}</div>'
        
    finally:
        client.close()
        
    # Remove thinking and add AI response
    thinking_placeholder.empty()
    st.session_state.messages.append({"is_user": False, "content": response_html, "is_html": True})
    # render_message(False, response_html, is_html=True)
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
