"""
EFS Core Engine — Dynamic Streamlit App
- LlamaParse  : document ingestion → raw markdown (displayed as JSON)
- Groq API    : dynamically generates PlantUML sequence + FSM from document content
- Kroki.io    : renders PlantUML strings → inline SVG (no local install needed)
"""

import os
import json
import base64
import zlib
import tempfile
import re
import requests
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "")
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
KROKI_URL           = "https://kroki.io/plantuml/svg"

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EFS Core Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Space+Grotesk:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: #0d0f14; color: #e2e8f0; }
section[data-testid="stSidebar"] { background: #111318 !important; border-right: 1px solid #1e2330; }

.stage-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase;
    display: inline-block; padding: 3px 10px; border-radius: 4px; margin-bottom: 6px;
}
.badge-purple { background: #2d1f5e; color: #c4b5fd; }
.badge-teal   { background: #0c2e24; color: #5eead4; }
.badge-amber  { background: #3d2c0e; color: #fcd34d; }
.badge-blue   { background: #1e3a5f; color: #7dd3fc; }

.section-header {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem;
    letter-spacing: 0.18em; text-transform: uppercase; color: #475569;
    margin: 1.2rem 0 0.5rem; padding-bottom: 4px;
    border-bottom: 1px solid #1e293b;
}
.metric-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 0.5rem 0 1rem; }
.metric-box { background: #13161f; border: 1px solid #1e2330; border-radius: 8px;
              padding: 0.5rem 1rem; text-align: center; min-width: 80px; }
.metric-box .val { font-family: 'IBM Plex Mono', monospace; font-size: 1.3rem;
                   font-weight: 700; color: #38bdf8; }
.metric-box .lbl { font-size: 0.62rem; color: #64748b; text-transform: uppercase; letter-spacing: .1em; }

.json-wrap { background: #0a0c10; border: 1px solid #1e2330; border-radius: 8px;
             padding: 1rem; overflow-x: auto; max-height: 500px; overflow-y: auto; }

.puml-wrap { background: #0a0c10; border: 1px solid #1e2330; border-radius: 8px;
             padding: 1rem; font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem;
             line-height: 1.7; color: #a5f3fc; overflow-x: auto; white-space: pre; }

.svg-wrap { background: #ffffff; border-radius: 10px; padding: 1rem;
            display: flex; justify-content: center; }
.svg-wrap svg { max-width: 100%; height: auto; }

.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important; font-weight: 600 !important;
    letter-spacing: 0.06em !important;
}
.stButton > button:hover { background: linear-gradient(135deg, #2563eb, #3b82f6) !important; }
hr { border-color: #1e2330 !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def parse_with_llamaparse(file_bytes: bytes, filename: str) -> list[dict]:
    """Call LlamaParse and return list of page dicts with text + metadata."""
    from llama_parse import LlamaParse

    parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="markdown",
        verbose=False,
        language="en",
        parsing_instruction=(
            "Extract all text including section headings, tables, register maps, "
            "signal names, actors, states, protocol conditions, and timing rules. "
            "Preserve document hierarchy using markdown headings."
        ),
    )
    suffix = Path(filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(file_bytes)
        tmp = f.name
    try:
        docs = parser.load_data(tmp)
    finally:
        os.unlink(tmp)

    pages = []
    for i, doc in enumerate(docs):
        pages.append({
            "page": i + 1,
            "char_count": len(doc.text),
            "word_count": len(doc.text.split()),
            "text": doc.text,
            "metadata": getattr(doc, "metadata", {}),
        })
    return pages


def llamaparse_to_json(pages: list[dict]) -> dict:
    """Build the structured JSON we display in the UI."""
    combined = "\n\n".join(p["text"] for p in pages)
    return {
        "source": "LlamaParse",
        "result_type": "markdown",
        "total_pages": len(pages),
        "total_chars": sum(p["char_count"] for p in pages),
        "total_words": sum(p["word_count"] for p in pages),
        "pages": pages,
        "combined_text_preview": combined[:1500] + ("…" if len(combined) > 1500 else ""),
    }


SPEC_SYSTEM = """You are an expert EDA/semiconductor specification analyst.
Given raw markdown extracted from a protocol specification document, extract a structured spec model.

Return ONLY a valid JSON object — no markdown fences, no commentary — with exactly these keys:
{
  "protocol": "<primary protocol name, e.g. AXI4, PCIe, CXL, CHI, AHB, APB, or UNKNOWN>",
  "actors": ["<actor1>", "<actor2>", ...],
  "signals": ["<signal1>", ...],
  "registers": ["<reg1>", ...],
  "conditions": [
    {"raw": "<original text>", "normalized": "<IF signal==1 AND signal==0 style>"},
    ...
  ],
  "transactions": [
    {"name": "<txn name>", "steps": ["<step1>", "<step2>", ...]},
    ...
  ],
  "states": ["<state1>", "<state2>", ...],
  "state_transitions": [
    {"from": "<state>", "to": "<state>", "trigger": "<event or condition>"},
    ...
  ],
  "key_rules": ["<rule1>", "<rule2>", ...]
}

Base EVERYTHING on the actual document content. Do not invent data not present in the document.
Keep arrays to reasonable size (signals ≤30, conditions ≤15, states ≤12, transactions ≤6).
"""

SEQUENCE_SYSTEM = """You are a PlantUML expert. Generate a PlantUML sequence diagram
from the provided spec model JSON.

Rules:
- Use the actors from the spec model as participants
- Show ALL transactions from the spec model
- Use correct PlantUML syntax (@startuml ... @enduml)
- Add a title
- Use activate/deactivate for request-response pairs where appropriate
- Use == TxnName == group separators between transactions
- Add notes where useful for key protocol rules
- Return ONLY the raw PlantUML text, no markdown fences, no explanation
"""

FSM_SYSTEM = """You are a PlantUML expert. Generate a PlantUML state diagram (FSM)
from the provided spec model JSON.

Rules:
- Use the states from the spec model
- Use the state_transitions from the spec model
- Use correct PlantUML state diagram syntax (@startuml ... @enduml)
- Add a title
- Use [*] --> <initial_state> as the entry point
- Add state descriptions (state "Name" as X) where the name is long
- Add notes on states where useful
- Return ONLY the raw PlantUML text, no markdown fences, no explanation
"""


def run_groq(system_prompt: str, user_content: str, max_tokens: int = 2000) -> str:
    """Call Groq API and return the text response."""
    client = Groq(api_key=GROQ_API_KEY)
    chat_completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # fast, capable Groq model
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return chat_completion.choices[0].message.content.strip()


def render_plantuml_via_kroki(puml_text: str) -> str | None:
    """
    Encode PlantUML text using deflate+base64 (Kroki encoding),
    POST to Kroki and return the SVG string, or None on failure.
    """
    encoded = base64.urlsafe_b64encode(
        zlib.compress(puml_text.encode("utf-8"), 9)
    ).decode("ascii")
    url = f"{KROKI_URL}/{encoded}"
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code == 200:
            return resp.text
    except Exception:
        pass

    # fallback: POST
    try:
        resp = requests.post(
            "https://kroki.io/",
            json={"diagram_source": puml_text, "diagram_type": "plantuml", "output_format": "svg"},
            timeout=20,
        )
        if resp.status_code == 200:
            return resp.text
    except Exception:
        pass
    return None


# ── Demo data (used when no API key or no file) ───────────────────────────────
DEMO_MARKDOWN = """
# AXI4 Full Interconnect Protocol Specification v2.0

## 1. Introduction
This document specifies the AXI4 (Advanced eXtensible Interface) protocol
requirements for the SoC interconnect fabric.

## 2. Actors
- **AXI Master** (Manager): initiates all read and write transactions
- **AXI Slave** (Subordinate): responds to transactions from the master
- **Interconnect**: routes transactions between master and slave

## 3. Signal Definitions
| Signal  | Direction      | Description             |
|---------|----------------|-------------------------|
| ARVALID | Master→Slave   | Read address valid      |
| ARREADY | Slave→Master   | Read address ready      |
| ARADDR  | Master→Slave   | Read address            |
| ARID    | Master→Slave   | Read transaction ID     |
| AWVALID | Master→Slave   | Write address valid     |
| AWREADY | Slave→Master   | Write address ready     |
| WVALID  | Master→Slave   | Write data valid        |
| WREADY  | Slave→Master   | Write data ready        |
| WLAST   | Master→Slave   | Write last beat         |
| WSTRB   | Master→Slave   | Write byte strobe       |
| RVALID  | Slave→Master   | Read data valid         |
| RREADY  | Master→Slave   | Read data ready         |
| RLAST   | Slave→Master   | Read last beat          |
| BVALID  | Slave→Master   | Write response valid    |
| BREADY  | Master→Slave   | Write response ready    |
| BRESP   | Slave→Master   | Write response code     |

## 4. Protocol Rules — Shall Requirements
4.1 A source shall not make VALID dependent on the corresponding READY signal.
4.2 If ARVALID is asserted and ARREADY is high, the read address phase completes on the rising clock edge.
4.3 If AWVALID is asserted and AWREADY is high, the write address phase completes.
4.4 WLAST shall be asserted on the final data beat of a burst.
4.5 If WVALID is asserted and WREADY is high, the write data beat is transferred.
4.6 The master shall not change ARADDR while ARVALID is asserted.
4.7 If RVALID is asserted and RREADY is high, a read data beat is accepted.
4.8 BVALID shall be asserted only after the last write data beat.

## 5. Read Transaction Flow
1. Master drives ARVALID, ARADDR, ARID, ARLEN, ARSIZE, ARBURST
2. Slave asserts ARREADY — handshake completes
3. Slave drives RVALID, RDATA, RLAST, RID, RRESP beat by beat
4. Master asserts RREADY — each beat transferred
5. On RLAST=1 and RREADY=1, the read transaction is complete

## 6. Write Transaction Flow
1. Master drives AWVALID, AWADDR, AWID, AWLEN, AWSIZE, AWBURST
2. Slave asserts AWREADY — address handshake completes
3. Master drives WVALID, WDATA, WSTRB, WLAST for each beat
4. Slave asserts WREADY — each beat accepted
5. After WLAST, Slave drives BVALID, BRESP, BID
6. Master asserts BREADY — write response accepted, transaction complete

## 7. FSM States
The AXI slave interface operates the following state machine:
- IDLE: no transaction active
- AR_PENDING: waiting for ARREADY to be asserted
- R_ACTIVE: transferring read data beats
- AW_PENDING: waiting for AWREADY to be asserted
- W_ACTIVE: receiving write data beats
- B_PENDING: waiting for BREADY after write response issued

## 8. Register Map
| Offset | Name       | Access | Description        |
|--------|------------|--------|--------------------|
| 0x00   | CTRL_REG   | RW     | Control register   |
| 0x04   | STATUS_REG | RO     | Status register    |
| 0x08   | ADDR_BASE  | RW     | Base address       |
| 0x0C   | CFG_REG    | RW     | Configuration      |
"""


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='padding:.4rem 0 1rem'>
        <div style='font-family:IBM Plex Mono,monospace;font-size:.62rem;letter-spacing:.2em;color:#475569;text-transform:uppercase'>EFS Platform</div>
        <div style='font-size:1.3rem;font-weight:700;color:#e2e8f0;margin-top:3px'>Core Engine<br><span style='color:#38bdf8'>Dynamic Mode</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="section-header">API Keys (.env)</div>', unsafe_allow_html=True)
    llama_key = st.text_input("LlamaCloud API Key", value=LLAMA_CLOUD_API_KEY, type="password", help="LLAMA_CLOUD_API_KEY")
    if llama_key: LLAMA_CLOUD_API_KEY = llama_key
    groq_key = st.text_input("Groq API Key", value=GROQ_API_KEY, type="password", help="GROQ_API_KEY")
    if groq_key: GROQ_API_KEY = groq_key

    st.markdown("---")
    st.markdown('<div class="section-header">Pipeline status</div>', unsafe_allow_html=True)
    stages_status = {
        "LlamaParse ingestion": st.session_state.get("done_llama", False),
        "Spec Intelligence (LLM)": st.session_state.get("done_spec", False),
        "Sequence diagram (LLM)": st.session_state.get("done_seq", False),
        "FSM diagram (LLM)": st.session_state.get("done_fsm", False),
    }
    for name, done in stages_status.items():
        icon  = "✓" if done else "○"
        color = "#4ade80" if done else "#334155"
        st.markdown(f'<div style="display:flex;gap:8px;padding:3px 0;color:{color};font-family:IBM Plex Mono,monospace;font-size:.73rem"><span>{icon}</span><span>{name}</span></div>', unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("Install"):
        st.code("pip install llama-parse groq python-dotenv streamlit requests", language="bash")
        st.code("# .env\nLLAMA_CLOUD_API_KEY=llx-...\nGROQ_API_KEY=gsk_...", language="bash")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style='margin-bottom:1.4rem'>
    <div style='font-family:IBM Plex Mono,monospace;font-size:.62rem;letter-spacing:.2em;color:#475569;text-transform:uppercase;margin-bottom:3px'>Electronic Function Specification</div>
    <h1 style='margin:0;font-size:1.9rem;font-weight:700;color:#e2e8f0'>EFS Core Engine <span style='color:#38bdf8;font-size:1rem;font-weight:400;font-family:IBM Plex Mono,monospace'>— Dynamic</span></h1>
    <p style='color:#64748b;margin:3px 0 0;font-size:.88rem'>LlamaParse + Groq API + Kroki rendering</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-header">Upload & run</div>', unsafe_allow_html=True)

uploaded = st.file_uploader("Upload specification (PDF or DOCX)", type=["pdf", "docx"])
demo_mode = st.checkbox("Use built-in AXI4 demo (no LlamaCloud key needed)", value=not bool(LLAMA_CLOUD_API_KEY))

col_run, col_clear = st.columns([2, 1])
with col_run:
    run_btn = st.button("▶  Run full pipeline", use_container_width=True)
with col_clear:
    if st.button("✕  Clear", use_container_width=True):
        for k in ["done_llama","done_spec","done_seq","done_fsm",
                  "llama_json","spec_model","seq_puml","fsm_puml",
                  "seq_svg","fsm_svg","combined_text"]:
            st.session_state.pop(k, None)
        st.rerun()


if run_btn:
    if not demo_mode and not uploaded:
        st.error("Upload a document or enable demo mode.")
        st.stop()
    if not demo_mode and not LLAMA_CLOUD_API_KEY:
        st.error("LlamaCloud API key required for live parsing.")
        st.stop()
    if not GROQ_API_KEY:
        st.error("Groq API key required for dynamic diagram generation.")
        st.stop()

    # ── Step 1: Ingest ─────────────────────────────────────────────────────────
    with st.status("Step 1 — LlamaParse document ingestion", expanded=True) as s1:
        if demo_mode:
            pages = [{"page":1,"char_count":len(DEMO_MARKDOWN),"word_count":len(DEMO_MARKDOWN.split()),"text":DEMO_MARKDOWN,"metadata":{}}]
            st.write("Demo markdown loaded")
        else:
            file_bytes = uploaded.read()
            st.write(f"Parsing {uploaded.name} ({len(file_bytes):,} bytes) via LlamaParse…")
            pages = parse_with_llamaparse(file_bytes, uploaded.name)
            st.write(f"✓ {len(pages)} page(s) extracted")

        llama_json = llamaparse_to_json(pages)
        combined_text = "\n\n".join(p["text"] for p in pages)
        st.session_state["llama_json"] = llama_json
        st.session_state["combined_text"] = combined_text
        st.session_state["done_llama"] = True
        s1.update(label="✓ Ingestion complete", state="complete")

    # ── Step 2: Spec Intelligence via Groq ────────────────────────────────────
    with st.status("Step 2 — Spec Intelligence (Groq API)", expanded=True) as s2:
        st.write("Sending markdown to Groq for structured extraction…")
        truncated = combined_text[:12000]
        raw_spec = run_groq(
            SPEC_SYSTEM,
            f"Here is the specification document markdown:\n\n{truncated}",
            max_tokens=3000,
        )
        try:
            clean = re.sub(r"^```[a-z]*\n?|```$", "", raw_spec.strip(), flags=re.MULTILINE).strip()
            spec_model = json.loads(clean)
        except json.JSONDecodeError:
            st.error(f"Groq returned invalid JSON. Raw output:\n{raw_spec[:500]}")
            st.stop()

        st.session_state["spec_model"] = spec_model
        st.session_state["done_spec"] = True
        st.write(f"✓ Protocol: {spec_model.get('protocol','?')} | "
                 f"{len(spec_model.get('signals',[]))} signals | "
                 f"{len(spec_model.get('states',[]))} states | "
                 f"{len(spec_model.get('transactions',[]))} transactions")
        s2.update(label="✓ Spec Intelligence complete", state="complete")

    # ── Step 3: Sequence diagram ───────────────────────────────────────────────
    with st.status("Step 3 — Sequence diagram generation (Groq API)", expanded=True) as s3:
        st.write("Generating PlantUML sequence diagram from spec model…")
        seq_puml = run_groq(
            SEQUENCE_SYSTEM,
            f"Spec model JSON:\n{json.dumps(spec_model, indent=2)}",
            max_tokens=2000,
        )
        seq_puml = re.sub(r"^```[a-z]*\n?|```$", "", seq_puml.strip(), flags=re.MULTILINE).strip()
        if not seq_puml.startswith("@startuml"):
            seq_puml = "@startuml\n" + seq_puml
        if not seq_puml.endswith("@enduml"):
            seq_puml += "\n@enduml"

        st.write("Rendering via Kroki…")
        seq_svg = render_plantuml_via_kroki(seq_puml)
        st.session_state["seq_puml"] = seq_puml
        st.session_state["seq_svg"]  = seq_svg
        st.session_state["done_seq"] = True
        s3.update(label=f"✓ Sequence diagram {'rendered' if seq_svg else 'generated (render failed)'}", state="complete")

    # ── Step 4: FSM diagram ────────────────────────────────────────────────────
    with st.status("Step 4 — FSM state diagram (Groq API)", expanded=True) as s4:
        st.write("Generating PlantUML FSM from spec model…")
        fsm_puml = run_groq(
            FSM_SYSTEM,
            f"Spec model JSON:\n{json.dumps(spec_model, indent=2)}",
            max_tokens=2000,
        )
        fsm_puml = re.sub(r"^```[a-z]*\n?|```$", "", fsm_puml.strip(), flags=re.MULTILINE).strip()
        if not fsm_puml.startswith("@startuml"):
            fsm_puml = "@startuml\n" + fsm_puml
        if not fsm_puml.endswith("@enduml"):
            fsm_puml += "\n@enduml"

        st.write("Rendering via Kroki…")
        fsm_svg = render_plantuml_via_kroki(fsm_puml)
        st.session_state["fsm_puml"] = fsm_puml
        st.session_state["fsm_svg"]  = fsm_svg
        st.session_state["done_fsm"] = True
        s4.update(label=f"✓ FSM diagram {'rendered' if fsm_svg else 'generated (render failed)'}", state="complete")

    st.success("Pipeline complete.")
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.get("done_llama"):
    llama_json   = st.session_state["llama_json"]
    spec_model   = st.session_state.get("spec_model", {})
    seq_puml     = st.session_state.get("seq_puml", "")
    fsm_puml     = st.session_state.get("fsm_puml", "")
    seq_svg      = st.session_state.get("seq_svg")
    fsm_svg      = st.session_state.get("fsm_svg")

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 LlamaParse JSON",
        "🧠 Spec Model",
        "🔁 Sequence Diagram",
        "⚙️ FSM Diagram",
    ])

    # ── Tab 1: LlamaParse raw JSON ────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="stage-badge badge-blue">LlamaParse Output</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-box"><div class="val">{llama_json['total_pages']}</div><div class="lbl">Pages</div></div>
            <div class="metric-box"><div class="val">{llama_json['total_words']:,}</div><div class="lbl">Words</div></div>
            <div class="metric-box"><div class="val">{llama_json['total_chars']:,}</div><div class="lbl">Chars</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Full structured JSON</div>', unsafe_allow_html=True)
        st.json(llama_json)

        st.markdown('<div class="section-header">Download</div>', unsafe_allow_html=True)
        st.download_button(
            "⬇ Download LlamaParse JSON",
            data=json.dumps(llama_json, indent=2),
            file_name="llamaparse_output.json",
            mime="application/json",
        )

    # ── Tab 2: Spec model ─────────────────────────────────────────────────────
    with tab2:
        if not spec_model:
            st.info("Run pipeline to see spec model.")
        else:
            st.markdown('<div class="stage-badge badge-purple">Spec Intelligence — Groq API</div>', unsafe_allow_html=True)

            proto = spec_model.get("protocol", "?")
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-box"><div class="val">{proto}</div><div class="lbl">Protocol</div></div>
                <div class="metric-box"><div class="val">{len(spec_model.get('signals',[]))}</div><div class="lbl">Signals</div></div>
                <div class="metric-box"><div class="val">{len(spec_model.get('states',[]))}</div><div class="lbl">States</div></div>
                <div class="metric-box"><div class="val">{len(spec_model.get('transactions',[]))}</div><div class="lbl">Transactions</div></div>
                <div class="metric-box"><div class="val">{len(spec_model.get('conditions',[]))}</div><div class="lbl">Conditions</div></div>
            </div>
            """, unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown('<div class="section-header">Actors</div>', unsafe_allow_html=True)
                for a in spec_model.get("actors", []):
                    st.markdown(f'<div style="padding:5px 8px;background:#0d1117;border-left:3px solid #7c3aed;border-radius:0 6px 6px 0;margin-bottom:5px;font-size:.82rem;color:#c4b5fd">{a}</div>', unsafe_allow_html=True)

                st.markdown('<div class="section-header">Key rules</div>', unsafe_allow_html=True)
                for r in spec_model.get("key_rules", [])[:8]:
                    st.markdown(f'<div style="display:flex;gap:8px;padding:4px 0;border-bottom:1px solid #1e293b"><span style="color:#38bdf8;font-size:.8rem">▸</span><span style="font-size:.78rem;color:#cbd5e1">{r}</span></div>', unsafe_allow_html=True)

            with col_b:
                st.markdown('<div class="section-header">Conditions (normalized)</div>', unsafe_allow_html=True)
                for c in spec_model.get("conditions", [])[:8]:
                    st.markdown(f'<div style="margin-bottom:8px;padding:8px 10px;background:#0d1117;border-radius:6px;border-left:3px solid #1d4ed8"><div style="font-size:.7rem;color:#64748b">Raw:</div><div style="font-size:.76rem;color:#cbd5e1">{c.get("raw","")[:90]}</div><div style="font-size:.7rem;color:#64748b;margin-top:3px">Normalized:</div><div style="font-family:IBM Plex Mono,monospace;font-size:.73rem;color:#7dd3fc">{c.get("normalized","")[:90]}</div></div>', unsafe_allow_html=True)

                st.markdown('<div class="section-header">Transactions</div>', unsafe_allow_html=True)
                for txn in spec_model.get("transactions", []):
                    with st.expander(txn.get("name", "?")):
                        for i, step in enumerate(txn.get("steps", []), 1):
                            st.markdown(f'<div style="font-size:.78rem;color:#94a3b8;padding:2px 0"><span style="color:#475569">{i}.</span> {step}</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-header">Full spec model JSON</div>', unsafe_allow_html=True)
            st.json(spec_model)
            st.download_button("⬇ Download spec model JSON", data=json.dumps(spec_model, indent=2), file_name="spec_model.json", mime="application/json")

    # ── Tab 3: Sequence diagram ────────────────────────────────────────────────
    with tab3:
        if not seq_puml:
            st.info("Run pipeline to generate sequence diagram.")
        else:
            st.markdown('<div class="stage-badge badge-teal">Sequence Diagram — Groq API + Kroki</div>', unsafe_allow_html=True)

            col_p, col_s = st.columns([1, 1])
            with col_p:
                st.markdown('<div class="section-header">Generated PlantUML</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="puml-wrap">{seq_puml}</div>', unsafe_allow_html=True)
                st.download_button("⬇ Download .puml", data=seq_puml, file_name="sequence.puml", mime="text/plain")

            with col_s:
                st.markdown('<div class="section-header">Rendered diagram (Kroki)</div>', unsafe_allow_html=True)
                if seq_svg:
                    st.markdown(f'<div class="svg-wrap">{seq_svg}</div>', unsafe_allow_html=True)
                    st.download_button("⬇ Download SVG", data=seq_svg, file_name="sequence.svg", mime="image/svg+xml")
                else:
                    st.warning("Kroki rendering failed — copy the PlantUML text into plantuml.com/plantuml for offline rendering.")

    # ── Tab 4: FSM diagram ─────────────────────────────────────────────────────
    with tab4:
        if not fsm_puml:
            st.info("Run pipeline to generate FSM diagram.")
        else:
            st.markdown('<div class="stage-badge badge-amber">FSM State Diagram — Groq API + Kroki</div>', unsafe_allow_html=True)

            col_fp, col_fs = st.columns([1, 1])
            with col_fp:
                st.markdown('<div class="section-header">Generated PlantUML</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="puml-wrap">{fsm_puml}</div>', unsafe_allow_html=True)
                st.download_button("⬇ Download .puml", data=fsm_puml, file_name="fsm.puml", mime="text/plain")

            with col_fs:
                st.markdown('<div class="section-header">Rendered diagram (Kroki)</div>', unsafe_allow_html=True)
                if fsm_svg:
                    st.markdown(f'<div class="svg-wrap">{fsm_svg}</div>', unsafe_allow_html=True)
                    st.download_button("⬇ Download SVG", data=fsm_svg, file_name="fsm.svg", mime="image/svg+xml")
                else:
                    st.warning("Kroki rendering failed — copy the PlantUML text into plantuml.com/plantuml for offline rendering.")

else:
    st.markdown("""
    <div style='text-align:center;padding:3.5rem 2rem;background:linear-gradient(135deg,#0d1117,#111827);border:1px dashed #1e293b;border-radius:16px;margin-top:1.5rem'>
        <div style='font-size:2.5rem;margin-bottom:.8rem'>⚡</div>
        <div style='font-size:1.15rem;font-weight:600;color:#e2e8f0;margin-bottom:.4rem'>Ready to process your specification</div>
        <div style='color:#475569;font-size:.85rem;max-width:480px;margin:0 auto 1.5rem'>
            Upload a PDF/DOCX and click Run — or enable Demo Mode.<br>
            Groq reads the document and dynamically generates real diagrams.
        </div>
        <div style='display:flex;justify-content:center;gap:1.5rem;flex-wrap:wrap'>
            <div style='text-align:center'><div style='font-family:IBM Plex Mono,monospace;font-size:.65rem;color:#38bdf8;letter-spacing:.1em'>STEP 01</div><div style='color:#64748b;font-size:.78rem'>LlamaParse → JSON</div></div>
            <div style='color:#1e293b;align-self:center'>→</div>
            <div style='text-align:center'><div style='font-family:IBM Plex Mono,monospace;font-size:.65rem;color:#c4b5fd;letter-spacing:.1em'>STEP 02</div><div style='color:#64748b;font-size:.78rem'>Groq → Spec model</div></div>
            <div style='color:#1e293b;align-self:center'>→</div>
            <div style='text-align:center'><div style='font-family:IBM Plex Mono,monospace;font-size:.65rem;color:#5eead4;letter-spacing:.1em'>STEP 03</div><div style='color:#64748b;font-size:.78rem'>Groq → Sequence UML</div></div>
            <div style='color:#1e293b;align-self:center'>→</div>
            <div style='text-align:center'><div style='font-family:IBM Plex Mono,monospace;font-size:.65rem;color:#fcd34d;letter-spacing:.1em'>STEP 04</div><div style='color:#64748b;font-size:.78rem'>Groq → FSM + Kroki SVG</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
