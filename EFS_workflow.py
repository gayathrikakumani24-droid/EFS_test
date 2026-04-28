"""
EFS Core Engine — Streamlit App
Covers: Spec Intelligence → Protocol Intelligence → Flow Generator
Uses LlamaParse (via dotenv) for document ingestion
"""

import os
import json
import re
import tempfile
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# ── Load secrets ──────────────────────────────────────────────────────────────
load_dotenv()
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")       # optional — used for LLM enrichment
ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY")    # optional — alternative LLM

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EFS Core Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Space+Grotesk:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* Dark industrial theme */
.stApp { background: #0d0f14; color: #e2e8f0; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111318 !important;
    border-right: 1px solid #1e2330;
}

/* Stage cards */
.stage-card {
    background: linear-gradient(135deg, #13161f 0%, #1a1e2e 100%);
    border: 1px solid #252a3d;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.stage-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent);
}
.stage-card.active::before { background: #38bdf8; }
.stage-card.done::before   { background: #4ade80; }
.stage-card.idle::before   { background: #334155; }

.stage-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.stage-title.active { color: #38bdf8; }
.stage-title.done   { color: #4ade80; }
.stage-title.idle   { color: #475569; }

.badge {
    display: inline-block;
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.06em;
}
.badge-blue   { background: #1e3a5f; color: #7dd3fc; }
.badge-green  { background: #14342b; color: #6ee7b7; }
.badge-amber  { background: #3d2c0e; color: #fcd34d; }
.badge-purple { background: #2d1f5e; color: #c4b5fd; }
.badge-slate  { background: #1e293b; color: #94a3b8; }

.signal-chip {
    display: inline-block;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #cbd5e1;
    margin: 2px;
}

.plantuml-block {
    background: #0a0c10;
    border: 1px solid #1e2330;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    line-height: 1.7;
    color: #a5f3fc;
    overflow-x: auto;
    white-space: pre;
}

.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #475569;
    margin: 1.4rem 0 0.6rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #1e293b;
}

.metric-row {
    display: flex; gap: 12px; flex-wrap: wrap; margin: 0.6rem 0;
}
.metric-box {
    background: #13161f;
    border: 1px solid #1e2330;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    text-align: center;
    min-width: 80px;
}
.metric-box .val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: #38bdf8;
}
.metric-box .lbl {
    font-size: 0.65rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

div[data-testid="stExpander"] > div:first-child {
    background: #13161f !important;
    border: 1px solid #1e2330 !important;
    border-radius: 8px !important;
}

.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(59,130,246,0.35) !important;
}

.stFileUploader > div { background: #13161f !important; border: 1px dashed #334155 !important; border-radius: 10px !important; }

hr { border-color: #1e2330 !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def check_dependencies():
    """Verify required packages are installed."""
    missing = []
    try:
        import llama_parse  # noqa
    except ImportError:
        missing.append("llama-parse")
    try:
        from dotenv import load_dotenv  # noqa
    except ImportError:
        missing.append("python-dotenv")
    return missing


def parse_document_with_llamaparse(file_bytes: bytes, filename: str) -> str:
    """
    Use LlamaParse to extract structured markdown from a PDF/DOCX.
    Returns the concatenated markdown string.
    """
    from llama_parse import LlamaParse

    parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="markdown",
        verbose=False,
        language="en",
        parsing_instruction=(
            "Extract all text including tables, figures captions, section headings, "
            "register maps, signal names, protocol conditions, and timing rules. "
            "Preserve hierarchy with markdown headings."
        ),
    )

    suffix = Path(filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        documents = parser.load_data(tmp_path)
        combined = "\n\n".join(doc.text for doc in documents)
        return combined
    finally:
        os.unlink(tmp_path)


# ── Spec Intelligence helpers ──────────────────────────────────────────────────

SECTION_RE = re.compile(r'^(#{1,4})\s+(.+)', re.MULTILINE)
SIGNAL_RE  = re.compile(
    r'\b([A-Z][A-Z0-9_]{2,}(?:VALID|READY|DATA|ADDR|EN|SEL|REQ|ACK|CLK|RST|WE|RE|STRB|RESP|PROT|ID|LEN|SIZE|BURST|LOCK|CACHE|QOS|REGION|USER|LAST)?)\b'
)
CONDITION_RE = re.compile(
    r'[Ii]f\s+(.{10,120}?)(?:,|\.|then|is|are|assert)', re.DOTALL
)
REGISTER_RE = re.compile(
    r'\b([A-Z][A-Z0-9_]*(?:_REG|_CTRL|_STATUS|_CFG|_ADDR|_DATA|_BASE|_OFFSET)?)\b'
)

PROTOCOL_KEYWORDS = {
    "AXI":   ["ARVALID","ARREADY","AWVALID","AWREADY","WVALID","WREADY",
               "RVALID","RREADY","BVALID","BREADY","ARID","AWID","WID","RID","BID"],
    "PCIe":  ["TLP","DLLP","completion","requester","completer","BAR","MRd","MWr",
               "CplD","FC","LTSSM","L0s","L1"],
    "CXL":   ["CXL.io","CXL.cache","CXL.mem","HDM","DVSEC","DOE","snoop","bias"],
    "CHI":   ["REQ","RSP","DAT","SNP","flit","link layer","transaction layer",
               "home node","request node","slave node"],
    "AHB":   ["HADDR","HTRANS","HSIZE","HBURST","HPROT","HWDATA","HRDATA",
               "HREADY","HRESP","HMASTER","HSEL"],
    "APB":   ["PADDR","PSEL","PENABLE","PWRITE","PWDATA","PRDATA","PREADY","PSLVERR"],
}


def run_spec_intelligence(markdown_text: str) -> dict:
    """
    Parse markdown from LlamaParse into structured spec model.
    Returns dict with sections, signals, conditions, registers, summary stats.
    """
    sections = []
    for m in SECTION_RE.finditer(markdown_text):
        level = len(m.group(1))
        title = m.group(2).strip()
        # classify normative vs informative
        normative = any(kw in title.lower() for kw in
                        ["shall","must","requirement","mandatory","compliance"])
        sections.append({"level": level, "title": title, "normative": normative})

    # Signal extraction (deduplicated, sorted)
    raw_signals = SIGNAL_RE.findall(markdown_text)
    signals = sorted(set(s for s in raw_signals if len(s) >= 4))

    # Condition extraction
    conditions = []
    for m in CONDITION_RE.finditer(markdown_text):
        cond_text = m.group(1).strip().replace("\n", " ")
        if len(cond_text) > 10:
            # Naive normalization
            normalized = cond_text
            normalized = re.sub(r'is asserted|is high|== 1|= 1', '== 1', normalized)
            normalized = re.sub(r'is deasserted|is low|== 0|= 0', '== 0', normalized)
            conditions.append({
                "raw": cond_text[:120],
                "normalized": normalized[:120],
            })
    conditions = conditions[:30]  # cap

    # Register extraction
    raw_regs = REGISTER_RE.findall(markdown_text)
    registers = sorted(set(r for r in raw_regs
                           if len(r) >= 5
                           and any(suffix in r for suffix in
                                   ["_REG","_CTRL","_STATUS","_CFG","_ADDR",
                                    "_DATA","_BASE","_OFFSET"])))

    return {
        "sections": sections,
        "signals": signals[:60],
        "conditions": conditions,
        "registers": registers[:30],
        "word_count": len(markdown_text.split()),
        "char_count": len(markdown_text),
    }


def run_protocol_intelligence(spec_model: dict, markdown_text: str) -> dict:
    """
    Classify protocol, map transactions, assign actor roles.
    Returns protocol dict.
    """
    signals_set = set(spec_model["signals"])
    text_upper  = markdown_text.upper()

    # Score each protocol
    scores = {}
    for proto, keywords in PROTOCOL_KEYWORDS.items():
        hit = sum(1 for kw in keywords if kw.upper() in text_upper)
        scores[proto] = hit

    detected = [p for p, s in sorted(scores.items(), key=lambda x: -x[1]) if s > 0]
    primary  = detected[0] if detected else "UNKNOWN"

    # Transaction templates per protocol
    TRANSACTION_TEMPLATES = {
        "AXI": {
            "Read":  ["AR channel: ARVALID/ARREADY handshake",
                      "R channel: RVALID/RREADY data transfer",
                      "RLAST asserted on final beat"],
            "Write": ["AW channel: AWVALID/AWREADY handshake",
                      "W channel: WVALID/WREADY data + WSTRB",
                      "WLAST on final beat",
                      "B channel: BVALID/BREADY response"],
        },
        "PCIe": {
            "Memory Read":  ["Requester sends MRd TLP","Completer returns CplD TLP","FC credits updated"],
            "Memory Write": ["Requester sends MWr TLP","Posted — no completion required","FC credits updated"],
        },
        "CXL": {
            "CXL.mem Read":  ["Host sends MemRd","Device returns MemData","HDM decode"],
            "CXL.cache Snoop": ["Home sends SNP","Device responds SnpResp","Cache state transition"],
        },
        "AHB": {
            "Transfer": ["Master drives HTRANS/HADDR","Slave samples on HCLK","HREADY extended wait","HRDATA driven"],
        },
        "APB": {
            "Write": ["PSEL asserted","PENABLE low → high","PWRITE + PWDATA stable","PREADY"],
            "Read":  ["PSEL asserted","PENABLE low → high","PWRITE low","PRDATA captured"],
        },
    }

    transactions = TRANSACTION_TEMPLATES.get(primary, {
        "Generic Transaction": ["Request issued","Acknowledge received","Data exchanged","Response returned"]
    })

    # Actor-role assignment
    actor_roles = {
        "AXI":  {"Manager": "AXI Master (initiates AR/AW)", "Subordinate": "AXI Slave (responds R/B)"},
        "PCIe": {"Requester": "PCIe Root/EP (sends TLP)", "Completer": "Target EP/RC (returns Cpl)"},
        "CXL":  {"Host": "CPU Complex", "Device": "CXL Device (Type 1/2/3)"},
        "CHI":  {"Request Node": "CPU / DMA", "Home Node": "Interconnect", "Slave Node": "Memory"},
        "AHB":  {"Master": "Bus Master (DMA/CPU)", "Slave": "Peripheral/Memory"},
        "APB":  {"Master": "APB Bridge", "Slave": "APB Peripheral"},
    }.get(primary, {"Initiator": "Source component", "Target": "Destination component"})

    # Handshake rules
    handshake_rules = {
        "AXI":  ["VALID may not depend on READY","READY may depend on VALID",
                 "Handshake completes when both VALID and READY are HIGH on rising edge"],
        "PCIe": ["Ordering rules: Posted > Non-Posted > Completion",
                 "Flow control credits must be available before TLP transmission"],
        "AHB":  ["Transfer starts when HTRANS != IDLE","Slave extends by deasserting HREADY"],
        "APB":  ["Two-phase: SETUP then ENABLE","PSEL must remain asserted throughout"],
    }.get(primary, ["Request acknowledged before data transfer",
                     "Response must match outstanding transaction ID"])

    return {
        "primary_protocol": primary,
        "detected_protocols": detected,
        "scores": scores,
        "transactions": transactions,
        "actor_roles": actor_roles,
        "handshake_rules": handshake_rules,
    }


def run_flow_generator(spec_model: dict, proto_model: dict) -> dict:
    """
    Generate PlantUML sequence + state diagrams from protocol model.
    """
    primary  = proto_model["primary_protocol"]
    actors   = proto_model["actor_roles"]
    txns     = proto_model["transactions"]

    actor_names = list(actors.keys())
    a1 = actor_names[0] if len(actor_names) > 0 else "Initiator"
    a2 = actor_names[1] if len(actor_names) > 1 else "Target"

    # ── Sequence diagram ──────────────────────────────────────────────────────
    seq_lines = ["@startuml", "skinparam monochrome true",
                 "skinparam sequenceMessageAlign center",
                 f'participant "{a1}" as A',
                 f'participant "{a2}" as B', ""]

    for txn_name, steps in txns.items():
        seq_lines.append(f"== {txn_name} ==")
        for i, step in enumerate(steps):
            if "→" in step or "return" in step.lower() or "response" in step.lower() or "data" in step.lower():
                seq_lines.append(f'B --> A: {step}')
            elif i % 2 == 0:
                seq_lines.append(f'A -> B: {step}')
            else:
                seq_lines.append(f'B -> A: {step}')
        seq_lines.append("")

    seq_lines.append("@enduml")
    sequence_uml = "\n".join(seq_lines)

    # ── State diagram ─────────────────────────────────────────────────────────
    STATE_MACHINES = {
        "AXI": {
            "states": ["IDLE","AR_PENDING","R_TRANSFER","AW_PENDING",
                       "W_TRANSFER","B_PENDING"],
            "transitions": [
                ("IDLE","AR_PENDING","ARVALID"),
                ("AR_PENDING","R_TRANSFER","ARREADY"),
                ("R_TRANSFER","IDLE","RLAST && RREADY"),
                ("IDLE","AW_PENDING","AWVALID"),
                ("AW_PENDING","W_TRANSFER","AWREADY"),
                ("W_TRANSFER","B_PENDING","WLAST && WREADY"),
                ("B_PENDING","IDLE","BREADY"),
            ]
        },
        "PCIe": {
            "states": ["IDLE","TLP_BUILD","TLP_SEND","AWAIT_CPL","CPL_RECV","ERROR"],
            "transitions": [
                ("IDLE","TLP_BUILD","request_issued"),
                ("TLP_BUILD","TLP_SEND","FC_credits_ok"),
                ("TLP_SEND","AWAIT_CPL","MRd sent"),
                ("AWAIT_CPL","CPL_RECV","CplD received"),
                ("CPL_RECV","IDLE","transfer_done"),
                ("AWAIT_CPL","ERROR","timeout"),
                ("ERROR","IDLE","recovery"),
            ]
        },
        "APB": {
            "states": ["IDLE","SETUP","ENABLE","DONE"],
            "transitions": [
                ("IDLE","SETUP","PSEL"),
                ("SETUP","ENABLE","PENABLE"),
                ("ENABLE","DONE","PREADY"),
                ("DONE","IDLE","xfer_complete"),
            ]
        },
    }

    sm = STATE_MACHINES.get(primary, {
        "states": ["IDLE","REQUEST","TRANSFER","RESPONSE","DONE"],
        "transitions": [
            ("IDLE","REQUEST","req_valid"),
            ("REQUEST","TRANSFER","req_ack"),
            ("TRANSFER","RESPONSE","data_done"),
            ("RESPONSE","DONE","resp_ok"),
            ("DONE","IDLE","reset"),
        ]
    })

    fsm_lines = ["@startuml", "skinparam monochrome true",
                 "[*] --> IDLE", ""]
    for (src, dst, label) in sm["transitions"]:
        fsm_lines.append(f'{src} --> {dst} : {label}')
    fsm_lines += ["", "@enduml"]
    state_uml = "\n".join(fsm_lines)

    # ── Condition logic ───────────────────────────────────────────────────────
    conditions = spec_model.get("conditions", [])
    logic_table = []
    for c in conditions[:10]:
        logic_table.append({
            "condition": c["raw"][:80],
            "normalized": c["normalized"][:80],
        })

    return {
        "sequence_uml": sequence_uml,
        "state_uml":    state_uml,
        "transactions":  list(txns.keys()),
        "fsm_states":    sm["states"],
        "logic_table":   logic_table,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='padding:0.6rem 0 1rem 0'>
        <div style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;
                    letter-spacing:0.2em;color:#475569;text-transform:uppercase'>
            EFS Platform
        </div>
        <div style='font-size:1.4rem;font-weight:700;color:#e2e8f0;
                    line-height:1.2;margin-top:4px'>
            Core Engine<br>
            <span style='color:#38bdf8'>Spec → Flow</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div class='section-header'>Pipeline Stages</div>
    """, unsafe_allow_html=True)

    # Dynamic stage status from session_state
    stage_status = {
        "01 Spec Intelligence":    st.session_state.get("spec_done", False),
        "02 Protocol Intelligence":st.session_state.get("proto_done", False),
        "03 Flow Generator":       st.session_state.get("flow_done", False),
    }
    for name, done in stage_status.items():
        status_cls = "done" if done else "idle"
        icon = "✓" if done else "○"
        color = "#4ade80" if done else "#334155"
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:8px;
                    padding:0.35rem 0;color:{color};
                    font-family:IBM Plex Mono,monospace;font-size:0.75rem'>
            <span>{icon}</span><span>{name}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div class="section-header">Configuration</div>', unsafe_allow_html=True)

    api_key_input = st.text_input(
        "LlamaCloud API Key",
        value=LLAMA_CLOUD_API_KEY or "",
        type="password",
        help="Set LLAMA_CLOUD_API_KEY in .env or enter here",
    )
    if api_key_input:
        LLAMA_CLOUD_API_KEY = api_key_input

    result_type = st.selectbox("LlamaParse Output", ["markdown", "text"], index=0)

    st.markdown("---")
    missing = check_dependencies()
    if missing:
        st.warning(f"Missing packages: `{', '.join(missing)}`\n\n"
                   f"Install: `pip install {' '.join(missing)}`")
    else:
        st.success("All dependencies installed ✓")

    with st.expander("Install commands"):
        st.code("pip install llama-parse python-dotenv streamlit", language="bash")
        st.code("""# .env file
LLAMA_CLOUD_API_KEY=llx-...
OPENAI_API_KEY=sk-...      # optional
ANTHROPIC_API_KEY=sk-ant-... # optional
""", language="bash")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style='margin-bottom:1.6rem'>
    <div style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;
                letter-spacing:0.2em;color:#475569;text-transform:uppercase;
                margin-bottom:4px'>Electronic Function Specification</div>
    <h1 style='margin:0;font-size:2rem;font-weight:700;color:#e2e8f0'>
        EFS Core Engine
        <span style='color:#38bdf8;font-size:1.1rem;font-weight:400;
                     margin-left:12px;font-family:IBM Plex Mono,monospace'>
            v1.0
        </span>
    </h1>
    <p style='color:#64748b;margin:4px 0 0 0;font-size:0.9rem'>
        Spec Intelligence · Protocol Intelligence · Flow Generator
    </p>
</div>
""", unsafe_allow_html=True)

# ── Upload zone ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Document Ingestion via LlamaParse</div>',
            unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload specification (PDF or DOCX)",
    type=["pdf", "docx"],
    help="LlamaParse will extract structured markdown for all three pipeline stages.",
)

demo_mode = st.checkbox(
    "Use built-in demo (AXI4 snippet — no API key needed)",
    value=not bool(LLAMA_CLOUD_API_KEY),
)

DEMO_MARKDOWN = """
# AXI4 Full Interconnect Protocol Specification

## 1. Introduction
This document specifies the AXI4 (Advanced eXtensible Interface 4) protocol requirements
for all IP cores in the SoC interconnect fabric.

## 2. Normative Signal Definitions
The following signals are mandatory for all AXI4-compliant interfaces:

| Signal     | Direction | Width | Description            |
|------------|-----------|-------|------------------------|
| ARVALID    | Master→Slave | 1  | Read address valid     |
| ARREADY    | Slave→Master | 1  | Read address ready     |
| ARADDR     | Master→Slave | 32 | Read address           |
| ARID       | Master→Slave | 4  | Read transaction ID    |
| AWVALID    | Master→Slave | 1  | Write address valid    |
| AWREADY    | Slave→Master | 1  | Write address ready    |
| WVALID     | Master→Slave | 1  | Write data valid       |
| WREADY     | Slave→Master | 1  | Write data ready       |
| WSTRB      | Master→Slave | 4  | Write byte strobe      |
| WLAST      | Master→Slave | 1  | Write last beat        |
| RVALID     | Slave→Master | 1  | Read data valid        |
| RREADY     | Master→Slave | 1  | Read data ready        |
| BVALID     | Slave→Master | 1  | Write response valid   |
| BREADY     | Master→Slave | 1  | Write response ready   |
| BRESP      | Slave→Master | 2  | Write response code    |

## 3. Handshake Rules — Shall Requirements
3.1 A source shall not make VALID dependent on the corresponding READY signal.
3.2 If ARVALID is asserted and ARREADY is high, the read address phase completes.
3.3 If AWVALID is asserted and AWREADY is high, the write address phase completes.
3.4 WLAST shall be asserted on the final data beat of a burst.
3.5 The master shall not change ARADDR while ARVALID is asserted.

## 4. Ordering Rules
4.1 Transactions with the same ID shall complete in order.
4.2 The slave shall return RDATA in the order read addresses were received.

## 5. Register Map (AXI Slave Control Registers)
| Offset | Name         | Width | Access | Description          |
|--------|--------------|-------|--------|----------------------|
| 0x00   | CTRL_REG     | 32    | RW     | Control register     |
| 0x04   | STATUS_REG   | 32    | RO     | Status register      |
| 0x08   | ADDR_BASE    | 32    | RW     | Base address         |
| 0x0C   | CFG_REG      | 32    | RW     | Configuration        |
| 0x10   | DATA_REG     | 32    | RW     | Data register        |
"""

run_btn = st.button("▶  Run EFS Pipeline", use_container_width=False)

if run_btn:
    # ── Validate ──────────────────────────────────────────────────────────────
    if not demo_mode and not uploaded:
        st.error("Please upload a document or enable demo mode.")
        st.stop()
    if not demo_mode and not LLAMA_CLOUD_API_KEY:
        st.error("LlamaCloud API key is required. Set LLAMA_CLOUD_API_KEY in .env or the sidebar.")
        st.stop()

    # ── Stage 1: LlamaParse ingestion ─────────────────────────────────────────
    with st.status("🔍 Stage 1 — LlamaParse Document Ingestion", expanded=True) as status1:
        st.write("Parsing document with LlamaParse…")
        try:
            if demo_mode:
                markdown_text = DEMO_MARKDOWN
                st.write("✓ Demo markdown loaded (AXI4 spec snippet)")
            else:
                file_bytes = uploaded.read()
                st.write(f"File size: {len(file_bytes):,} bytes — uploading to LlamaParse…")
                markdown_text = parse_document_with_llamaparse(file_bytes, uploaded.name)
                st.write(f"✓ LlamaParse returned {len(markdown_text):,} characters")
            st.session_state["markdown_text"] = markdown_text
            status1.update(label="✓ Document ingested", state="complete")
        except Exception as e:
            status1.update(label="✗ Ingestion failed", state="error")
            st.error(f"LlamaParse error: {e}")
            st.stop()

    # ── Stage 2: Spec Intelligence ────────────────────────────────────────────
    with st.status("🧠 Stage 2 — Spec Intelligence", expanded=True) as status2:
        st.write("Extracting signals, conditions, registers, sections…")
        spec_model = run_spec_intelligence(markdown_text)
        st.session_state["spec_model"] = spec_model
        st.session_state["spec_done"]  = True
        st.write(f"✓ {len(spec_model['sections'])} sections  "
                 f"| {len(spec_model['signals'])} signals  "
                 f"| {len(spec_model['conditions'])} conditions  "
                 f"| {len(spec_model['registers'])} registers")
        status2.update(label="✓ Spec Intelligence complete", state="complete")

    # ── Stage 3: Protocol Intelligence ───────────────────────────────────────
    with st.status("⚙️ Stage 3 — Protocol Intelligence", expanded=True) as status3:
        st.write("Classifying protocol, mapping transactions, assigning roles…")
        proto_model = run_protocol_intelligence(spec_model, markdown_text)
        st.session_state["proto_model"] = proto_model
        st.session_state["proto_done"]  = True
        st.write(f"✓ Primary protocol: {proto_model['primary_protocol']}  "
                 f"| {len(proto_model['transactions'])} transaction types")
        status3.update(label="✓ Protocol Intelligence complete", state="complete")

    # ── Stage 4: Flow Generator ────────────────────────────────────────────────
    with st.status("🔁 Stage 4 — Flow Generator", expanded=True) as status4:
        st.write("Generating PlantUML sequence & state diagrams…")
        flow_model = run_flow_generator(spec_model, proto_model)
        st.session_state["flow_model"] = flow_model
        st.session_state["flow_done"]  = True
        st.write(f"✓ {len(flow_model['transactions'])} flows  "
                 f"| {len(flow_model['fsm_states'])} FSM states  "
                 f"| {len(flow_model['logic_table'])} condition rules")
        status4.update(label="✓ Flow Generator complete", state="complete")

    st.success("🎉 EFS Pipeline complete — all three stages finished successfully.")
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.get("spec_done"):
    spec_model  = st.session_state["spec_model"]
    proto_model = st.session_state.get("proto_model", {})
    flow_model  = st.session_state.get("flow_model", {})
    markdown_text = st.session_state.get("markdown_text", "")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🧠 Spec Intelligence",
        "⚙️ Protocol Intelligence",
        "🔁 Flow Generator",
        "📄 Raw Markdown",
    ])

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 1 — Spec Intelligence
    # ──────────────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="stage-card done">'
                    '<div class="stage-title done">STAGE 01 — SPEC INTELLIGENCE</div>'
                    '<span class="badge badge-green">COMPLETE</span>'
                    '</div>', unsafe_allow_html=True)

        # Metrics row
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-box"><div class="val">{len(spec_model['sections'])}</div><div class="lbl">Sections</div></div>
            <div class="metric-box"><div class="val">{len(spec_model['signals'])}</div><div class="lbl">Signals</div></div>
            <div class="metric-box"><div class="val">{len(spec_model['conditions'])}</div><div class="lbl">Conditions</div></div>
            <div class="metric-box"><div class="val">{len(spec_model['registers'])}</div><div class="lbl">Registers</div></div>
            <div class="metric-box"><div class="val">{spec_model['word_count']:,}</div><div class="lbl">Words</div></div>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            # Section hierarchy
            st.markdown('<div class="section-header">Section Hierarchy</div>',
                        unsafe_allow_html=True)
            for sec in spec_model["sections"][:25]:
                indent = "&nbsp;" * ((sec["level"] - 1) * 4)
                badge  = '<span class="badge badge-amber">NORMATIVE</span>' \
                         if sec["normative"] else ''
                st.markdown(
                    f'<div style="font-size:0.82rem;color:#cbd5e1;'
                    f'padding:3px 0;border-left:2px solid #1e2330;padding-left:8px">'
                    f'{indent}<span style="color:#64748b">{"#"*sec["level"]}</span> '
                    f'{sec["title"]} {badge}</div>',
                    unsafe_allow_html=True
                )

            # Registers
            if spec_model["registers"]:
                st.markdown('<div class="section-header">Register Map Fragments</div>',
                            unsafe_allow_html=True)
                reg_html = "".join(
                    f'<span class="signal-chip">{r}</span>'
                    for r in spec_model["registers"]
                )
                st.markdown(reg_html, unsafe_allow_html=True)

        with col_b:
            # Signals
            st.markdown('<div class="section-header">Signal Catalog</div>',
                        unsafe_allow_html=True)
            sig_html = "".join(
                f'<span class="signal-chip">{s}</span>'
                for s in spec_model["signals"]
            )
            st.markdown(sig_html, unsafe_allow_html=True)

            # Conditions
            if spec_model["conditions"]:
                st.markdown('<div class="section-header">Condition Rules (Normalized)</div>',
                            unsafe_allow_html=True)
                for i, c in enumerate(spec_model["conditions"][:8]):
                    st.markdown(
                        f'<div style="margin-bottom:8px;padding:8px 10px;'
                        f'background:#0d1117;border-radius:6px;'
                        f'border-left:3px solid #1d4ed8">'
                        f'<div style="font-size:0.7rem;color:#64748b;margin-bottom:2px">Raw:</div>'
                        f'<div style="font-size:0.78rem;color:#cbd5e1">{c["raw"]}</div>'
                        f'<div style="font-size:0.7rem;color:#64748b;margin-top:4px">Normalized:</div>'
                        f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:#7dd3fc">'
                        f'{c["normalized"]}</div></div>',
                        unsafe_allow_html=True
                    )

        # Full JSON export
        with st.expander("Export: Spec Model JSON"):
            st.json(spec_model)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 2 — Protocol Intelligence
    # ──────────────────────────────────────────────────────────────────────────
    with tab2:
        if not proto_model:
            st.info("Run the pipeline to see Protocol Intelligence results.")
        else:
            st.markdown('<div class="stage-card done">'
                        '<div class="stage-title done">STAGE 02 — PROTOCOL INTELLIGENCE</div>'
                        '<span class="badge badge-green">COMPLETE</span>'
                        '</div>', unsafe_allow_html=True)

            # Protocol classification
            st.markdown('<div class="section-header">Protocol Classification</div>',
                        unsafe_allow_html=True)

            scores = proto_model["scores"]
            sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
            score_html = ""
            for proto, score in sorted_scores:
                if score > 0:
                    pct = min(score * 12, 100)
                    is_primary = proto == proto_model["primary_protocol"]
                    badge_cls  = "badge-blue" if is_primary else "badge-slate"
                    score_html += f"""
                    <div style='margin-bottom:8px'>
                        <div style='display:flex;justify-content:space-between;
                                    align-items:center;margin-bottom:3px'>
                            <span style='font-family:IBM Plex Mono,monospace;
                                         font-size:0.82rem;color:#e2e8f0'>{proto}</span>
                            <span class='badge {badge_cls}'>
                                {'PRIMARY' if is_primary else f'{score} kw'}
                            </span>
                        </div>
                        <div style='height:4px;background:#1e293b;border-radius:2px'>
                            <div style='height:4px;width:{pct}%;
                                        background:{"#38bdf8" if is_primary else "#475569"};
                                        border-radius:2px;transition:width 0.4s'></div>
                        </div>
                    </div>
                    """
            st.markdown(score_html, unsafe_allow_html=True)

            col_c, col_d = st.columns(2)

            with col_c:
                # Actor roles
                st.markdown('<div class="section-header">Actor-Role Mapping</div>',
                            unsafe_allow_html=True)
                for role, desc in proto_model["actor_roles"].items():
                    st.markdown(
                        f'<div style="padding:8px 10px;background:#0d1117;'
                        f'border-radius:6px;border-left:3px solid #7c3aed;margin-bottom:6px">'
                        f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.8rem;'
                        f'color:#c4b5fd">{role}</div>'
                        f'<div style="font-size:0.78rem;color:#94a3b8;margin-top:2px">{desc}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                # Handshake rules
                st.markdown('<div class="section-header">Protocol Rules</div>',
                            unsafe_allow_html=True)
                for rule in proto_model["handshake_rules"]:
                    st.markdown(
                        f'<div style="display:flex;gap:8px;padding:5px 0;'
                        f'border-bottom:1px solid #1e293b">'
                        f'<span style="color:#38bdf8;font-size:0.8rem">▸</span>'
                        f'<span style="font-size:0.8rem;color:#cbd5e1">{rule}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            with col_d:
                # Transactions
                st.markdown('<div class="section-header">Transaction Recognition</div>',
                            unsafe_allow_html=True)
                for txn_name, steps in proto_model["transactions"].items():
                    st.markdown(
                        f'<div style="margin-bottom:10px;padding:10px 12px;'
                        f'background:#0d1117;border-radius:8px;'
                        f'border:1px solid #1e2330">'
                        f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.78rem;'
                        f'color:#fbbf24;margin-bottom:6px">{txn_name}</div>',
                        unsafe_allow_html=True
                    )
                    for i, step in enumerate(steps):
                        st.markdown(
                            f'<div style="font-size:0.76rem;color:#94a3b8;padding:2px 0;'
                            f'padding-left:12px">'
                            f'<span style="color:#334155">{i+1}.</span> {step}</div>',
                            unsafe_allow_html=True
                        )
                    st.markdown('</div>', unsafe_allow_html=True)

            with st.expander("Export: Protocol Model JSON"):
                st.json(proto_model)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 3 — Flow Generator
    # ──────────────────────────────────────────────────────────────────────────
    with tab3:
        if not flow_model:
            st.info("Run the pipeline to see Flow Generator results.")
        else:
            st.markdown('<div class="stage-card done">'
                        '<div class="stage-title done">STAGE 03 — FLOW GENERATOR</div>'
                        '<span class="badge badge-green">COMPLETE</span>'
                        '</div>', unsafe_allow_html=True)

            col_e, col_f = st.columns(2)

            with col_e:
                st.markdown('<div class="section-header">PlantUML — Sequence Diagram</div>',
                            unsafe_allow_html=True)
                st.markdown(
                    f'<div class="plantuml-block">{flow_model["sequence_uml"]}</div>',
                    unsafe_allow_html=True
                )

            with col_f:
                st.markdown('<div class="section-header">PlantUML — FSM State Diagram</div>',
                            unsafe_allow_html=True)
                st.markdown(
                    f'<div class="plantuml-block">{flow_model["state_uml"]}</div>',
                    unsafe_allow_html=True
                )

            # FSM states
            st.markdown('<div class="section-header">FSM State Inventory</div>',
                        unsafe_allow_html=True)
            states_html = "".join(
                f'<span class="signal-chip" style="color:#a5f3fc">{s}</span>'
                for s in flow_model["fsm_states"]
            )
            st.markdown(states_html, unsafe_allow_html=True)

            # Condition logic table
            if flow_model["logic_table"]:
                st.markdown('<div class="section-header">Condition → Logic Table</div>',
                            unsafe_allow_html=True)
                import pandas as pd
                df = pd.DataFrame(flow_model["logic_table"])
                df.columns = ["Condition (Raw)", "Normalized Form"]
                st.dataframe(df, use_container_width=True, hide_index=True)

            # Download buttons
            st.markdown('<div class="section-header">Download Artifacts</div>',
                        unsafe_allow_html=True)
            d1, d2, d3 = st.columns(3)
            with d1:
                st.download_button(
                    "⬇ Sequence UML",
                    data=flow_model["sequence_uml"],
                    file_name="efs_sequence.puml",
                    mime="text/plain",
                )
            with d2:
                st.download_button(
                    "⬇ State UML",
                    data=flow_model["state_uml"],
                    file_name="efs_state.puml",
                    mime="text/plain",
                )
            with d3:
                full_json = json.dumps(
                    {"spec": spec_model, "protocol": proto_model, "flow": flow_model},
                    indent=2,
                )
                st.download_button(
                    "⬇ Full JSON",
                    data=full_json,
                    file_name="efs_pipeline_output.json",
                    mime="application/json",
                )

            with st.expander("Export: Flow Model JSON"):
                st.json(flow_model)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 4 — Raw Markdown
    # ──────────────────────────────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">LlamaParse Extracted Markdown</div>',
                    unsafe_allow_html=True)
        st.text_area(
            label="",
            value=markdown_text,
            height=500,
            label_visibility="collapsed",
        )

else:
    # ── Empty state ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center;padding:4rem 2rem;
                background:linear-gradient(135deg,#0d1117,#111827);
                border:1px dashed #1e293b;border-radius:16px;margin-top:2rem'>
        <div style='font-size:3rem;margin-bottom:1rem'>⚡</div>
        <div style='font-size:1.2rem;font-weight:600;color:#e2e8f0;margin-bottom:0.5rem'>
            Ready to process your specification
        </div>
        <div style='color:#475569;font-size:0.9rem;max-width:500px;margin:0 auto'>
            Upload a PDF/DOCX protocol specification — or enable Demo Mode —
            then click <strong style='color:#38bdf8'>▶ Run EFS Pipeline</strong>
            to execute all three stages.
        </div>
        <div style='margin-top:2rem;display:flex;justify-content:center;gap:2rem;
                    flex-wrap:wrap'>
            <div style='text-align:center'>
                <div style='font-family:IBM Plex Mono,monospace;font-size:0.7rem;
                             color:#38bdf8;letter-spacing:0.1em'>STAGE 01</div>
                <div style='color:#64748b;font-size:0.8rem'>Spec Intelligence</div>
            </div>
            <div style='color:#1e293b;font-size:1.2rem;align-self:center'>→</div>
            <div style='text-align:center'>
                <div style='font-family:IBM Plex Mono,monospace;font-size:0.7rem;
                             color:#a78bfa;letter-spacing:0.1em'>STAGE 02</div>
                <div style='color:#64748b;font-size:0.8rem'>Protocol Intelligence</div>
            </div>
            <div style='color:#1e293b;font-size:1.2rem;align-self:center'>→</div>
            <div style='text-align:center'>
                <div style='font-family:IBM Plex Mono,monospace;font-size:0.7rem;
                             color:#4ade80;letter-spacing:0.1em'>STAGE 03</div>
                <div style='color:#64748b;font-size:0.8rem'>Flow Generator</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
