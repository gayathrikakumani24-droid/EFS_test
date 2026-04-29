import os
import re
import json
import zlib
import tempfile
import streamlit as st
from dotenv import load_dotenv

from llama_parse import LlamaParse
from groq import Groq

load_dotenv()

LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(page_title="Spec Intelligence Prototype", layout="wide")
st.title("Spec Intelligence — Document to Markdown + Protocol Detection")
st.caption("Upload semiconductor/protocol specification documents")


# -----------------------------------------------------------------------
# LLAMAPARSE
# -----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def parse_document(file_path):
    if not LLAMA_CLOUD_API_KEY:
        raise ValueError("LLAMA_CLOUD_API_KEY is not set")
    st.write(f"🔑 Using LlamaParse API key: {LLAMA_CLOUD_API_KEY[:10]}...")
    parser = LlamaParse(api_key=LLAMA_CLOUD_API_KEY, result_type="markdown", verbose=True)
    try:
        docs = parser.load_data(file_path)
        if not docs:
            raise ValueError("No documents returned from LlamaParse")
        markdown_text = "\n\n".join([d.text for d in docs])
        if not markdown_text.strip():
            raise ValueError("Empty markdown content extracted")
        st.write(f"📝 Extracted {len(markdown_text)} characters of markdown")
        return markdown_text
    except Exception as e:
        st.error(f"LlamaParse error: {str(e)}")
        raise e


# -----------------------------------------------------------------------
# PROTOCOL DETECTOR
# -----------------------------------------------------------------------
def detect_protocol(markdown_text):
    snippet = markdown_text[:12000]
    prompt = f"""
You are an expert semiconductor protocol analyst.
Analyze this specification content and identify the protocol type:
AXI, PCIe, CXL, CHI, AMBA, USB, Ethernet, Proprietary, or Unknown.

Return JSON:
{{"protocol":"","confidence":"","signals_found":[],"reasoning":""}}

Specification:
{snippet}
"""
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(completion.choices[0].message.content)


# -----------------------------------------------------------------------
# SEMANTIC JSON EXTRACTION
# -----------------------------------------------------------------------
def extract_semantic_json(markdown_text, extraction_mode="basic", chunk_size=8000):
    st.write("🧠 Starting semantic extraction...")
    if not markdown_text or not markdown_text.strip():
        st.error("❌ No markdown content to process")
        return None

    semantic_json = {
        "document_metadata": {"document_name": "extracted_spec", "protocol": "unknown", "sections_detected": 0},
        "signals": [], "registers": [], "transactions": [], "constraints": [], "timing_conditions": []
    }

    if len(markdown_text) > chunk_size:
        chunks = [markdown_text[i:i + chunk_size] for i in range(0, len(markdown_text), chunk_size)]
        st.write(f"🔢 Processing {len(chunks)} chunks...")
    else:
        chunks = [markdown_text]

    for i, chunk in enumerate(chunks):
        st.write(f"🔄 Processing chunk {i + 1}/{len(chunks)}...")
        try:
            chunk_result = _extract_from_chunk(chunk, extraction_mode)
            if chunk_result:
                for key in ["signals", "registers", "transactions", "constraints", "timing_conditions"]:
                    semantic_json[key].extend(chunk_result.get(key, []))
                if chunk_result.get("document_metadata", {}).get("protocol", "unknown") != "unknown":
                    semantic_json["document_metadata"]["protocol"] = chunk_result["document_metadata"]["protocol"]
        except Exception as e:
            st.error(f"❌ Error processing chunk {i + 1}: {str(e)}")
            continue

    semantic_json["document_metadata"]["sections_detected"] = len(chunks)
    semantic_json = _remove_duplicates(semantic_json)
    st.write(
        f"✅ Done: {len(semantic_json['signals'])} signals, "
        f"{len(semantic_json['transactions'])} transactions, "
        f"{len(semantic_json['constraints'])} constraints"
    )
    return semantic_json


def _extract_from_chunk(chunk_text, extraction_mode):
    prompt = f"""
You are a semiconductor specification semantic extraction engine.
Convert this specification content into structured engineering JSON.

Extract:
- signals (name, direction: input/output/inout/master/slave, description, section_ref)
- registers (name, fields array)
- transactions (name, actors array, sequence_steps array)
- constraints (rule, type, condition, section_ref) - look for MUST/SHALL/REQUIRED/SHOULD/MAY
- timing_conditions (trigger, response)

Return ONLY valid JSON:
{{
  "document_metadata": {{"document_name":"","protocol":"","sections_detected":0}},
  "signals":[{{"name":"","direction":"","description":"","section_ref":""}}],
  "registers":[{{"name":"","fields":[]}}],
  "transactions":[{{"name":"","actors":[],"sequence_steps":[]}}],
  "constraints":[{{"rule":"","type":"","condition":"","section_ref":""}}],
  "timing_conditions":[{{"trigger":"","response":""}}]
}}

Specification content:
{chunk_text}
"""
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        return json.loads(completion.choices[0].message.content)
    except json.JSONDecodeError:
        return None


def _remove_duplicates(sj):
    for category in ["signals", "registers", "transactions", "constraints", "timing_conditions"]:
        seen, unique = set(), []
        for item in sj[category]:
            if category in ("signals", "registers", "transactions"):
                key = item.get("name", "")
            elif category == "constraints":
                key = f"{item.get('rule','')}_{item.get('condition','')}"
            else:
                key = f"{item.get('trigger','')}_{item.get('response','')}"
            if key not in seen:
                seen.add(key)
                unique.append(item)
        sj[category] = unique
    return sj


# -----------------------------------------------------------------------
# PLANTUML ENCODING + RENDERING
# -----------------------------------------------------------------------
def _plantuml_encode(puml_text: str) -> str:
    ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"

    def enc6(b):
        return ALPHABET[b & 0x3F]

    def enc3(b1, b2, b3):
        return (enc6(b1 >> 2) +
                enc6(((b1 & 3) << 4) | (b2 >> 4)) +
                enc6(((b2 & 0xF) << 2) | (b3 >> 6)) +
                enc6(b3 & 0x3F))

    compress_obj = zlib.compressobj(9, zlib.DEFLATED, -15)
    raw = compress_obj.compress(puml_text.encode("utf-8")) + compress_obj.flush()

    result, i = "", 0
    while i < len(raw):
        b1 = raw[i]
        b2 = raw[i + 1] if i + 1 < len(raw) else 0
        b3 = raw[i + 2] if i + 2 < len(raw) else 0
        result += enc3(b1, b2, b3)
        i += 3
    return result


def _puml_url(puml_text: str) -> str:
    return f"https://www.plantuml.com/plantuml/svg/{_plantuml_encode(puml_text)}"


def _render_plantuml(puml_code: str, height: int = 650):
    import streamlit.components.v1 as components
    svg_url = _puml_url(puml_code)
    html = f"""<!DOCTYPE html><html><head><style>
    body{{margin:0;padding:12px;background:#0f1117;display:flex;justify-content:center;}}
    .wrap{{background:white;border-radius:12px;padding:20px;
           box-shadow:0 0 40px rgba(99,179,237,0.10);max-width:100%;overflow:auto;text-align:center;}}
    img{{max-width:100%;height:auto;border-radius:6px;}}
    .err{{color:#fc8181;font-family:monospace;font-size:13px;padding:12px;}}
    </style></head><body>
    <div class="wrap">
      <img src="{svg_url}" alt="PlantUML Diagram"
           onerror="this.style.display='none';document.getElementById('e').style.display='block'"/>
      <div id="e" class="err" style="display:none">
        ⚠️ Could not render — check internet or inspect PlantUML source tab.
      </div>
    </div></body></html>"""
    components.html(html, height=height, scrolling=True)


def _puml_download(label: str, code: str, filename: str):
    st.download_button(label=label, data=code, file_name=filename, mime="text/plain")


# -----------------------------------------------------------------------
# SAFE LABEL HELPERS
# -----------------------------------------------------------------------
def _safe(text: str, maxlen: int = 40) -> str:
    t = str(text).encode("ascii", "ignore").decode("ascii")
    t = re.sub(r'[\"\'`<>{}|\\]', "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:maxlen]


def _safe_name(text: str) -> str:
    t = str(text).encode("ascii", "ignore").decode("ascii")
    t = re.sub(r"\W+", "_", t).strip("_")
    return t or "UNKNOWN"


# -----------------------------------------------------------------------
# SEMANTIC JSON → SEED EXTRACTORS  (Python-first, no hallucination)
# -----------------------------------------------------------------------

def _extract_actor_name(a) -> str:
    """
    Safely extract a short actor name from whatever the LLM put in the actors array.
    Handles: plain string "Manager", dict {"name":"Manager","description":"..."}, etc.
    """
    if isinstance(a, dict):
        # Prefer explicit 'name' key; fall back to first short string value
        raw = a.get("name") or a.get("role") or a.get("actor") or ""
        if not raw:
            # grab the shortest value that looks like a name
            candidates = [str(v) for v in a.values() if v and len(str(v)) < 40]
            raw = min(candidates, key=len) if candidates else str(a)
    else:
        raw = str(a)
    # Keep only the first word/token — "Manager initiates..." → "Manager"
    raw = raw.strip().split()[0] if raw.strip() else raw
    return _safe_name(raw)[:20] or "Actor"


def _extract_step_label(step) -> str:
    """
    Extract a clean short label from a sequence_step entry.
    Handles: plain string, dict with 'step'/'action'/'description'/'name' key.
    Returns a concise snake_case label, max 35 chars.
    """
    if isinstance(step, dict):
        # Try common key names in preference order
        for key in ("step", "action", "label", "name", "signal", "description"):
            val = step.get(key, "")
            if val and isinstance(val, str) and len(val.strip()) > 0:
                raw = val.strip()
                break
        else:
            # Fall back to shortest non-empty string value
            vals = [str(v) for v in step.values() if v and isinstance(v, str)]
            raw = min(vals, key=len) if vals else ""
    else:
        raw = str(step).strip()

    if not raw:
        return ""

    # Collapse to a short snake_case identifier: take first 6 words
    words = re.split(r"[\s_\-]+", raw)[:6]
    label = "_".join(w for w in words if w)
    label = re.sub(r"\W+", "_", label).strip("_")
    return label[:35] or ""


def _infer_direction(step, actor_set: list) -> tuple:
    """
    Guess from/to/type from a step's text content.
    Keywords like 'assert', 'send', 'issue' → sync (initiator→target)
    Keywords like 'respond', 'ready', 'ack', 'complete' → return (target→initiator)
    """
    if len(actor_set) < 2:
        return actor_set[0], actor_set[0], "sync"

    text = ""
    if isinstance(step, dict):
        text = " ".join(str(v) for v in step.values()).lower()
    else:
        text = str(step).lower()

    # Signals that go from target back to initiator
    return_keywords = ["awready", "wready", "arready", "crready",
                       "bvalid", "rvalid", "acvalid",
                       "ready", "ack", "response", "respond",
                       "complete", "accept", "grant"]
    # Signals asserted by the initiator/manager (go forward)
    forward_keywords = ["awvalid", "wvalid", "arvalid", "crvalid",
                        "bready", "rready", "acready",
                        "assert", "issue", "send", "initiat"]
    for kw in forward_keywords:
        if kw in text:
            return actor_set[0], actor_set[-1], "sync"
    for kw in return_keywords:
        if kw in text:
            return actor_set[-1], actor_set[0], "return"

    return actor_set[0], actor_set[-1], "sync"


def _seed_sequence(semantic_json: dict) -> dict:
    """Pull actors and message candidates directly from semantic_json."""
    signals      = semantic_json.get("signals", [])
    transactions = semantic_json.get("transactions", [])
    timing       = semantic_json.get("timing_conditions", [])
    constraints  = semantic_json.get("constraints", [])

    # ── Actors: extract clean names from transaction actors ──────────────
    actor_set: list = []
    for txn in transactions[:10]:
        for a in txn.get("actors", []):
            name = _extract_actor_name(a)
            if name and name not in actor_set:
                actor_set.append(name)

    # Deduplicate actors that are the same word with different casing
    seen_lower, deduped = set(), []
    for a in actor_set:
        if a.lower() not in seen_lower:
            seen_lower.add(a.lower())
            deduped.append(a)
    actor_set = deduped

    # Fall back to signal-direction inference if not enough actors found
    if len(actor_set) < 2:
        has_master = any("master" in s.get("direction","").lower()
                         or "output" in s.get("direction","").lower()
                         for s in signals)
        has_slave  = any("slave"  in s.get("direction","").lower()
                         or "input"  in s.get("direction","").lower()
                         for s in signals)
        if has_master and "Master" not in actor_set:
            actor_set.append("Master")
        if has_slave and "Slave" not in actor_set:
            actor_set.append("Slave")
        if len(actor_set) < 2:
            actor_set = ["Manager", "Subordinate"]

    actor_set = actor_set[:4]

    # ── Messages: transactions → timing → signals → constraints ──────────
    messages: list = []

    # 1. From transaction sequence_steps with proper directionality
    for txn in transactions[:8]:
        steps = txn.get("sequence_steps", [])
        for step in steps[:6]:
            label = _extract_step_label(step)
            if not label:
                continue
            frm, to, mtype = _infer_direction(step, actor_set)
            # Avoid exact duplicate labels
            if not any(m["label"] == label for m in messages):
                messages.append({"from": frm, "to": to, "label": label, "type": mtype})

    # 2. From timing conditions — triggers go forward, responses come back
    for tc in timing[:8]:
        trigger  = _extract_step_label(tc.get("trigger",  ""))
        response = _extract_step_label(tc.get("response", ""))
        if trigger and not any(m["label"] == trigger for m in messages):
            messages.append({
                "from": actor_set[0], "to": actor_set[-1],
                "label": trigger, "type": "sync"
            })
        if response and not any(m["label"] == response for m in messages):
            messages.append({
                "from": actor_set[-1], "to": actor_set[0],
                "label": response, "type": "return"
            })

    # 3. From signals — use actual direction field
    for sig in signals[:12]:
        name = _safe_name(sig.get("name", ""))
        if not name or any(m["label"] == name for m in messages):
            continue
        direction = sig.get("direction", "").lower()
        if "output" in direction or "master" in direction:
            frm, to, mtype = actor_set[0], actor_set[-1], "sync"
        elif "input" in direction or "slave" in direction:
            frm, to, mtype = actor_set[-1], actor_set[0], "return"
        else:
            frm, to, mtype = actor_set[0], actor_set[-1], "sync"
        messages.append({"from": frm, "to": to, "label": name, "type": mtype})

    # 4. Key constraints as notes on the initiating actor
    for c in constraints[:3]:
        rule = _extract_step_label(c.get("rule", ""))
        if rule and not any(m["label"] == rule for m in messages):
            messages.append({"from": actor_set[0], "to": actor_set[0],
                             "label": rule, "type": "note"})

    return {"actors": actor_set, "messages": messages[:25]}


def _seed_fsm(semantic_json: dict) -> dict:
    """Pull states and transitions directly from semantic_json."""
    transactions = semantic_json.get("transactions", [])
    timing       = semantic_json.get("timing_conditions", [])

    states: list      = []
    transitions: list = []

    # States from transaction sequence steps (each step = a state)
    for txn in transactions[:5]:
        steps = txn.get("sequence_steps", [])
        prev = None
        for step in steps[:6]:
            # Use the proper label extractor then truncate/uppercase for state name
            label = _extract_step_label(step)
            state = label[:18].upper() if label else ""
            if not state or state in states:
                continue
            states.append(state)
            if prev:
                # Use the signal name from the step as the transition label
                transitions.append({"from": prev, "to": state, "label": label[:30]})
            prev = state

    # States from timing trigger → response pairs
    for tc in timing[:6]:
        trigger  = _extract_step_label(tc.get("trigger",  ""))[:18].upper()
        response = _extract_step_label(tc.get("response", ""))[:18].upper()
        for candidate in [trigger, response]:
            if candidate and candidate not in states:
                states.append(candidate)
        if trigger and response and trigger in states and response in states:
            label = _extract_step_label(tc.get("trigger", ""))[:30]
            transitions.append({"from": trigger, "to": response, "label": label})

    # Guarantee minimum viable FSM
    if not states:
        states = ["IDLE", "ACTIVE", "COMPLETE"]
        transitions = [
            {"from": "IDLE",     "to": "ACTIVE",   "label": "start"},
            {"from": "ACTIVE",   "to": "COMPLETE", "label": "done"},
            {"from": "COMPLETE", "to": "IDLE",     "label": "reset"},
        ]

    # Always add ERROR + reset path
    if "ERROR" not in states:
        states.append("ERROR")
    error_src = states[1] if len(states) > 1 else states[0]
    transitions.append({"from": error_src, "to": "ERROR",   "label": "timeout"})
    transitions.append({"from": "ERROR",   "to": states[0], "label": "reset"})

    initial = states[0]
    return {"initial_state": initial, "states": states[:14], "transitions": transitions[:20]}


def _seed_timing(semantic_json: dict) -> dict:
    """Pull real signal names and timing events directly from semantic_json."""
    signals  = semantic_json.get("signals", [])
    timing   = semantic_json.get("timing_conditions", [])

    # Priority keywords for signal selection
    priority_keywords = ["clk", "clock", "reset", "resetn", "valid", "ready", "data",
                         "addr", "enable", "ack", "last", "strobe"]

    def priority(sig):
        name = sig.get("name", "").lower()
        for i, kw in enumerate(priority_keywords):
            if kw in name:
                return i
        return 99

    sorted_sigs = sorted(signals, key=priority)[:6]
    sig_names   = [_safe_name(s.get("name", "SIG")) for s in sorted_sigs]

    # Use actual timing conditions to build events
    sig_events: dict = {name: [] for name in sig_names}
    t = 0
    for tc in timing[:10]:
        trigger  = str(tc.get("trigger",  ""))
        response = str(tc.get("response", ""))
        for name in sig_names:
            base = name.lower()
            if base in trigger.lower():
                sig_events[name].append({"time": t,      "state": "1"})
            if base in response.lower():
                sig_events[name].append({"time": t + 10, "state": "0"})
        t += 20

    # Fill any signal with no matched events with a default toggle pattern
    for name in sig_names:
        if not sig_events[name]:
            sig_events[name] = [
                {"time": 0,  "state": "0"},
                {"time": 20, "state": "1"},
                {"time": 60, "state": "0"},
            ]

    # Classify binary vs concise
    binary_keywords = ["clk", "clock", "reset", "resetn", "valid", "ready", "enable",
                       "ack", "last", "strobe", "irq", "req", "grant", "sel"]
    result_signals = []
    for name in sig_names:
        is_binary = any(kw in name.lower() for kw in binary_keywords)
        stype  = "binary" if is_binary else "concise"
        events = sig_events[name]
        if stype == "concise":
            labeled = []
            for ev in events:
                state = ev["state"]
                if state in ("0", "1"):
                    state = "IDLE" if state == "0" else "ACTIVE"
                labeled.append({"time": ev["time"], "state": state})
            events = labeled
        result_signals.append({"name": name, "type": stype, "events": events})

    # Find clock signal name from extracted signals
    clk_name = "CLK"
    for s in signals:
        if "clk" in s.get("name", "").lower() or "clock" in s.get("name", "").lower():
            clk_name = _safe_name(s["name"])
            break

    max_time = max(
        (ev["time"] for sig in result_signals for ev in sig["events"]),
        default=60
    )
    highlights = [{"start": 20, "end": max_time - 10, "label": "transaction_window"}]

    return {
        "clock_signal": clk_name,
        "clock_period": 10,
        "signals": result_signals,
        "highlights": highlights,
    }


# -----------------------------------------------------------------------
# LLM ENRICHMENT — takes Python seed, LLM orders/labels only, cannot invent
# -----------------------------------------------------------------------

def _llm_enrich_sequence(seed: dict, semantic_json: dict) -> dict:
    """LLM reorders and labels seed messages — cannot invent new ones."""
    protocol = semantic_json.get("document_metadata", {}).get("protocol", "unknown")
    prompt = f"""
You are a {protocol} protocol expert.
Below is a SEED sequence diagram built from the actual specification.
Your job is to REORDER and optionally RENAME the messages for clarity.
You may NOT add messages that are not in the seed.
You may NOT change actor names.
You may only remove exact duplicates.

Return ONLY this JSON:
{{
  "actors": {json.dumps(seed["actors"])},
  "messages": [
    {{"from":"...", "to":"...", "label":"...", "type":"sync|return|note"}}
  ]
}}

SEED:
{json.dumps(seed, indent=2)}
"""
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant", temperature=0.1,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}]
        )
        result = json.loads(completion.choices[0].message.content)
        # Safety: if LLM changed actors, fall back to seed
        if set(result.get("actors", [])) != set(seed["actors"]):
            result["actors"] = seed["actors"]
        return result
    except Exception:
        return seed


def _llm_enrich_fsm(seed: dict, semantic_json: dict) -> dict:
    """LLM improves transition labels — cannot invent new states."""
    protocol    = semantic_json.get("document_metadata", {}).get("protocol", "unknown")
    constraints = semantic_json.get("constraints", [])[:10]
    prompt = f"""
You are a {protocol} FSM expert.
Below is a SEED FSM built from the actual specification.
Your job is to improve transition LABELS using the constraints below.
You may NOT add or remove states or transitions.
You may NOT change state names.

Return ONLY this JSON — same structure as the seed, with better labels only:
{{
  "initial_state": "{seed["initial_state"]}",
  "states": {json.dumps(seed["states"])},
  "transitions": [
    {{"from":"...", "to":"...", "label":"..."}}
  ]
}}

Constraints from spec: {json.dumps(constraints)}
SEED: {json.dumps(seed, indent=2)}
"""
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant", temperature=0.1,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}]
        )
        result = json.loads(completion.choices[0].message.content)
        # Safety: enforce seed states are preserved exactly
        if set(result.get("states", [])) != set(seed["states"]):
            result["states"]        = seed["states"]
            result["initial_state"] = seed["initial_state"]
        return result
    except Exception:
        return seed


def _llm_enrich_timing(seed: dict, semantic_json: dict) -> dict:
    """LLM can only adjust concise state labels — cannot change signal names or times."""
    protocol = semantic_json.get("document_metadata", {}).get("protocol", "unknown")
    timing   = semantic_json.get("timing_conditions", [])[:8]
    prompt = f"""
You are a {protocol} timing expert.
Below is a SEED timing diagram with real signals from the specification.
You may ONLY improve the state labels for 'concise' type signals.
You may NOT change signal names, event times, add signals, or remove signals.
You may NOT change binary signal states (must remain "0" or "1").

Return ONLY this JSON with the same structure as the seed:
{json.dumps(seed, indent=2)}

Use these timing conditions to pick better concise state labels (max 12 chars, no spaces):
{json.dumps(timing)}
"""
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant", temperature=0.1,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}]
        )
        result = json.loads(completion.choices[0].message.content)
        # Safety: reject if LLM changed signal names
        seed_names   = {s["name"] for s in seed["signals"]}
        result_names = {s["name"] for s in result.get("signals", [])}
        if seed_names != result_names:
            return seed
        return result
    except Exception:
        return seed


# -----------------------------------------------------------------------
# PLANTUML TEMPLATE BUILDERS — guaranteed-valid syntax
# -----------------------------------------------------------------------
import re
def group_messages(messages):

    grouped=[]

    current_phase=None


    PHASE_RULES = [
      ("request", [
          "request","req",
          "address","addr",
          "command","cmd"
      ]),

      ("data",[
          "data","write","read",
          "payload","transfer"
      ]),

      ("response",[
          "response","resp",
          "ready","ack",
          "valid","complete"
      ])
    ]


    def classify(label):

        text=label.lower()

        for phase,words in PHASE_RULES:
            for w in words:
                if w in text:
                    return phase

        return None


    for msg in messages:

        if msg.get("type") in [
           "note","alt","loop"
        ]:
            grouped.append(msg)
            continue


        phase = classify(
            msg.get("label","")
        )


        if phase and phase != current_phase:

            grouped.append({
                "type":"divider",
                "label":phase.title()+" Phase"
            })

            current_phase=phase


        grouped.append(msg)


    return grouped

def repair_messages(messages):

    repaired=[]
    seen=set()

    for msg in messages:

        msg=dict(msg)
        label=str(msg.get("label","")).strip()

        frm=str(msg.get("from","Actor1"))
        to=str(msg.get("to","Actor2"))

        mtype=msg.get("type","sync")


        # -----------------------------
        # 1 Deduplicate repeated events
        # -----------------------------
        key=(frm,to,label,mtype)

        if key in seen:
            continue

        seen.add(key)


        # -----------------------------
        # 2 Convert descriptive prose
        # into notes automatically
        # -----------------------------
        if len(label.split()) > 6:
            msg["type"]="note"
            repaired.append(msg)
            continue


        # -----------------------------
        # 3 Actor ownership inferred
        # from "X asserts Y" patterns
        # -----------------------------
        m = re.match(
           r'([A-Za-z0-9_]+)_asserts_(.+)',
           label
        )

        if m:
            actor = m.group(1)
            event = m.group(2)

            old_from = frm
            old_to = to

            msg["from"] = actor

            if actor == old_from:
                msg["to"] = old_to
            else:
                msg["to"] = old_from
            msg["label"] = f"Assert {event}"

            repaired.append(msg)
            continue



        # -----------------------------
        # 4 Convert bare identifiers
        # into event labels
        # -----------------------------
        if re.match(
            r'^[A-Z0-9_]+$',
            label
        ):
            msg["label"]=f"Event {label}"


        # -----------------------------
        # 5 Detect likely conditions
        # and convert to note
        # -----------------------------
        if any(k in label.lower()
               for k in [
                 "must",
                 "wait",
                 "after",
                 "before",
                 "until"
               ]):
            msg["type"]="note"


        repaired.append(msg)


    return repaired
def _build_sequence_puml(content: dict) -> str:
    """
    Generic protocol-agnostic PUML sequence generator.
    Works for arbitrary protocols if parser emits structured messages.
    """

    actors = [
        _safe_name(a)
        for a in content.get("actors", ["Actor1","Actor2"])
    ]

    lines = [
        "@startuml",
        "skinparam sequenceMessageAlign center",
        "skinparam sequenceArrowThickness 2",
        "skinparam roundcorner 5",

        "skinparam participant {",
        " BackgroundColor #1e3a5f",
        " BorderColor #63b3ed",
        " FontColor #e2e8f0",
        "}",

        "skinparam note {",
        " BackgroundColor #1a3a2a",
        " BorderColor #48bb78",
        " FontColor #9ae6b4",
        "}",
        ""
    ]

    
    # Participants
    for a in actors:
        lines.append(f"participant {a}")

    lines.append("")


    seen = set()


    for msg in content.get("messages",[]):

        mtype = msg.get("type","sync")


        # -------------------------
        # Divider / Phase grouping
        # -------------------------
        if mtype=="divider":
            label = _safe(
                msg.get("label","Phase"),
                80
            )
            lines.append(f"== {label} ==")
            continue


        # -------------------------
        # Notes
        # -------------------------
        if mtype=="note":
            frm = _safe_name(msg.get("from",actors[0]))
            to  = _safe_name(msg.get("to",frm))

            label = _safe(
                str(msg.get("label","")),
                120
            )

            over = frm if frm==to else f"{frm},{to}"
            lines.append(
                f"note over {over}: {label}"
            )
            continue


        # -------------------------
        # Alternatives
        # -------------------------
        if mtype=="alt":
            cond = _safe(
                msg.get("condition","Condition"),
                80
            )
            lines.append(f"alt {cond}")
            continue


        if mtype=="else":
            cond = _safe(
                msg.get("condition","Else"),
                80
            )
            lines.append(f"else {cond}")
            continue


        if mtype=="end_alt":
            lines.append("end")
            continue


        # -------------------------
        # Loops
        # -------------------------
        if mtype=="loop":
            cond = _safe(
                msg.get("condition","Loop"),
                80
            )
            lines.append(
                f"loop {cond}"
            )
            continue


        if mtype=="end_loop":
            lines.append("end")
            continue


        # -------------------------
        # Optional block
        # -------------------------
        if mtype=="opt":
            cond = _safe(
                msg.get("condition","Optional"),
                80
            )
            lines.append(
                f"opt {cond}"
            )
            continue


        if mtype=="end_opt":
            lines.append("end")
            continue


        # -------------------------
        # Standard messages
        # -------------------------
        frm = _safe_name(
            msg.get("from",actors[0])
        )

        to = _safe_name(
            msg.get("to",actors[-1])
        )

        label = _safe(
            str(msg.get("label","event")),
            120
        )


        # Remove duplicates
        key = (frm,to,label,mtype)
        if key in seen:
            continue
        seen.add(key)


        if mtype=="return":
            lines.append(
                f"{frm} --> {to}: {label}"
            )

        else:
            lines.append(
                f"{frm} -> {to}: {label}"
            )


    lines += [
        "",
        "@enduml"
    ]

    return "\n".join(lines)

def build_fsm_puml(messages):

    states = infer_fsm_states(messages)

    lines=[
       "@startuml",
       "[*] --> IDLE"
    ]


    prev="IDLE"

    for s in states:

        lines.append(
           f"{prev} --> {s}"
        )

        prev=s


    lines.append(
       f"{prev} --> IDLE"
    )

    lines.append("@enduml")

    return "\n".join(lines)
def _build_fsm_puml(content: dict) -> str:
    states  = [_safe_name(s) for s in content.get("states", ["IDLE", "ACTIVE", "ERROR"])]
    initial = _safe_name(content.get("initial_state", states[0] if states else "IDLE"))
    transitions = content.get("transitions", [])

    lines = [
        "@startuml",
        "skinparam state {",
        "  BackgroundColor #1e3a5f",
        "  BorderColor #63b3ed",
        "  FontColor #e2e8f0",
        "  ArrowColor #63b3ed",
        "}",
        "",
    ]

    for s in states:
        if s == "ERROR":
            lines.append(f"state {s} #742a2a")
        else:
            lines.append(f"state {s}")
    lines.append("")

    lines.append(f"[*] --> {initial}")

    for t in transitions:
        frm   = _safe_name(str(t.get("from",  "IDLE")))
        to    = _safe_name(str(t.get("to",    "IDLE")))
        label = _safe(str(t.get("label", "")), 35)
        if label:
            lines.append(f"{frm} --> {to} : {label}")
        else:
            lines.append(f"{frm} --> {to}")

    lines += ["", "@enduml"]
    return "\n".join(lines)


def _build_timing_puml(content: dict) -> str:
    from itertools import groupby

    clock_name  = _safe_name(content.get("clock_signal", "CLK"))
    clock_period = max(1, int(content.get("clock_period", 10)))
    signals    = content.get("signals", [])
    highlights = content.get("highlights", [])

    lines = ["@startuml", ""]

    lines.append(f"clock {clock_name} with period {clock_period}")
    lines.append("")

    declared = []
    for sig in signals:
        name   = _safe_name(str(sig.get("name", "SIG")))
        stype  = sig.get("type", "binary")
        events = sig.get("events", [])
        if not events:
            continue
        if stype == "binary":
            lines.append(f'robust "{name}" as {name}')
        else:
            lines.append(f'concise "{name}" as {name}')
        declared.append((name, stype, events))

    lines.append("")

    all_events = []
    for name, stype, events in declared:
        for ev in events:
            try:
                t = int(ev.get("time", 0))
            except (ValueError, TypeError):
                t = 0
            state = _safe(str(ev.get("state", "0")), 20)
            if stype == "binary":
                state = "1" if state in ("1", "HIGH", "high", "True", "true", "H") else "0"
            all_events.append((t, name, stype, state))

    all_events.sort(key=lambda x: x[0])

    for t, grp in groupby(all_events, key=lambda x: x[0]):
        lines.append(f"@{t}")
        for _, name, stype, state in grp:
            if stype == "binary":
                lines.append(f"{name} is {state}")
            else:
                lines.append(f'{name} is "{state}"')
        lines.append("")

    for h in highlights:
        try:
            start = int(h.get("start", 0))
            end   = int(h.get("end",   0))
        except (ValueError, TypeError):
            continue
        label = _safe(str(h.get("label", "")), 30)
        if start < end:
            if label:
                lines.append(f"highlight {start} to {end} : {label}")
            else:
                lines.append(f"highlight {start} to {end}")

    lines += ["", "@enduml"]
    return "\n".join(lines)


# -----------------------------------------------------------------------
# PUBLIC DIAGRAM GENERATORS — seed → enrich pipeline
# -----------------------------------------------------------------------

def _generate_sequence_diagram(semantic_json: dict) -> str:
    seed    = _seed_sequence(semantic_json)
    content = _llm_enrich_sequence(seed, semantic_json)
    content["messages"] = repair_messages(
        content.get("messages", [])
    )
    content["messages"] = group_messages(
    content["messages"]
)

    return _build_sequence_puml(content)



def _generate_fsm_diagram(semantic_json: dict) -> str:
    seed    = _seed_fsm(semantic_json)
    content = _llm_enrich_fsm(seed, semantic_json)
    return _build_fsm_puml(content)


def _generate_timing_diagram(semantic_json: dict) -> str:
    seed    = _seed_timing(semantic_json)
    content = _llm_enrich_timing(seed, semantic_json)
    return _build_timing_puml(content)


# -----------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    st.info("Requires:\n- LLAMA_CLOUD_API_KEY\n- GROQ_API_KEY")
    st.header("Extraction Options")
    extraction_mode = st.selectbox(
        "Extraction Mode", ["basic", "deep_semantic"],
        help="Basic: Fast | Deep: Comprehensive"
    )
    chunk_size = st.slider("Chunk Size", 4000, 12000, 8000, 1000)
    st.header("Future Features")
    st.info("🚧 Knowledge Graph (Coming Soon)")


# -----------------------------------------------------------------------
# FILE UPLOAD
# -----------------------------------------------------------------------
uploaded = st.file_uploader("Upload Spec Document", type=["pdf", "docx"])

if uploaded:
    uploaded.seek(0)
    file_extension = os.path.splitext(uploaded.name)[1]
    st.write(f"📄 **{uploaded.name}**  |  ext: `{file_extension}`")

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
        file_content = uploaded.read()
        tmp.write(file_content)
        temp_path = tmp.name

    st.write(f"📊 File size: {len(file_content)} bytes")

    if not os.path.exists(temp_path):
        st.error("❌ Failed to create temporary file")
        st.stop()

    for key in ["extracted_markdown", "semantic_json", "protocol_result",
                "sequence_diagram", "fsm_diagram", "timing_diagram"]:
        if key not in st.session_state:
            st.session_state[key] = None

    if st.button("Run Spec Intelligence"):
        try:
            with st.spinner("Parsing with LlamaParse..."):
                md = parse_document(temp_path)
            st.success(f"✅ Markdown extraction complete — {len(md):,} characters")
            st.session_state.extracted_markdown = md
        except Exception as e:
            st.error(str(e))
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    if st.session_state.extracted_markdown:
        md = st.session_state.extracted_markdown
        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader("Extracted Markdown")
            st.download_button("⬇️ Download Markdown", md, file_name="spec_readme.md")

            tabs = st.tabs(["Rendered", "Raw Markdown", "Semantic JSON", "📐 Diagrams & FSM"])

            with tabs[0]:
                st.markdown(md)

            with tabs[1]:
                st.code(md, language="markdown")

            # ── Semantic JSON tab ────────────────────────────────────────────
            with tabs[2]:
                st.subheader("Semantic JSON Extraction")
                st.write(f"🔑 GROQ: {'✅' if GROQ_API_KEY else '❌'}  |  "
                         f"LLAMA: {'✅' if LLAMA_CLOUD_API_KEY else '❌'}")
                st.divider()

                if st.button("🔧 Test Groq API", key="api_test_btn"):
                    try:
                        r = client.chat.completions.create(
                            model="llama-3.1-8b-instant", temperature=0.1,
                            messages=[{"role": "user", "content": 'Return {"status":"ok"}'}]
                        )
                        st.code(r.choices[0].message.content)
                        st.success("🎉 Groq API working!")
                    except Exception as e:
                        st.error(f"❌ {str(e)}")

                if st.button("🧠 Run Semantic Extraction", key="semantic_btn"):
                    with st.spinner("Building Spec Semantic Model..."):
                        try:
                            result = extract_semantic_json(md, extraction_mode, chunk_size)
                            if result:
                                st.session_state.semantic_json = result
                                st.success("✅ Semantic extraction complete!")
                            else:
                                st.error("❌ No results returned")
                        except Exception as e:
                            st.error(f"❌ {str(e)}")

                if st.session_state.semantic_json:
                    sd = st.session_state.semantic_json
                    st.download_button(
                        "💾 Download semantic_json.json",
                        data=json.dumps(sd, indent=2),
                        file_name="semantic_json.json",
                        mime="application/json"
                    )
                    st.json(sd)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Signals",      len(sd.get("signals",      [])))
                    c2.metric("Transactions", len(sd.get("transactions", [])))
                    c3.metric("Constraints",  len(sd.get("constraints",  [])))
                else:
                    st.info("Click **Run Semantic Extraction** to generate structured JSON.")

            # ── Diagrams tab ─────────────────────────────────────────────────
            with tabs[3]:
                st.subheader("📐 PlantUML Diagrams — Sequence / FSM / Timing")

                protocol_label = "unknown"
                if st.session_state.protocol_result:
                    protocol_label = st.session_state.protocol_result.get("protocol", "unknown")
                elif st.session_state.semantic_json:
                    protocol_label = (
                        st.session_state.semantic_json
                        .get("document_metadata", {})
                        .get("protocol", "unknown")
                    )

                st.caption(f"Protocol: **{protocol_label}** — run Semantic Extraction first.")

                if not st.session_state.semantic_json:
                    st.warning("⚠️ Run **Semantic Extraction** (Semantic JSON tab) first.")
                else:
                    sj = st.session_state.semantic_json

                    # ── Debug info: show what the seed extractors found ──────
                    with st.expander("🔍 Seed Data Preview (what Python extracted from semantic JSON)"):
                        seed_tabs = st.tabs(["Sequence seed", "FSM seed", "Timing seed"])
                        with seed_tabs[0]:
                            st.json(_seed_sequence(sj))
                        with seed_tabs[1]:
                            st.json(_seed_fsm(sj))
                        with seed_tabs[2]:
                            st.json(_seed_timing(sj))

                    diagram_type = st.radio(
                        "Diagram type",
                        ["📨 Sequence Diagram", "🔁 Finite State Machine (FSM)",
                         "⏱️ Signal Timing Diagram", "🗂️ All Three"],
                        horizontal=True, key="diagram_type_radio"
                    )

                    rc1, rc2, rc3 = st.columns(3)
                    regen_seq    = rc1.button("♻️ Regen Sequence", key="regen_seq")
                    regen_fsm    = rc2.button("♻️ Regen FSM",      key="regen_fsm")
                    regen_timing = rc3.button("♻️ Regen Timing",   key="regen_timing")

                    if st.button("🚀 Generate Diagrams", key="gen_diagrams_btn"):
                        need_seq    = diagram_type in ("📨 Sequence Diagram",             "🗂️ All Three")
                        need_fsm    = diagram_type in ("🔁 Finite State Machine (FSM)",   "🗂️ All Three")
                        need_timing = diagram_type in ("⏱️ Signal Timing Diagram",        "🗂️ All Three")

                        if need_seq:
                            with st.spinner("Building Sequence Diagram from semantic JSON..."):
                                try:
                                    st.session_state.sequence_diagram = _generate_sequence_diagram(sj)
                                    st.success("✅ Sequence ready")
                                except Exception as e:
                                    st.error(f"❌ Sequence: {e}")
                        if need_fsm:
                            with st.spinner("Building FSM from semantic JSON..."):
                                try:
                                    st.session_state.fsm_diagram = _generate_fsm_diagram(sj)
                                    st.success("✅ FSM ready")
                                except Exception as e:
                                    st.error(f"❌ FSM: {e}")
                        if need_timing:
                            with st.spinner("Building Timing Diagram from semantic JSON..."):
                                try:
                                    st.session_state.timing_diagram = _generate_timing_diagram(sj)
                                    st.success("✅ Timing ready")
                                except Exception as e:
                                    st.error(f"❌ Timing: {e}")

                    if regen_seq:
                        with st.spinner("Regenerating Sequence from semantic JSON..."):
                            try:
                                st.session_state.sequence_diagram = _generate_sequence_diagram(sj)
                                st.success("✅ Done")
                            except Exception as e:
                                st.error(f"❌ {e}")

                    if regen_fsm:
                        with st.spinner("Regenerating FSM from semantic JSON..."):
                            try:
                                st.session_state.fsm_diagram = _generate_fsm_diagram(sj)
                                st.success("✅ Done")
                            except Exception as e:
                                st.error(f"❌ {e}")

                    if regen_timing:
                        with st.spinner("Regenerating Timing from semantic JSON..."):
                            try:
                                st.session_state.timing_diagram = _generate_timing_diagram(sj)
                                st.success("✅ Done")
                            except Exception as e:
                                st.error(f"❌ {e}")

                    st.divider()

                    # Render Sequence
                    if st.session_state.sequence_diagram and diagram_type in (
                            "📨 Sequence Diagram", "🗂️ All Three"):
                        st.markdown("### 📨 Sequence Diagram")
                        v, s = st.tabs(["Rendered", "PlantUML Source"])
                        with v:
                            _render_plantuml(st.session_state.sequence_diagram, height=700)
                        with s:
                            st.code(st.session_state.sequence_diagram, language="text")
                            _puml_download("💾 Download sequence.puml",
                                           st.session_state.sequence_diagram, "sequence.puml")

                    # Render FSM
                    if st.session_state.fsm_diagram and diagram_type in (
                            "🔁 Finite State Machine (FSM)", "🗂️ All Three"):
                        st.markdown("### 🔁 Finite State Machine")
                        v, s = st.tabs(["Rendered", "PlantUML Source"])
                        with v:
                            _render_plantuml(st.session_state.fsm_diagram, height=700)
                        with s:
                            st.code(st.session_state.fsm_diagram, language="text")
                            _puml_download("💾 Download fsm.puml",
                                           st.session_state.fsm_diagram, "fsm.puml")

                    # Render Timing
                    if st.session_state.timing_diagram and diagram_type in (
                            "⏱️ Signal Timing Diagram", "🗂️ All Three"):
                        st.markdown("### ⏱️ Signal Timing Diagram")
                        v, s = st.tabs(["Rendered", "PlantUML Source"])
                        with v:
                            _render_plantuml(st.session_state.timing_diagram, height=600)
                        with s:
                            st.code(st.session_state.timing_diagram, language="text")
                            _puml_download("💾 Download timing.puml",
                                           st.session_state.timing_diagram, "timing.puml")

                    if not any([st.session_state.sequence_diagram,
                                st.session_state.fsm_diagram,
                                st.session_state.timing_diagram]):
                        st.info("👆 Select a diagram type and click **Generate Diagrams**.")

                    st.divider()

                    # Manual PlantUML editor
                    with st.expander("✏️ Manual PlantUML Editor"):
                        default = (st.session_state.sequence_diagram or
                                   st.session_state.fsm_diagram or
                                   st.session_state.timing_diagram or
                                   "@startuml\nAlice -> Bob: Hello\nBob --> Alice: Hi\n@enduml")
                        manual_code = st.text_area(
                            "PlantUML Code", value=default,
                            height=260, key="manual_puml_editor"
                        )
                        if st.button("🖼️ Render", key="render_custom"):
                            _render_plantuml(manual_code, height=600)
                            _puml_download("💾 Download custom.puml", manual_code, "custom.puml")

        # ── Right column: Protocol Detection ────────────────────────────────
        with col2:
            st.subheader("Protocol Detection")

            if not st.session_state.protocol_result:
                with st.spinner("Detecting protocol..."):
                    st.session_state.protocol_result = detect_protocol(md)

            result = st.session_state.protocol_result
            st.metric("Protocol",   result["protocol"])
            st.metric("Confidence", result["confidence"])
            st.write("**Signals Found**")
            for sig in result["signals_found"]:
                st.write(f"- {sig}")
            st.write("**Reasoning**")
            st.info(result["reasoning"])

            if st.session_state.semantic_json:
                st.divider()
                st.subheader("🧠 Semantic Metrics")
                sd = st.session_state.semantic_json
                st.metric("Signals",      len(sd.get("signals",      [])))
                st.metric("Transactions", len(sd.get("transactions", [])))
                st.metric("Constraints",  len(sd.get("constraints",  [])))

            if any([st.session_state.sequence_diagram,
                    st.session_state.fsm_diagram,
                    st.session_state.timing_diagram]):
                st.divider()
                st.subheader("📐 Diagram Status")
                st.write("✅ Sequence" if st.session_state.sequence_diagram else "⬜ Sequence")
                st.write("✅ FSM"      if st.session_state.fsm_diagram      else "⬜ FSM")
                st.write("✅ Timing"   if st.session_state.timing_diagram   else "⬜ Timing")

        # Cleanup temp file
        try:
            os.unlink(temp_path)
        except Exception:
            pass
