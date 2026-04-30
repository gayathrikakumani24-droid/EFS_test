import json
import os
import re
import tempfile
import zlib
from collections import defaultdict
from itertools import groupby
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from llama_parse import LlamaParse


load_dotenv()

LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


st.set_page_config(page_title="Spec Intelligence Optimized", layout="wide")
st.title("Spec Intelligence - Optimized Semantic Extraction")
st.caption("Document to Markdown + grounded semantic JSON + PlantUML diagrams")


SEMANTIC_TEMPLATE: dict[str, Any] = {
    "document_metadata": {
        "document_name": "extracted_spec",
        "protocol": "unknown",
        "version": "",
        "sections_detected": 0,
        "extraction_quality": {
            "signal_count": 0,
            "register_count": 0,
            "transaction_count": 0,
            "warnings": [],
        },
    },
    "signals": [],
    "registers": [],
    "transactions": [],
    "constraints": [],
    "timing_conditions": [],
}


PROTOCOL_HINTS = {
    "AXI": ["AWVALID", "AWREADY", "WVALID", "WREADY", "BVALID", "BREADY", "ARVALID", "RVALID", "RREADY"],
    "AXI4-Stream": ["TVALID", "TREADY", "TDATA", "TLAST", "TKEEP", "TSTRB"],
    "APB": ["PSEL", "PENABLE", "PREADY", "PWRITE", "PWDATA", "PRDATA"],
    "AHB": ["HADDR", "HTRANS", "HWRITE", "HREADY", "HRESP", "HWDATA", "HRDATA"],
    "CHI": ["REQ", "RSP", "DAT", "SNP", "TXREQ", "RXRSP", "TXDAT", "RXDAT"],
    "PCIe": ["TLP", "DLLP", "LTSSM", "CFG", "MSI", "MSI-X", "FLIT"],
    "CXL": ["CXL.io", "CXL.cache", "CXL.mem", "FLIT", "HDM", "BI"],
    "USB": ["DP", "DM", "VBUS", "D+", "D-", "USB"],
    "Ethernet": ["TXD", "RXD", "TX_EN", "RX_DV", "MDC", "MDIO", "XGMII", "RGMII"],
}


def groq_client() -> Groq:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set")
    return Groq(api_key=GROQ_API_KEY)


def call_json_llm(system: str, user: str, model: str, max_tokens: int = 4096) -> dict:
    response = groq_client().chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = response.choices[0].message.content or "{}"
    return safe_json_loads(content)


def safe_json_loads(text: str) -> dict:
    text = strip_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group(0))
        raise


def strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json|text)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def clean_text(value: Any, max_len: int = 300) -> str:
    value = "" if value is None else str(value)
    value = re.sub(r"\s+", " ", value).strip()
    return value[:max_len]


def canonical_name(value: Any, default: str = "UNKNOWN") -> str:
    text = clean_text(value, 120)
    text = text.strip("`'\" ")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_\-\[\]:/.+]", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or default


def puml_name(value: Any, default: str = "UNKNOWN") -> str:
    text = canonical_name(value, default)
    text = re.sub(r"\W+", "_", text).strip("_")
    if not text:
        text = default
    if re.match(r"^\d", text):
        text = f"N_{text}"
    return text[:50]


def normalize_direction(value: Any, name: str = "") -> str:
    text = clean_text(value, 40).lower()
    name_l = name.lower()
    if text in {"input", "in", "rx", "sink", "target", "slave", "subordinate", "sampled"}:
        return "input"
    if text in {"output", "out", "tx", "source", "master", "manager", "initiator", "driven"}:
        return "output"
    if text in {"inout", "bidirectional", "bidir", "io"}:
        return "inout"
    if name_l.startswith(("rx", "rxd")):
        return "input"
    if name_l.startswith(("tx", "txd")):
        return "output"
    return "unknown"


def normalize_width(value: Any) -> str:
    text = clean_text(value, 40)
    if not text:
        return "1"
    bit_range = re.search(r"\[?\s*(\d+)\s*:\s*(\d+)\s*\]?", text)
    if bit_range:
        hi, lo = int(bit_range.group(1)), int(bit_range.group(2))
        return str(abs(hi - lo) + 1)
    single_bit = re.search(r"\[\s*(\d+)\s*\]", text)
    if single_bit:
        return "1"
    number = re.search(r"\b(\d+)\b", text)
    return number.group(1) if number else text[:20]


def is_active_low_signal(signal_name: Any) -> bool:
    name = canonical_name(signal_name).lower()
    if re.search(r"(^|_)(resetn|rstn|aresetn)$", name):
        return True
    if re.search(r"(^|_)(reset|rst|clear|clr|enable|en|valid|ready|req|ack)(_?n|_?b)$", name):
        return True
    if re.search(r"(^|_)(reset|rst|clear|clr|enable|en|valid|ready|req|ack)_?bar$", name):
        return True
    return False


def normalize_signal_value(value: Any, signal_name: Any = "") -> str | None:
    """
    Convert prose signal states to physical binary values.

    Physical level words always map directly:
      high/1/true -> 1, low/0/false -> 0

    Logical assertion words respect active-low naming:
      ARESETn asserted -> 0, ARESETn deasserted -> 1
      AWVALID asserted -> 1, AWVALID deasserted -> 0
    """
    text = clean_text(value, 160).lower()
    if not text:
        return None

    active_low = is_active_low_signal(signal_name)

    if re.search(r"(?<!\w)(1|one|high|hi|true|logic\s*1|driven\s*high|goes\s*high|is\s*high)(?!\w)", text):
        return "1"
    if re.search(r"(?<!\w)(0|zero|low|lo|false|logic\s*0|driven\s*low|goes\s*low|is\s*low)(?!\w)", text):
        return "0"

    if re.search(r"\b(deasserted|de-asserted|deassert|de-assert|inactive|negated|cleared|clear)\b", text):
        return "1" if active_low else "0"
    if re.search(r"\b(asserted|assert|active|set|enabled|enable)\b", text):
        return "0" if active_low else "1"

    assignment = re.search(r"(?:=|==|is)\s*([01])\b", text)
    if assignment:
        return assignment.group(1)

    return None


def normalize_access(value: Any) -> str:
    text = clean_text(value, 20).upper().replace(" ", "")
    aliases = {
        "R/W": "RW",
        "READWRITE": "RW",
        "READONLY": "RO",
        "WRITEONLY": "WO",
        "W1C": "W1C",
        "RW1C": "RW1C",
    }
    return aliases.get(text, text or "RW")


def clone_template() -> dict:
    return json.loads(json.dumps(SEMANTIC_TEMPLATE))


@st.cache_data(show_spinner=False)
def parse_document(file_path: str) -> str:
    if not LLAMA_CLOUD_API_KEY:
        raise ValueError("LLAMA_CLOUD_API_KEY is not set")
    parser = LlamaParse(api_key=LLAMA_CLOUD_API_KEY, result_type="markdown", verbose=True)
    docs = parser.load_data(file_path)
    if not docs:
        raise ValueError("No documents returned from LlamaParse")
    markdown = "\n\n".join(clean_text(getattr(doc, "text", ""), 1_000_000) for doc in docs)
    if not markdown.strip():
        raise ValueError("Empty markdown returned from LlamaParse")
    return markdown


def split_markdown_sections(markdown: str, target_size: int = 9000, overlap: int = 900) -> list[dict]:
    heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.M)
    matches = list(heading_re.finditer(markdown))
    if not matches:
        return sliding_chunks(markdown, target_size, overlap)

    sections = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown)
        title = clean_text(match.group(2), 120)
        body = markdown[start:end].strip()
        if body:
            sections.append({"section_ref": title, "text": body})

    chunks = []
    current_title = []
    current_text = ""
    for section in sections:
        addition = section["text"]
        if current_text and len(current_text) + len(addition) > target_size:
            chunks.append({"section_ref": " / ".join(current_title), "text": current_text})
            current_text = current_text[-overlap:] + "\n\n" + addition
            current_title = [section["section_ref"]]
        else:
            current_text = f"{current_text}\n\n{addition}".strip()
            current_title.append(section["section_ref"])
    if current_text:
        chunks.append({"section_ref": " / ".join(current_title), "text": current_text})
    return chunks


def sliding_chunks(text: str, target_size: int, overlap: int) -> list[dict]:
    chunks = []
    step = max(1000, target_size - overlap)
    for start in range(0, len(text), step):
        chunk = text[start : start + target_size]
        if chunk.strip():
            chunks.append({"section_ref": f"chars_{start}_{start + len(chunk)}", "text": chunk})
    return chunks


def parse_markdown_tables(markdown: str) -> list[dict]:
    tables = []
    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        if "|" not in lines[i]:
            i += 1
            continue
        if i + 1 >= len(lines) or not re.match(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", lines[i + 1]):
            i += 1
            continue
        header = split_table_row(lines[i])
        rows = []
        i += 2
        while i < len(lines) and "|" in lines[i] and lines[i].strip():
            row = split_table_row(lines[i])
            if len(row) == len(header):
                rows.append(dict(zip(header, row)))
            i += 1
        if rows:
            tables.append({"headers": header, "rows": rows})
    return tables


def split_table_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [clean_text(cell, 500) for cell in line.split("|")]


def deterministic_extract(markdown: str) -> dict:
    result = clone_template()
    tables = parse_markdown_tables(markdown)
    result["signals"].extend(extract_signals_from_tables(tables))
    result["registers"].extend(extract_registers_from_tables(tables, markdown))
    result["constraints"].extend(extract_constraints_regex(markdown))
    result["timing_conditions"].extend(extract_timing_regex(markdown))
    protocol = infer_protocol(markdown, result["signals"])
    result["document_metadata"]["protocol"] = protocol
    return result


def header_lookup(row: dict, *names: str) -> str:
    lowered = {k.lower().strip(): v for k, v in row.items()}
    for name in names:
        for key, value in lowered.items():
            if name in key:
                return value
    return ""


def table_kind(headers: list[str]) -> str:
    text = " ".join(h.lower() for h in headers)
    if any(x in text for x in ["signal", "port", "pin"]) and any(x in text for x in ["direction", "dir", "i/o", "type"]):
        return "signals"
    if any(x in text for x in ["field", "bit", "bits"]) and any(x in text for x in ["reset", "access", "description"]):
        return "fields"
    if any(x in text for x in ["register", "address", "offset"]) and not any(x in text for x in ["field", "bits"]):
        return "registers"
    return "unknown"


def extract_signals_from_tables(tables: list[dict]) -> list[dict]:
    signals = []
    for table in tables:
        if table_kind(table["headers"]) != "signals":
            continue
        for row in table["rows"]:
            name = header_lookup(row, "signal", "port", "pin", "name")
            if not is_signal_like(name):
                continue
            signals.append(
                {
                    "name": canonical_name(name),
                    "direction": normalize_direction(header_lookup(row, "direction", "dir", "i/o", "type"), name),
                    "width": normalize_width(header_lookup(row, "width", "bits", "bit")),
                    "description": header_lookup(row, "description", "function", "purpose", "definition"),
                    "section_ref": "",
                    "source": "table",
                    "confidence": 0.9,
                }
            )
    return signals


def is_signal_like(name: Any) -> bool:
    text = clean_text(name, 120)
    if not text or len(text) > 80:
        return False
    if re.search(r"\s{2,}", text):
        return False
    if re.search(r"\b(the|shall|must|when|if|transaction|register)\b", text, re.I):
        return False
    return bool(re.search(r"[A-Za-z_][A-Za-z0-9_\[\]:]*", text))


def extract_registers_from_tables(tables: list[dict], markdown: str) -> list[dict]:
    registers = []
    field_tables = [t for t in tables if table_kind(t["headers"]) == "fields"]
    register_tables = [t for t in tables if table_kind(t["headers"]) == "registers"]

    for table in register_tables:
        for row in table["rows"]:
            name = header_lookup(row, "register", "name")
            if not name:
                continue
            registers.append(
                {
                    "name": canonical_name(name),
                    "address": clean_text(header_lookup(row, "address", "offset"), 40),
                    "description": header_lookup(row, "description", "function", "purpose"),
                    "fields": [],
                    "section_ref": "",
                    "source": "table",
                    "confidence": 0.85,
                }
            )

    current_register = infer_register_headings(markdown)
    for idx, table in enumerate(field_tables):
        fields = []
        for row in table["rows"]:
            fname = header_lookup(row, "field", "name")
            if not fname:
                continue
            fields.append(
                {
                    "name": canonical_name(fname),
                    "bits": clean_text(header_lookup(row, "bits", "bit"), 30),
                    "access": normalize_access(header_lookup(row, "access", "type")),
                    "reset_value": clean_text(header_lookup(row, "reset", "default"), 40) or "0x0",
                    "description": header_lookup(row, "description", "function", "purpose"),
                }
            )
        if fields:
            heading = current_register[idx] if idx < len(current_register) else f"REGISTER_{idx + 1}"
            registers.append(
                {
                    "name": heading["name"],
                    "address": heading["address"],
                    "description": heading["description"],
                    "fields": fields,
                    "section_ref": heading["section_ref"],
                    "source": "field_table",
                    "confidence": 0.75,
                }
            )
    return registers


def infer_register_headings(markdown: str) -> list[dict]:
    pattern = re.compile(
        r"(?im)^#{1,6}\s*(?:register\s*)?([A-Za-z][A-Za-z0-9_./-]{1,60})"
        r"(?:\s*[-:]\s*(.*?))?(?:\s*@\s*(0x[0-9A-Fa-f_]+))?\s*$"
    )
    headings = []
    for match in pattern.finditer(markdown):
        title = clean_text(match.group(1), 80)
        if not re.search(r"(reg|ctrl|status|cfg|config|command|addr|data|mask|enable|intr|int)", title, re.I):
            continue
        headings.append(
            {
                "name": canonical_name(title),
                "address": clean_text(match.group(3) or "", 40),
                "description": clean_text(match.group(2) or "", 160),
                "section_ref": title,
            }
        )
    return headings


def extract_constraints_regex(markdown: str) -> list[dict]:
    constraints = []
    sentence_re = re.compile(r"([^.\n]*(?:MUST|SHALL|REQUIRED|SHOULD|MAY|must|shall|required|should|may)[^.\n]*[.\n])")
    for match in sentence_re.finditer(markdown):
        sentence = clean_text(match.group(1), 400)
        if len(sentence) < 12:
            continue
        req_type = re.search(r"\b(MUST|SHALL|REQUIRED|SHOULD|MAY)\b", sentence, re.I)
        constraints.append(
            {
                "rule": sentence,
                "type": req_type.group(1).upper() if req_type else "REQUIREMENT",
                "condition": "",
                "section_ref": nearest_heading(markdown, match.start()),
                "source": "regex",
                "confidence": 0.8,
            }
        )
    return constraints[:300]


def extract_timing_regex(markdown: str) -> list[dict]:
    timing = []
    patterns = [
        r"when\s+(.{3,80}?)\s*,?\s*(.{3,120}?\s+(?:within|after|before|until)\s+.{3,80}?)[.\n]",
        r"if\s+(.{3,80}?)\s*,?\s*(.{3,120}?\s+(?:within|after|before|until)\s+.{3,80}?)[.\n]",
        r"(.{3,60}?\s+asserted).{0,40}?\s+(within|after)\s+(.{3,80}?)[.\n]",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, markdown, re.I):
            if len(match.groups()) >= 3 and match.group(2).lower() in {"within", "after"}:
                trigger = match.group(1)
                response = f"{match.group(2)} {match.group(3)}"
            else:
                trigger = match.group(1)
                response = match.group(2)
            timing.append(
                {
                    "trigger": clean_text(trigger, 160),
                    "response": clean_text(response, 160),
                    "section_ref": nearest_heading(markdown, match.start()),
                    "source": "regex",
                    "confidence": 0.65,
                }
            )
    return timing[:120]


def nearest_heading(markdown: str, pos: int) -> str:
    prefix = markdown[:pos]
    matches = list(re.finditer(r"^#{1,6}\s+(.+?)\s*$", prefix, re.M))
    return clean_text(matches[-1].group(1), 120) if matches else ""


def infer_protocol(markdown: str, signals: list[dict]) -> str:
    haystack = markdown[:50000] + " " + " ".join(s.get("name", "") for s in signals)
    scores = {}
    for protocol, hints in PROTOCOL_HINTS.items():
        scores[protocol] = sum(1 for hint in hints if re.search(rf"(?<![A-Za-z0-9_]){re.escape(hint)}(?![A-Za-z0-9_])", haystack, re.I))
    best, score = max(scores.items(), key=lambda x: x[1])
    return best if score >= 2 else "unknown"


EXTRACTION_SYSTEM = """
You extract semiconductor specification data into strict JSON.

Rules:
- Extract only items explicitly present in the snippet.
- Do not place registers in signals.
- Do not place signal rows in registers.
- Do not turn prose headings into signal names.
- Preserve exact signal/register names.
- For transactions, include only protocol flows or bus operations, not register fields.
- Every item must include section_ref and evidence.
- If unsure, omit the item.

Return JSON with exactly these top-level keys:
document_metadata, signals, registers, transactions, constraints, timing_conditions.

Schema:
{
  "document_metadata": {"document_name":"","protocol":"","version":"","sections_detected":0},
  "signals": [{"name":"","direction":"input|output|inout|unknown","width":"1","description":"","section_ref":"","evidence":""}],
  "registers": [{"name":"","address":"","description":"","section_ref":"","evidence":"","fields":[{"name":"","bits":"","access":"RW|RO|WO|W1C|RW1C","reset_value":"0x0","description":""}]}],
  "transactions": [{"name":"","type":"","initiator":"","target":"","actors":[],"sequence_steps":[{"index":1,"actor":"","action":"","signal_refs":[],"condition":"","evidence":""}],"section_ref":"","evidence":""}],
  "constraints": [{"rule":"","type":"MUST|SHALL|REQUIRED|SHOULD|MAY","condition":"","section_ref":"","evidence":""}],
  "timing_conditions": [{"trigger":"","response":"","cycles":"","section_ref":"","evidence":""}]
}
"""


def llm_extract_chunk(chunk: dict, model: str) -> dict:
    section = chunk["section_ref"]
    text = chunk["text"]
    user = f"Section reference: {section}\n\nSpecification markdown snippet:\n{text}"
    return call_json_llm(EXTRACTION_SYSTEM, user, model=model, max_tokens=6000)


def extract_semantic_json(markdown_text: str, extraction_mode: str, chunk_size: int, model: str) -> dict | None:
    if not markdown_text.strip():
        st.error("No markdown content to process.")
        return None

    deterministic = deterministic_extract(markdown_text)
    chunks = split_markdown_sections(markdown_text, target_size=chunk_size, overlap=max(500, chunk_size // 10))
    if extraction_mode == "fast":
        chunks = chunks[: min(4, len(chunks))]

    merged = clone_template()
    merge_semantic_json(merged, deterministic)

    progress = st.progress(0)
    status = st.empty()
    for idx, chunk in enumerate(chunks, 1):
        status.write(f"Extracting chunk {idx}/{len(chunks)}: {chunk['section_ref'][:80]}")
        try:
            chunk_json = llm_extract_chunk(chunk, model)
            tag_missing_section_refs(chunk_json, chunk["section_ref"])
            merge_semantic_json(merged, chunk_json)
        except Exception as exc:
            warnings = merged["document_metadata"]["extraction_quality"]["warnings"]
            warnings.append(f"Chunk {idx} failed: {exc}")
        progress.progress(idx / len(chunks))

    merged["document_metadata"]["sections_detected"] = len(chunks)
    merged = normalize_and_validate(merged, markdown_text)
    merged = align_transactions_to_signals(merged)
    merged = align_timing_to_signals(merged)
    update_quality_metrics(merged)
    return merged


def tag_missing_section_refs(data: dict, section_ref: str) -> None:
    for key in ["signals", "registers", "transactions", "constraints", "timing_conditions"]:
        for item in data.get(key, []) or []:
            if isinstance(item, dict) and not item.get("section_ref"):
                item["section_ref"] = section_ref


def merge_semantic_json(base: dict, incoming: dict) -> None:
    meta = incoming.get("document_metadata", {}) if isinstance(incoming, dict) else {}
    if meta.get("protocol") and meta["protocol"].lower() != "unknown":
        base["document_metadata"]["protocol"] = meta["protocol"]
    if meta.get("version"):
        base["document_metadata"]["version"] = meta["version"]
    for key in ["signals", "registers", "transactions", "constraints", "timing_conditions"]:
        items = incoming.get(key, []) if isinstance(incoming, dict) else []
        if isinstance(items, list):
            base[key].extend(x for x in items if isinstance(x, dict))


def normalize_signal(item: dict) -> dict | None:
    name = canonical_name(item.get("name", ""))
    if not is_signal_like(name):
        return None
    return {
        "name": name,
        "direction": normalize_direction(item.get("direction", ""), name),
        "width": normalize_width(item.get("width", "1")),
        "description": clean_text(item.get("description", ""), 300),
        "section_ref": clean_text(item.get("section_ref", ""), 120),
        "evidence": clean_text(item.get("evidence", ""), 300),
        "source": clean_text(item.get("source", "llm"), 40),
        "confidence": float(item.get("confidence", 0.7) or 0.7),
    }


def normalize_register(item: dict) -> dict | None:
    name = canonical_name(item.get("name", ""))
    if not name or name == "UNKNOWN":
        return None
    fields = []
    for field in item.get("fields", []) or []:
        if not isinstance(field, dict):
            continue
        fname = canonical_name(field.get("name", ""))
        if not fname:
            continue
        fields.append(
            {
                "name": fname,
                "bits": clean_text(field.get("bits", ""), 30),
                "access": normalize_access(field.get("access", "RW")),
                "reset_value": clean_text(field.get("reset_value", "0x0"), 40) or "0x0",
                "description": clean_text(field.get("description", ""), 300),
            }
        )
    return {
        "name": name,
        "address": clean_text(item.get("address", ""), 50),
        "description": clean_text(item.get("description", ""), 300),
        "section_ref": clean_text(item.get("section_ref", ""), 120),
        "evidence": clean_text(item.get("evidence", ""), 300),
        "fields": fields,
        "source": clean_text(item.get("source", "llm"), 40),
        "confidence": float(item.get("confidence", 0.7) or 0.7),
    }


def normalize_transaction(item: dict) -> dict | None:
    name = clean_text(item.get("name", ""), 100)
    if not name:
        return None
    actors = []
    for actor in item.get("actors", []) or []:
        actor_name = actor.get("name") if isinstance(actor, dict) else actor
        actor_name = puml_name(actor_name, "")
        if actor_name and actor_name not in actors:
            actors.append(actor_name)
    for actor in [item.get("initiator", ""), item.get("target", "")]:
        actor_name = puml_name(actor, "")
        if actor_name and actor_name not in actors:
            actors.append(actor_name)
    steps = []
    for idx, step in enumerate(item.get("sequence_steps", []) or [], 1):
        if isinstance(step, str):
            step = {"index": idx, "action": step}
        if not isinstance(step, dict):
            continue
        action = clean_text(step.get("action") or step.get("step") or step.get("description"), 220)
        if not action:
            continue
        signal_refs = step.get("signal_refs", []) or []
        if isinstance(signal_refs, str):
            signal_refs = [signal_refs]
        steps.append(
            {
                "index": int(step.get("index", idx) or idx),
                "actor": puml_name(step.get("actor", ""), ""),
                "action": action,
                "signal_refs": [canonical_name(s) for s in signal_refs if clean_text(s)],
                "condition": clean_text(step.get("condition", ""), 160),
                "evidence": clean_text(step.get("evidence", ""), 260),
            }
        )
    return {
        "name": name,
        "type": clean_text(item.get("type", ""), 60),
        "initiator": puml_name(item.get("initiator", actors[0] if actors else "Initiator")),
        "target": puml_name(item.get("target", actors[-1] if len(actors) > 1 else "Target")),
        "actors": actors or ["Initiator", "Target"],
        "sequence_steps": sorted(steps, key=lambda s: s["index"]),
        "section_ref": clean_text(item.get("section_ref", ""), 120),
        "evidence": clean_text(item.get("evidence", ""), 300),
    }


def normalize_constraint(item: dict) -> dict | None:
    rule = clean_text(item.get("rule", ""), 500)
    if not rule:
        return None
    req_type = clean_text(item.get("type", ""), 30).upper()
    if req_type not in {"MUST", "SHALL", "REQUIRED", "SHOULD", "MAY"}:
        found = re.search(r"\b(MUST|SHALL|REQUIRED|SHOULD|MAY)\b", rule, re.I)
        req_type = found.group(1).upper() if found else "REQUIREMENT"
    return {
        "rule": rule,
        "type": req_type,
        "condition": clean_text(item.get("condition", ""), 250),
        "section_ref": clean_text(item.get("section_ref", ""), 120),
        "evidence": clean_text(item.get("evidence", ""), 300),
    }


def normalize_timing(item: dict) -> dict | None:
    trigger = clean_text(item.get("trigger", ""), 200)
    response = clean_text(item.get("response", ""), 200)
    if not trigger or not response:
        return None
    return {
        "trigger": trigger,
        "response": response,
        "cycles": clean_text(item.get("cycles", ""), 40),
        "section_ref": clean_text(item.get("section_ref", ""), 120),
        "evidence": clean_text(item.get("evidence", ""), 300),
    }


def normalize_and_validate(data: dict, markdown: str) -> dict:
    normalized = clone_template()
    normalized["document_metadata"].update(data.get("document_metadata", {}))
    normalized["signals"] = dedupe_items(filter_none(normalize_signal(x) for x in data.get("signals", [])), signal_key, richer_signal)
    normalized["registers"] = dedupe_items(filter_none(normalize_register(x) for x in data.get("registers", [])), register_key, richer_register)
    normalized["transactions"] = dedupe_items(filter_none(normalize_transaction(x) for x in data.get("transactions", [])), lambda x: x["name"].lower(), richer_transaction)
    normalized["constraints"] = dedupe_items(filter_none(normalize_constraint(x) for x in data.get("constraints", [])), lambda x: (x["rule"][:120].lower(), x["condition"][:80].lower()), richer_text_item)
    normalized["timing_conditions"] = dedupe_items(filter_none(normalize_timing(x) for x in data.get("timing_conditions", [])), lambda x: (x["trigger"][:80].lower(), x["response"][:80].lower()), richer_text_item)

    remove_cross_category_noise(normalized)
    if normalized["document_metadata"].get("protocol", "unknown").lower() == "unknown":
        normalized["document_metadata"]["protocol"] = infer_protocol(markdown, normalized["signals"])
    return normalized


def filter_none(items):
    return [x for x in items if x is not None]


def dedupe_items(items: list[dict], key_fn, choose_fn) -> list[dict]:
    seen = {}
    for item in items:
        key = key_fn(item)
        if key not in seen:
            seen[key] = item
        else:
            seen[key] = choose_fn(seen[key], item)
    return list(seen.values())


def signal_key(item: dict) -> str:
    return re.sub(r"^(AXI|APB|AHB|CHI|PCIE|CXL)_", "", item["name"].upper())


def register_key(item: dict) -> tuple:
    address = item.get("address", "").lower()
    return (address or item["name"].lower(), item["name"].lower())


def info_score(item: dict) -> int:
    return sum(len(str(v)) for v in item.values() if not isinstance(v, list)) + sum(len(str(v)) for v in item.get("fields", []))


def richer_signal(a: dict, b: dict) -> dict:
    chosen = b if info_score(b) > info_score(a) else a
    other = a if chosen is b else b
    if chosen["direction"] == "unknown" and other["direction"] != "unknown":
        chosen["direction"] = other["direction"]
    if chosen["width"] == "1" and other["width"] != "1":
        chosen["width"] = other["width"]
    return chosen


def richer_register(a: dict, b: dict) -> dict:
    chosen = b if len(b.get("fields", [])) > len(a.get("fields", [])) else a
    other = a if chosen is b else b
    if not chosen.get("address") and other.get("address"):
        chosen["address"] = other["address"]
    return chosen


def richer_transaction(a: dict, b: dict) -> dict:
    return b if len(b.get("sequence_steps", [])) > len(a.get("sequence_steps", [])) else a


def richer_text_item(a: dict, b: dict) -> dict:
    return b if info_score(b) > info_score(a) else a


def remove_cross_category_noise(data: dict) -> None:
    register_names = {r["name"].upper() for r in data["registers"]}
    field_names = {f["name"].upper() for r in data["registers"] for f in r.get("fields", [])}
    data["signals"] = [
        s for s in data["signals"]
        if s["name"].upper() not in register_names and not looks_like_register_only(s["name"], field_names)
    ]


def looks_like_register_only(name: str, field_names: set[str]) -> bool:
    upper = name.upper()
    if upper in field_names and not re.search(r"(CLK|RST|VALID|READY|DATA|ADDR|REQ|ACK|TX|RX)", upper):
        return True
    return bool(re.search(r"(REGISTER|REG_FIELD|RESERVED)$", upper))


def align_transactions_to_signals(data: dict) -> dict:
    signal_names = [s["name"] for s in data["signals"]]
    signal_index = {s.upper(): s for s in signal_names}
    for txn in data["transactions"]:
        for step in txn.get("sequence_steps", []):
            refs = set(step.get("signal_refs", []))
            step_text = " ".join([step.get("action", ""), step.get("condition", ""), step.get("evidence", "")])
            for name in signal_names:
                if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", step_text, re.I):
                    refs.add(signal_index[name.upper()])
            step["signal_refs"] = sorted(refs)
    return data


def align_timing_to_signals(data: dict) -> dict:
    signal_names = [s["name"] for s in data["signals"]]
    for timing in data["timing_conditions"]:
        refs = []
        text = f"{timing.get('trigger', '')} {timing.get('response', '')}"
        for name in signal_names:
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text, re.I):
                refs.append(name)
        timing["signal_refs"] = refs
    return data


def update_quality_metrics(data: dict) -> None:
    quality = data["document_metadata"].setdefault("extraction_quality", {})
    quality["signal_count"] = len(data["signals"])
    quality["register_count"] = len(data["registers"])
    quality["transaction_count"] = len(data["transactions"])
    warnings = quality.setdefault("warnings", [])
    if not data["signals"]:
        warnings.append("No signals found. Check whether LlamaParse preserved signal tables.")
    if not data["transactions"]:
        warnings.append("No transactions found. Try Deep mode or a larger chunk size.")
    unaligned = sum(1 for t in data["transactions"] for s in t.get("sequence_steps", []) if not s.get("signal_refs"))
    if unaligned:
        warnings.append(f"{unaligned} transaction steps have no matched signal_refs.")


def detect_protocol(markdown_text: str, model: str) -> dict:
    deterministic_protocol = infer_protocol(markdown_text, [])
    system = "You identify semiconductor protocol specs. Return only JSON."
    user = f"""
Identify the primary protocol from this markdown. Prefer exact evidence.

Allowed examples: AXI, AXI4, AXI4-Stream, APB, AHB, CHI, PCIe, CXL, USB, Ethernet, Proprietary, Unknown.

Return:
{{"protocol":"","confidence":"high|medium|low","version":"","signals_found":[],"reasoning":""}}

Deterministic hint: {deterministic_protocol}

Markdown:
{markdown_text[:14000]}
"""
    try:
        result = call_json_llm(system, user, model=model, max_tokens=1500)
    except Exception:
        result = {}
    if not result.get("protocol"):
        result["protocol"] = deterministic_protocol
    result.setdefault("confidence", "low")
    result.setdefault("version", "")
    result.setdefault("signals_found", [])
    result.setdefault("reasoning", "")
    return result


def safe_label(text: Any, max_len: int = 80) -> str:
    value = clean_text(text, max_len)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r'[\"\'`<>{}|\\]', "", value)
    return value[:max_len] or "event"


def seed_sequence(data: dict) -> dict:
    actors = []
    messages = []
    for txn in data.get("transactions", [])[:12]:
        tx_actors = txn.get("actors", []) or [txn.get("initiator", "Initiator"), txn.get("target", "Target")]
        for actor in tx_actors:
            actor = puml_name(actor)
            if actor not in actors:
                actors.append(actor)
        if len(tx_actors) < 2:
            tx_actors = ["Initiator", "Target"]
        previous_actor = puml_name(tx_actors[0])
        target_actor = puml_name(tx_actors[-1])
        messages.append({"type": "divider", "label": txn["name"]})
        for step in txn.get("sequence_steps", [])[:12]:
            label = " / ".join(step.get("signal_refs", [])[:3]) or step.get("action", "event")
            actor = puml_name(step.get("actor") or previous_actor)
            to_actor = target_actor if actor != target_actor else puml_name(tx_actors[0])
            mtype = "return" if is_response_step(step.get("action", "")) else "sync"
            messages.append({"from": actor, "to": to_actor, "label": label, "type": mtype})
            previous_actor = actor
            if step.get("condition"):
                messages.append({"from": actor, "to": actor, "label": step["condition"], "type": "note"})
    if not actors:
        actors = ["Initiator", "Target"]
        for sig in data.get("signals", [])[:12]:
            frm, to = ("Initiator", "Target") if sig["direction"] != "input" else ("Target", "Initiator")
            messages.append({"from": frm, "to": to, "label": sig["name"], "type": "sync"})
    return {"actors": actors[:6], "messages": messages[:80]}


def is_response_step(text: str) -> bool:
    return bool(re.search(r"\b(response|respond|ready|ack|complete|grant|accept|return|rvalid|bvalid)\b", text, re.I))


def build_sequence_puml(content: dict) -> str:
    actors = [puml_name(a) for a in content.get("actors", ["Initiator", "Target"])]
    lines = [
        "@startuml",
        "skinparam sequenceMessageAlign center",
        "skinparam sequenceArrowThickness 2",
        "skinparam roundcorner 6",
        "",
    ]
    for actor in actors:
        lines.append(f"participant {actor}")
    lines.append("")
    seen = set()
    for msg in content.get("messages", []):
        mtype = msg.get("type", "sync")
        if mtype == "divider":
            lines.append(f"== {safe_label(msg.get('label', 'Phase'), 80)} ==")
            continue
        frm = puml_name(msg.get("from", actors[0]))
        to = puml_name(msg.get("to", actors[-1]))
        label = safe_label(msg.get("label", "event"), 120)
        key = (frm, to, label, mtype)
        if key in seen:
            continue
        seen.add(key)
        if mtype == "note":
            over = frm if frm == to else f"{frm},{to}"
            lines.append(f"note over {over}: {label}")
        elif mtype == "return":
            lines.append(f"{frm} --> {to}: {label}")
        else:
            lines.append(f"{frm} -> {to}: {label}")
    lines.append("@enduml")
    return "\n".join(lines)


def seed_fsm(data: dict) -> dict:
    states = ["IDLE"]
    transitions = []
    for txn in data.get("transactions", [])[:8]:
        prev = states[-1]
        for step in txn.get("sequence_steps", [])[:8]:
            label = puml_name(step.get("action", "ACTIVE"))[:24].upper()
            state = label or "ACTIVE"
            if state not in states:
                states.append(state)
            transitions.append({"from": prev, "to": state, "label": safe_label(step.get("action", ""), 40)})
            prev = state
    if len(states) == 1:
        states.extend(["ACTIVE", "COMPLETE"])
        transitions.extend([
            {"from": "IDLE", "to": "ACTIVE", "label": "start"},
            {"from": "ACTIVE", "to": "COMPLETE", "label": "done"},
            {"from": "COMPLETE", "to": "IDLE", "label": "reset"},
        ])
    if "ERROR" not in states:
        states.append("ERROR")
    transitions.append({"from": states[1], "to": "ERROR", "label": "timeout"})
    transitions.append({"from": "ERROR", "to": "IDLE", "label": "reset"})
    return {"initial_state": "IDLE", "states": states[:20], "transitions": transitions[:60]}


def build_fsm_puml(content: dict) -> str:
    states = [puml_name(s) for s in content.get("states", ["IDLE", "ACTIVE", "ERROR"])]
    initial = puml_name(content.get("initial_state", states[0]))
    lines = ["@startuml", "skinparam state {", "  RoundCorner 6", "}", ""]
    for state in states:
        lines.append(f"state {state} #742a2a" if state == "ERROR" else f"state {state}")
    lines.append("")
    lines.append(f"[*] --> {initial}")
    for trans in content.get("transitions", []):
        frm = puml_name(trans.get("from", "IDLE"))
        to = puml_name(trans.get("to", "IDLE"))
        label = safe_label(trans.get("label", ""), 50)
        lines.append(f"{frm} --> {to} : {label}" if label else f"{frm} --> {to}")
    lines.append("@enduml")
    return "\n".join(lines)


def seed_timing(data: dict) -> dict:
    priority = ["clk", "clock", "reset", "rst", "valid", "ready", "enable", "data", "addr", "req", "ack"]

    def score(signal):
        name = signal["name"].lower()
        for idx, word in enumerate(priority):
            if word in name:
                return idx
        return 99

    signals = sorted(data.get("signals", []), key=score)[:8]
    if not signals:
        signals = [{"name": "CLK", "direction": "input"}, {"name": "VALID", "direction": "output"}, {"name": "READY", "direction": "input"}]

    timing_conditions = data.get("timing_conditions", [])
    result = []
    for sig in signals:
        name = puml_name(sig["name"])
        binary = bool(re.search(r"(clk|clock|reset|rst|valid|ready|enable|ack|req|grant|irq|sel|last)", name, re.I))
        events = infer_binary_events_from_timing(name, timing_conditions) if binary else []
        if not events:
            active_value = "0" if is_active_low_signal(name) else "1"
            inactive_value = "1" if is_active_low_signal(name) else "0"
            events = [
                {"time": 0, "state": inactive_value},
                {"time": 20, "state": active_value},
                {"time": 60, "state": inactive_value},
            ]
        if not binary:
            events = infer_concise_events_from_timing(name, timing_conditions)
            if not events:
                events = [{"time": 0, "state": "IDLE"}, {"time": 20, "state": "ACTIVE"}, {"time": 60, "state": "IDLE"}]
        result.append({"name": name, "type": "binary" if binary else "concise", "events": events})

    clock = next((puml_name(s["name"]) for s in data.get("signals", []) if re.search(r"(clk|clock)", s["name"], re.I)), "CLK")
    return {"clock_signal": clock, "clock_period": 10, "signals": result, "highlights": [{"start": 20, "end": 60, "label": "transaction"}]}


def infer_binary_events_from_timing(signal_name: str, timing_conditions: list[dict]) -> list[dict]:
    events = []
    time = 20
    for timing in timing_conditions:
        for field, offset in [("trigger", 0), ("response", 10)]:
            text = clean_text(timing.get(field, ""), 240)
            if not mentions_signal(text, signal_name):
                continue
            value = normalize_signal_value(text, signal_name)
            if value is not None:
                events.append({"time": time + offset, "state": value})
        time += 20

    if not events:
        return []

    initial = "1" if is_active_low_signal(signal_name) else "0"
    normalized = [{"time": 0, "state": initial}]
    seen_times = {0}
    for event in sorted(events, key=lambda x: x["time"]):
        if event["time"] in seen_times:
            normalized[-1]["state"] = event["state"]
            continue
        seen_times.add(event["time"])
        normalized.append(event)
    return normalized


def infer_concise_events_from_timing(signal_name: str, timing_conditions: list[dict]) -> list[dict]:
    events = []
    time = 20
    for timing in timing_conditions:
        text = f"{timing.get('trigger', '')} {timing.get('response', '')}"
        if mentions_signal(text, signal_name):
            label = "ACTIVE"
            if re.search(r"\b(idle|wait|hold|stable)\b", text, re.I):
                label = "WAIT"
            elif re.search(r"\b(data|payload|transfer)\b", text, re.I):
                label = "DATA"
            elif re.search(r"\b(done|complete|response)\b", text, re.I):
                label = "DONE"
            events.append({"time": time, "state": label})
        time += 20
    if events:
        return [{"time": 0, "state": "IDLE"}] + events
    return []


def mentions_signal(text: Any, signal_name: str) -> bool:
    raw = clean_text(text, 500)
    if re.search(rf"(?<![A-Za-z0-9_]){re.escape(signal_name)}(?![A-Za-z0-9_])", raw, re.I):
        return True
    compact_text = re.sub(r"[^A-Za-z0-9]", "", raw).lower()
    compact_signal = re.sub(r"[^A-Za-z0-9]", "", signal_name).lower()
    return bool(compact_signal and compact_signal in compact_text)


def build_timing_puml(content: dict) -> str:
    clock = puml_name(content.get("clock_signal", "CLK"))
    period = int(content.get("clock_period", 10) or 10)
    lines = ["@startuml", "", f"clock {clock} with period {period}", ""]
    declared = []
    for sig in content.get("signals", []):
        name = puml_name(sig.get("name", "SIG"))
        stype = sig.get("type", "binary")
        if stype == "binary":
            lines.append(f'robust "{name}" as {name}')
        else:
            lines.append(f'concise "{name}" as {name}')
        declared.append((name, stype, sig.get("events", [])))
    lines.append("")
    events = []
    for name, stype, sig_events in declared:
        for event in sig_events:
            state = safe_label(event.get("state", "0"), 20)
            if stype == "binary":
                state = normalize_signal_value(state, name) or ("0" if state in {"0", "1"} else "0")
            events.append((int(event.get("time", 0) or 0), name, stype, state))
    events.sort(key=lambda x: x[0])
    for time, grouped in groupby(events, key=lambda x: x[0]):
        lines.append(f"@{time}")
        for _, name, stype, state in grouped:
            lines.append(f"{name} is {state}" if stype == "binary" else f'{name} is "{state}"')
        lines.append("")
    for h in content.get("highlights", []):
        start, end = int(h.get("start", 0) or 0), int(h.get("end", 0) or 0)
        if start < end:
            lines.append(f"highlight {start} to {end} : {safe_label(h.get('label', ''), 40)}")
    lines.append("@enduml")
    return "\n".join(lines)


def plantuml_encode(puml_text: str) -> str:
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"

    def enc6(byte):
        return alphabet[byte & 0x3F]

    def enc3(b1, b2, b3):
        return (
            enc6(b1 >> 2)
            + enc6(((b1 & 3) << 4) | (b2 >> 4))
            + enc6(((b2 & 0xF) << 2) | (b3 >> 6))
            + enc6(b3)
        )

    compressor = zlib.compressobj(9, zlib.DEFLATED, -15)
    raw = compressor.compress(puml_text.encode("utf-8")) + compressor.flush()
    output = []
    for i in range(0, len(raw), 3):
        b1 = raw[i]
        b2 = raw[i + 1] if i + 1 < len(raw) else 0
        b3 = raw[i + 2] if i + 2 < len(raw) else 0
        output.append(enc3(b1, b2, b3))
    return "".join(output)


def plantuml_url(puml_text: str) -> str:
    return f"https://www.plantuml.com/plantuml/svg/{plantuml_encode(puml_text)}"


def render_plantuml(puml_code: str, height: int = 650) -> None:
    import streamlit.components.v1 as components

    url = plantuml_url(puml_code)
    html = f"""
    <html>
      <body style="margin:0;padding:12px;background:#0f1117;">
        <div style="background:white;border-radius:8px;padding:16px;overflow:auto;text-align:center;">
          <img src="{url}" style="max-width:100%;height:auto;" />
        </div>
      </body>
    </html>
    """
    components.html(html, height=height, scrolling=True)


def download_json(label: str, data: dict, filename: str) -> None:
    st.download_button(label, json.dumps(data, indent=2), file_name=filename, mime="application/json")


def sidebar_config() -> tuple[str, int, str]:
    with st.sidebar:
        st.header("Configuration")
        st.info("Requires LLAMA_CLOUD_API_KEY and GROQ_API_KEY in .env")
        mode = st.selectbox("Extraction Mode", ["deep", "fast"], index=0)
        chunk_size = st.slider("Chunk Size", 5000, 16000, 9000, 1000)
        model = st.text_input("Groq Model", value=DEFAULT_MODEL)
        st.caption("Deep mode uses all section-aware chunks. Fast mode samples the first few chunks.")
    return mode, chunk_size, model


def init_state() -> None:
    for key in [
        "markdown",
        "semantic_json",
        "protocol_result",
        "sequence_puml",
        "fsm_puml",
        "timing_puml",
    ]:
        st.session_state.setdefault(key, None)


def render_semantic_json_tab(markdown: str, mode: str, chunk_size: int, model: str) -> None:
    st.subheader("Semantic JSON Extraction")
    c1, c2 = st.columns(2)
    c1.metric("Groq key", "set" if GROQ_API_KEY else "missing")
    c2.metric("Llama key", "set" if LLAMA_CLOUD_API_KEY else "missing")

    if st.button("Run Semantic Extraction", key="semantic_btn"):
        with st.spinner("Extracting grounded semantic JSON..."):
            st.session_state.semantic_json = extract_semantic_json(markdown, mode, chunk_size, model)
            st.session_state.sequence_puml = None
            st.session_state.fsm_puml = None
            st.session_state.timing_puml = None

    data = st.session_state.semantic_json
    if not data:
        st.info("Run semantic extraction after parsing markdown.")
        return

    quality = data["document_metadata"].get("extraction_quality", {})
    cols = st.columns(5)
    cols[0].metric("Protocol", data["document_metadata"].get("protocol", "unknown"))
    cols[1].metric("Signals", len(data.get("signals", [])))
    cols[2].metric("Registers", len(data.get("registers", [])))
    cols[3].metric("Transactions", len(data.get("transactions", [])))
    cols[4].metric("Timing", len(data.get("timing_conditions", [])))

    if quality.get("warnings"):
        with st.expander("Extraction warnings", expanded=True):
            for warning in quality["warnings"]:
                st.warning(warning)

    download_json("Download semantic_json.json", data, "semantic_json.json")

    with st.expander("Signals", expanded=True):
        st.dataframe(data.get("signals", []), use_container_width=True)

    with st.expander("Registers", expanded=True):
        for reg in data.get("registers", []):
            st.markdown(f"**{reg['name']}** {reg.get('address', '')}")
            if reg.get("fields"):
                st.dataframe(reg["fields"], use_container_width=True)

    with st.expander("Transactions", expanded=True):
        st.json(data.get("transactions", []))

    with st.expander("Constraints"):
        st.dataframe(data.get("constraints", []), use_container_width=True)

    with st.expander("Full JSON"):
        st.json(data)


def render_diagrams_tab() -> None:
    data = st.session_state.semantic_json
    st.subheader("PlantUML Diagrams")
    if not data:
        st.warning("Run Semantic Extraction first.")
        return

    diagram_type = st.radio("Diagram type", ["Sequence", "FSM", "Timing", "All Three"], horizontal=True)
    if st.button("Generate Diagrams"):
        if diagram_type in {"Sequence", "All Three"}:
            st.session_state.sequence_puml = build_sequence_puml(seed_sequence(data))
        if diagram_type in {"FSM", "All Three"}:
            st.session_state.fsm_puml = build_fsm_puml(seed_fsm(data))
        if diagram_type in {"Timing", "All Three"}:
            st.session_state.timing_puml = build_timing_puml(seed_timing(data))

    if st.session_state.sequence_puml and diagram_type in {"Sequence", "All Three"}:
        st.markdown("### Sequence Diagram")
        rendered, source = st.tabs(["Rendered", "Source"])
        with rendered:
            render_plantuml(st.session_state.sequence_puml, 700)
        with source:
            st.code(st.session_state.sequence_puml, language="text")
            st.download_button("Download sequence.puml", st.session_state.sequence_puml, "sequence.puml")

    if st.session_state.fsm_puml and diagram_type in {"FSM", "All Three"}:
        st.markdown("### FSM")
        rendered, source = st.tabs(["Rendered", "Source"])
        with rendered:
            render_plantuml(st.session_state.fsm_puml, 650)
        with source:
            st.code(st.session_state.fsm_puml, language="text")
            st.download_button("Download fsm.puml", st.session_state.fsm_puml, "fsm.puml")

    if st.session_state.timing_puml and diagram_type in {"Timing", "All Three"}:
        st.markdown("### Timing")
        rendered, source = st.tabs(["Rendered", "Source"])
        with rendered:
            render_plantuml(st.session_state.timing_puml, 600)
        with source:
            st.code(st.session_state.timing_puml, language="text")
            st.download_button("Download timing.puml", st.session_state.timing_puml, "timing.puml")


def main() -> None:
    init_state()
    mode, chunk_size, model = sidebar_config()

    uploaded = st.file_uploader("Upload Spec Document", type=["pdf", "docx"])
    if uploaded:
        file_ext = os.path.splitext(uploaded.name)[1]
        st.write(f"Uploaded: **{uploaded.name}**")
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(uploaded.read())
            temp_path = tmp.name
        try:
            if st.button("Run LlamaParse"):
                with st.spinner("Parsing with LlamaParse..."):
                    st.session_state.markdown = parse_document(temp_path)
                    st.session_state.semantic_json = None
                    st.session_state.protocol_result = None
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    markdown = st.session_state.markdown
    if not markdown:
        st.info("Upload a spec and click Run LlamaParse.")
        return

    if st.session_state.protocol_result is None:
        with st.spinner("Detecting protocol..."):
            st.session_state.protocol_result = detect_protocol(markdown, model)

    tabs = st.tabs(["Rendered Markdown", "Raw Markdown", "Semantic JSON", "Diagrams", "Protocol"])
    with tabs[0]:
        st.markdown(markdown)
    with tabs[1]:
        st.code(markdown, language="markdown")
        st.download_button("Download Markdown", markdown, "spec_readme.md")
    with tabs[2]:
        render_semantic_json_tab(markdown, mode, chunk_size, model)
    with tabs[3]:
        render_diagrams_tab()
    with tabs[4]:
        st.json(st.session_state.protocol_result)


if __name__ == "__main__":
    main()
