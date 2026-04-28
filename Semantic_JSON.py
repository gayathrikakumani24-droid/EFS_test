import streamlit as st
import pandas as pd
import re
import json
from io import StringIO

# ---------------- PROTOCOL DETECTION ---------------- #

def detect_protocol(name):
    name = name.lower()

    if "valid" in name or "ready" in name:
        return "AXI"
    if "req" in name or "request" in name or "ack" in name:
        return "ASYNC"
    if "clk" in name or "clock" in name:
        return "SYNCHRONOUS"

    return "GENERIC"


# ---------------- TRANSACTION BUILDER ---------------- #

def build_transactions(events):
    transactions = []
    temp = []

    for e in events:
        if e["protocol"] == "ASYNC":
            temp.append(e)

            # complete handshake when ACK appears
            if e["name"] and "ack" in e["name"].lower():
                transactions.append({
                    "type": "transaction",
                    "protocol": "ASYNC",
                    "steps": temp.copy()
                })
                temp = []

    return transactions


# ---------------- PARSER ---------------- #

class PUMLSemanticParser:
    def __init__(self):
        self.current_condition = None
        self.current_subflow = None
        self.loop_context = None
        self.events = []

    def parse_line(self, line):
        line = line.strip()

        if not line or line.startswith("@"):
            return

        # -------- SUBFLOW -------- #
        if line.startswith("==") and line.endswith("=="):
            self.current_subflow = line.replace("=", "").strip()
            return

        # -------- CONDITIONS -------- #
        if line.startswith(("alt", "opt")):
            self.current_condition = line.split(" ", 1)[1] if " " in line else None
            return

        if line.startswith("else"):
            self.current_condition = "else"
            return

        if line == "end":
            self.current_condition = None
            self.loop_context = None
            return

        # -------- LOOP -------- #
        if line.startswith(("loop", "while")):
            self.loop_context = line
            return

        # -------- SIGNAL / MESSAGE -------- #
        match = re.match(r"(\w+)\s*->\s*(\w+)\s*:\s*(.+)", line)
        if match:
            source = match.group(1)
            destination = match.group(2)
            content = match.group(3).strip()

            # ✅ FIXED: split only once
            if "=" in content:
                parts = content.split("=", 1)
                name = parts[0].strip()
                value = parts[1].strip()
                event_type = "signal"

                # detect expression vs constant
                if any(op in value for op in ["==", "!=", "&&", "||", ">", "<"]):
                    value_type = "expression"
                else:
                    value_type = "constant"

                # try numeric conversion
                try:
                    value = int(value)
                except:
                    pass

            else:
                name = content
                value = None
                value_type = None
                event_type = "message"

            protocol = detect_protocol(name)

            event = {
                "type": event_type,
                "protocol": protocol,
                "from": source,
                "to": destination,
                "name": name,
                "value": value,
                "value_type": value_type,
                "condition": self.current_condition,
                "loop": self.loop_context,
                "subflow": self.current_subflow,
                "context": "reset" if self.current_subflow and "reset" in self.current_subflow.lower() else "normal"
            }

            self.events.append(event)

    def parse(self, text):
        for line in text.split("\n"):
            self.parse_line(line)
        return self.events


# ---------------- STREAMLIT UI ---------------- #

st.set_page_config(page_title="EFS Semantic Parser", layout="wide")

st.title("🚀 PUML → CSV + Semantic JSON (Robust + Protocol-Aware)")

uploaded_file = st.file_uploader("Upload PUML file", type=["puml"])

if uploaded_file:
    content = uploaded_file.read().decode("utf-8")

    st.subheader("📄 PUML Preview")
    st.code(content, language="text")

    # Parse
    parser = PUMLSemanticParser()
    events = parser.parse(content)

    # Transactions
    transactions = build_transactions(events)

    # ---------------- CSV ---------------- #
    df = pd.DataFrame(events)

    st.subheader("📊 CSV Output")
    st.dataframe(df)

    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    st.download_button(
        label="⬇️ Download CSV",
        data=csv_buffer.getvalue(),
        file_name="efs_output.csv",
        mime="text/csv"
    )

    # ---------------- JSON ---------------- #
    semantic_json = {
        "flow": parser.current_subflow or "main_flow",
        "total_events": len(events),
        "protocols_detected": list(set([e["protocol"] for e in events])),
        "events": events,
        "transactions": transactions
    }

    st.subheader("🧠 Semantic JSON Output")
    st.json(semantic_json)

    st.download_button(
        label="⬇️ Download JSON",
        data=json.dumps(semantic_json, indent=4),
        file_name="efs_output.json",
        mime="application/json"
    )
