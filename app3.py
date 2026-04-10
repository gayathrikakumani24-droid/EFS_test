import streamlit as st
import re
import json
from typing import List, Dict
from groq import Groq

# =========================
# CONFIG
# =========================

client = Groq(api_key="")  # 🔴 replace

# =========================
# RULE-BASED PARSER (MAIN)
# =========================

class PUMLParser:
    def __init__(self):
        self.reset()

    def reset(self):
        self.current_condition = None
        self.loop_context = None
        self.current_subflow = None
        self.events = []

    def parse_line(self, line: str):
        line = line.strip()

        if not line or line.startswith("@") or line.startswith("!"):
            return

        # IF
        if line.startswith("if"):
            match = re.search(r"\((.*?)\)", line)
            if match:
                self.current_condition = match.group(1)
            return

        if line.startswith("endif"):
            self.current_condition = None
            return

        # LOOP
        if line.startswith("while"):
            match = re.search(r"\((.*?)\)", line)
            if match:
                self.loop_context = match.group(1)
            return

        if line.startswith("endwhile"):
            self.loop_context = None
            return

        # GROUP
        if line.startswith("group"):
            self.current_subflow = line.replace("group", "").strip()
            return

        if line.startswith("end group"):
            self.current_subflow = None
            return

        # MESSAGE
        if "->" in line:
            parts = re.split(r"->|:", line)

            if len(parts) >= 2:
                event = {
                    "type": "message",
                    "from": parts[0].strip(),
                    "to": parts[1].strip(),
                    "name": parts[2].strip() if len(parts) > 2 else "",
                    "condition": self.current_condition,
                    "loop": self.loop_context,
                    "subflow": self.current_subflow
                }

                self.events.append(event)

    def parse(self, text: str) -> List[Dict]:
        self.reset()
        for line in text.split("\n"):
            self.parse_line(line)
        return self.events


# =========================
# SAFE JSON EXTRACT
# =========================

def safe_json_extract(text):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    return []


# =========================
# AI ENHANCER (SAFE MODE)
# =========================

def ai_enhance_events(events: List[Dict]) -> List[Dict]:

    prompt = f"""
You are a hardware protocol assistant.

STRICT RULES:
- DO NOT change 'from', 'to', or 'name'
- DO NOT rename signals
- DO NOT invent new fields except allowed ones

ONLY:
- Add "type" (signal, handshake, transaction)
- Extract "condition" if explicitly present
- Add "dependencies" if obvious
- Add signal name
Return ONLY JSON array.

Input:
{json.dumps(events, indent=2)}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        ai_output = safe_json_extract(response.choices[0].message.content)

        # 🔥 SAFE MERGE
        enhanced = []

        for i, event in enumerate(events):
            ai_event = ai_output[i] if i < len(ai_output) else {}

            merged = {
                "type": ai_event.get("type", event.get("type", "message")),
                "from": event["from"],   # LOCKED
                "to": event["to"],       # LOCKED
                "name": event["name"],   # LOCKED
                "condition": ai_event.get("condition", event.get("condition")),
                "loop": event.get("loop"),
                "subflow": event.get("subflow"),
                "dependencies": ai_event.get("dependencies", [])
            }

            enhanced.append(merged)

        return enhanced

    except Exception as e:
        st.error(f"AI Error: {e}")
        return events


# =========================
# VALIDATION CHECK
# =========================

def validate_ai(rule_events, ai_events):
    for r, a in zip(rule_events, ai_events):
        if r["name"] != a["name"]:
            st.warning("⚠️ AI modified core fields → Reverting to rule-based output")
            return rule_events
    return ai_events


# =========================
# STREAMLIT UI
# =========================

st.set_page_config(page_title="PUML AI Parser", layout="wide")

st.title("🚀 Safe AI-Assisted PUML Parser (Groq)")
st.markdown("Rule-based core + AI enhancement (No data corruption)")

# Sidebar
st.sidebar.header("📥 Input Options")

uploaded_file = st.sidebar.file_uploader("Upload PUML", type=["txt", "puml"])
manual_input = st.sidebar.text_area("Or paste PUML")

use_ai = st.sidebar.checkbox("Enable AI Enhancement", value=True)

# Input handling
puml_text = ""

if uploaded_file:
    puml_text = uploaded_file.read().decode("utf-8")
elif manual_input:
    puml_text = manual_input

# Run button
if st.sidebar.button("⚡ Run Parser"):

    if not puml_text.strip():
        st.warning("Please provide PUML input")
    else:
        parser = PUMLParser()
        rule_output = parser.parse(puml_text)

        col1, col2 = st.columns(2)

        # LEFT SIDE
        with col1:
            st.subheader("📄 PUML Input")
            st.code(puml_text)

            st.subheader("🧱 Rule-Based Output (Ground Truth)")
            st.json(rule_output)

        # RIGHT SIDE
        final_output = rule_output

        if use_ai:
            ai_output = ai_enhance_events(rule_output)
            final_output = validate_ai(rule_output, ai_output)

            with col2:
                st.subheader("🤖 AI Enhanced Output (Safe Mode)")
                st.json(final_output)

        # DOWNLOAD
        st.subheader("📥 Download Output")

        st.download_button(
            "Download JSON",
            data=json.dumps(final_output, indent=2),
            file_name="output.json",
            mime="application/json"
        )

        st.success("✅ Parsing Completed Safely!")