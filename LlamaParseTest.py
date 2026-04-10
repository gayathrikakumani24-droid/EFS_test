%%writefile app.py
import streamlit as st
import os
import json

from llama_parse import LlamaParse
from llama_index.core import TreeIndex, Settings
from llama_index.llms.groq import Groq

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(layout="wide")
st.title("🌲 Structured Document AI (LlamaParse + JSON)")

# -------------------------------
# SET GROQ MODEL (FIXED)
# -------------------------------
MODEL = "llama-3.1-8b-instant"
Settings.llm = Groq(model=MODEL)

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

# -------------------------------
# MARKDOWN → JSON PARSER
# -------------------------------
def markdown_to_json(documents):
    structured = []

    current_h1 = None
    current_h2 = None

    for doc in documents:
        lines = doc.text.split("\n")

        for line in lines:
            line = line.strip()

            if not line:
                continue

            # H1
            if line.startswith("# "):
                current_h1 = {
                    "heading": line.replace("# ", ""),
                    "content": "",
                    "subsections": []
                }
                structured.append(current_h1)
                current_h2 = None

            # H2
            elif line.startswith("## "):
                if current_h1:
                    current_h2 = {
                        "subheading": line.replace("## ", ""),
                        "content": ""
                    }
                    current_h1["subsections"].append(current_h2)

            # TEXT
            else:
                if current_h2:
                    current_h2["content"] += line + " "
                elif current_h1:
                    current_h1["content"] += line + " "

    return structured


# -------------------------------
# PROCESS DOCUMENT
# -------------------------------
if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ File uploaded")

    if st.button("🚀 Parse & Build Index"):

        with st.spinner("Parsing document..."):

            # -------------------------------
            # LlamaParse (STRUCTURE)
            # -------------------------------
            parser = LlamaParse(
                api_key="",  # replace with your api key
                result_type="markdown"
            )

            documents = parser.load_data("temp.pdf")

            # -------------------------------
            # JSON STRUCTURE
            # -------------------------------
            structured_json = markdown_to_json(documents)

            # -------------------------------
            # TREE INDEX
            # -------------------------------
            index = TreeIndex.from_documents(
                documents,
                summary_mode="tree_summarize"
            )

            query_engine = index.as_query_engine()

            # SAVE STATE
            st.session_state["engine"] = query_engine
            st.session_state["docs"] = documents
            st.session_state["json"] = structured_json

        st.success("✅ Processing Complete!")

# -------------------------------
# SHOW STRUCTURED TEXT
# -------------------------------
if "docs" in st.session_state:
    st.subheader("📄 Structured Document")

    for i, doc in enumerate(st.session_state["docs"][:3]):
        st.markdown(doc.text[:1000])


# -------------------------------
# SHOW JSON
# -------------------------------
if "json" in st.session_state:
    st.subheader("📊 JSON Output (Headings + Subheadings)")
    st.json(st.session_state["json"])


# -------------------------------
# DOWNLOAD JSON
# -------------------------------
if "json" in st.session_state:
    json_str = json.dumps(st.session_state["json"], indent=2)

    st.download_button(
        label="⬇️ Download JSON",
        data=json_str,
        file_name="structured_doc.json",
        mime="application/json"
    )


# -------------------------------
# QUERY SYSTEM
# -------------------------------
# -------------------------------
# QUERY SYSTEM (STRICT MODE)
# -------------------------------
if "engine" in st.session_state:
    st.subheader("💬 Ask Questions (Strict Mode)")

    query = st.text_input("Enter your query")

    if st.button("Ask"):
        if query:
            with st.spinner("Searching document..."):

                strict_prompt = f"""
You must answer ONLY using the provided document context.

Rules:
- Do NOT use external knowledge
- Do NOT guess
- Do NOT infer beyond text
- If answer is not explicitly present → say:
  "Not found in document"

Question:
{query}
"""

                response = st.session_state["engine"].query(strict_prompt)

                answer = response.response

            # -------------------------------
            # DISPLAY ANSWER
            # -------------------------------
            st.subheader("📌 Answer")
            st.write(answer)

            # -------------------------------
            # SOURCE NODES (VERY IMPORTANT)
            # -------------------------------
            st.subheader("📄 Source Context")

            try:
                for node in response.source_nodes:
                    st.info(node.text[:500])
            except:
                st.warning("No source nodes available")


# -------------------------------
# STRUCTURED EXTRACTION (LLM)
# -------------------------------
if "engine" in st.session_state:
    st.subheader("⚡ Extract Signals / Flow")

    if st.button("Extract Signals"):

        prompt = f"""
        You are extracting hardware protocol information.

        Extract ONLY:

        1. signals (exact names like AWVALID, ARESETn)
        2. transactions (read, write, reset, handshake)
        3. conditions (if/when logic)

        Rules:
        - Do NOT hallucinate
        - Do NOT invent signals
        - Only extract what is explicitly present
        - Follow AXI protocol structure

        Return STRICT JSON:

        {{
  "signals": [],
  "transactions": [],
  "conditions": []
        }}
        """

        response = st.session_state["engine"].query(prompt)

        st.code(response.response, language="json")
