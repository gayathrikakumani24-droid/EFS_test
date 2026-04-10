%%writefile app.py
import streamlit as st
import os
from llama_index.core import SimpleDirectoryReader, TreeIndex
from llama_index.llms.groq import Groq
import json

st.set_page_config(layout="wide")
st.title("🌲 LlamaIndex TreeIndex App (Colab)")
def documents_to_json(documents):
    json_data = []

    for i, doc in enumerate(documents):
        json_data.append({
            "id": i,
            "text": doc.text,
            "metadata": doc.metadata
        })

    return json_data
# -------------------------------
# Upload PDF
# -------------------------------
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ File uploaded")

    # -------------------------------
    # Build TreeIndex
    # -------------------------------
    if st.button("🚀 Build Tree Index"):

        with st.spinner("Processing document..."):
            documents = SimpleDirectoryReader(
                input_files=["temp.pdf"]
            ).load_data()
            docs_json = documents_to_json(documents)
            st.session_state["docs_json"] = docs_json
            llm = Groq(model="llama-3.3-70b-versatile")  # fast model

            index = TreeIndex.from_documents(
                documents,
                llm=llm,
                summary_mode="tree_summarize"
            )

            query_engine = index.as_query_engine(llm=llm)

            st.session_state["engine"] = query_engine
            st.session_state["docs"] = documents

        st.success("✅ TreeIndex Ready!")

# -------------------------------
# Preview Text
# -------------------------------
if "docs" in st.session_state:
    st.subheader("📄 Document Preview")

    for i, doc in enumerate(st.session_state["docs"]):
        st.write(f"### Page {i+1}")
        st.write(doc.text)
if "docs_json" in st.session_state:
    st.subheader("📊 Documents (JSON Format)")
    st.json(st.session_state["docs_json"])


if "docs_json" in st.session_state:
    json_str = json.dumps(st.session_state["docs_json"], indent=2)

    st.download_button(
        label="⬇️ Download JSON",
        data=json_str,
        file_name="documents.json",
        mime="application/json"
    )
# -------------------------------
# Ask Questions
# -------------------------------
if "engine" in st.session_state:
    st.subheader("💬 Ask Questions")

    query = st.text_input("Enter your query")

    if st.button("Ask"):
        if query:
            with st.spinner("Thinking..."):
                response = st.session_state["engine"].query(query)

            # IMPORTANT FIX
            st.write(response.response)

# -------------------------------
# Structured Extraction
# -------------------------------
if "engine" in st.session_state:
    st.subheader("⚡ Extract Structured Data")

    if st.button("Extract Signals / Flow"):

        prompt = """
        Extract structured data from document.

        Format:
        {
          "signals": [],
          "conditions": [],
          "events": []
          "source":[],
          "destination":[]
        }

         Return ONLY JSON.
        """

        with st.spinner("Extracting..."):
            response = st.session_state["engine"].query(prompt)

        st.code(response.response, language="json")
