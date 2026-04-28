import streamlit as st
import json
import pandas as pd

st.set_page_config(page_title="JSON Protocol Comparator", layout="wide")

st.title("🔍 JSON Interaction Comparator")
st.write("Compare two interaction JSON files (User Spec vs Protocol Spec)")

# ----------------------------
# Upload Section
# ----------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload JSON 1")
    file1 = st.file_uploader(
        "Choose First JSON",
        type=["json"],
        key="file1"
    )

with col2:
    st.subheader("Upload JSON 2")
    file2 = st.file_uploader(
        "Choose Second JSON",
        type=["json"],
        key="file2"
    )


# ----------------------------
# Comparison Function
# ----------------------------

def compare_json(json1, json2):

    mismatches=[]
    matched=0

    max_steps=max(len(json1),len(json2))

    for i in range(max_steps):

        if i >= len(json1):
            mismatches.append({
                "Step":i+1,
                "Issue":"Extra in JSON2",
                "Details":json2[i]
            })
            continue

        if i >= len(json2):
            mismatches.append({
                "Step":i+1,
                "Issue":"Missing in JSON2",
                "Details":json1[i]
            })
            continue


        a=json1[i]
        b=json2[i]

        if a==b:
            matched+=1

        else:

            for field in ["from","to","signal"]:
                if a.get(field)!=b.get(field):

                    mismatches.append({
                        "Step":i+1,
                        "Issue":f"{field} mismatch",
                        "JSON1":a.get(field),
                        "JSON2":b.get(field)
                    })
    similarity=(matched/max_steps)*100

    return mismatches, similarity


# ----------------------------
# Run Comparison
# ----------------------------

if file1 and file2:

    json1=json.load(file1)
    json2=json.load(file2)

    st.divider()

    if st.button("Compare JSONs"):

        mismatches, similarity=compare_json(json1,json2)

        st.subheader("Similarity Score")
        st.metric(
            label="Protocol Match %",
            value=f"{similarity:.2f}%"
        )


        if len(mismatches)==0:
            st.success("✅ Perfect Match. No differences found.")

        else:
            st.error(f"Found {len(mismatches)} mismatches")

            df=pd.DataFrame(mismatches)
            st.dataframe(df,use_container_width=True)


        # Optional raw JSON view
        with st.expander("View JSON Files"):
            c1,c2=st.columns(2)

            with c1:
                st.json(json1)

            with c2:
                st.json(json2)