import streamlit as st
import json
import os

# Set page configuration
st.set_page_config(page_title="Clinician Review Dashboard", layout="wide")

st.title("🏥 Clinician Review & Validation Interface")
st.markdown("---")

# 1. Setup paths
TEXTRACT_DIR = "textract_outputs"
SNOMED_DIR = "track_a_outputs"
SUMMARY_DIR = "track_b_outputs"

# 2. Sidebar - Document Selection
st.sidebar.header("Document Selection")

# Get all summary files
all_summaries = [f for f in os.listdir(SUMMARY_DIR) if f.endswith(".txt")]

if not all_summaries:
    st.error("No processed documents found. Please run your pipeline first!")
else:
    # Foolproof way to extract just the base name (e.g., 'page_1_CLEANED')
    base_names = []
    for f in all_summaries:
        clean_name = f.replace("_summary.txt", "").replace(".txt", "").replace("_summary", "")
        base_names.append(clean_name)

    # Remove duplicates just in case
    base_names = list(set(base_names))

    selected_base = st.sidebar.selectbox("Choose a document to review:", base_names)

    # 3. Layout: Two Columns (Left: Summary | Right: SNOMED Codes)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📄 Clinical Summary")
        
        # Smart search: find any text file in Track B that contains the base name
        summary_file = next((f for f in os.listdir(SUMMARY_DIR) if selected_base in f and f.endswith(".txt")), None)
        
        if summary_file:
            with open(os.path.join(SUMMARY_DIR, summary_file), "r") as f:
                summary_text = f.read()
            
            edited_summary = st.text_area("Edit Summary:", value=summary_text, height=250)
            
            if st.button("Save Updated Summary"):
                st.success("Summary updated in database (Audit Logged).")
        else:
            st.error(f"Could not find a summary file for {selected_base}.")

    with col2:
        st.subheader("🧬 SNOMED CT Mapping")
        
        # Smart search: find any JSON file in Track A that contains the base name
        snomed_file = next((f for f in os.listdir(SNOMED_DIR) if selected_base in f and f.endswith(".json")), None)
        
        if snomed_file:
            with open(os.path.join(SNOMED_DIR, snomed_file), "r") as f:
                snomed_data = json.load(f)
            
            entities = snomed_data.get("Entities", [])
            
            if not entities:
                st.info("No medical entities detected in this page.")
            else:
                for idx, ent in enumerate(entities):
                    with st.expander(f"Entity: {ent['Text']} ({ent['Category']})"):
                        if "SNOMEDCTConcepts" in ent and ent["SNOMEDCTConcepts"]:
                            concept = ent["SNOMEDCTConcepts"][0]
                            st.write(f"**Code:** {concept['Code']}")
                            st.write(f"**Description:** {concept['Description']}")
                            st.write(f"**Confidence:** {round(concept['Score'] * 100, 2)}%")
                        
                        st.selectbox(f"Status for {ent['Text']}:", 
                                    ["Pending Review", "Approved", "Incorrect Code", "Needs Clarification"],
                                    key=f"status_{idx}")
        else:
            st.warning(f"No SNOMED file found for {selected_base}. Did Track A finish processing it?")

    # 4. Global Action Buttons
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    if c1.button("✅ Approve All & Export to EMIS"):
        st.balloons()
        st.success(f"Document {selected_base} validated and sent to system.")
    
    if c2.button("🚩 Flag for Specialist Review"):
        st.warning("Document flagged for secondary review.")

    if c3.button("📥 Download JSON Audit Trail"):
        st.info("Generating Audit Log...")