import streamlit as st
import json
import os
from datetime import datetime

from audit_dynamodb import get_audit_logger, get_current_user

# Set page configuration
st.set_page_config(page_title="Clinician Review Dashboard", layout="wide")

# Initialize audit logger and user
@st.cache_resource
def init_audit_logger():
    """Initialize DynamoDB audit logger (cached)."""
    try:
        return get_audit_logger()
    except Exception as e:
        st.error(f"Failed to initialize audit logger: {e}")
        return None

audit_logger = init_audit_logger()
current_user = get_current_user()

st.title("Clinician Review & Validation Interface")
st.markdown(f"**Logged in as:** `{current_user}` | **Session:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
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

            # Load structured Track B JSON (clinician role preferred) for validation flags
            summary_json_file = next(
                (f for f in os.listdir(SUMMARY_DIR)
                 if selected_base in f and f.endswith("_clinician_summary.json")),
                None
            )
            if not summary_json_file:
                summary_json_file = next(
                    (f for f in os.listdir(SUMMARY_DIR)
                     if selected_base in f and f.endswith("_summary.json")),
                    None
                )
            summary_meta = {}
            if summary_json_file:
                try:
                    with open(os.path.join(SUMMARY_DIR, summary_json_file), "r", encoding="utf-8") as jf:
                        summary_meta = json.load(jf)
                except Exception as e:
                    st.warning(f"Could not load summary metadata: {e}")

            if summary_meta.get("ocr_deviation_flag"):
                st.warning(
                    f"⚠ OCR Deviation Guard triggered | "
                    f"Score: {summary_meta.get('ocr_deviation_score', 0):.3f}"
                )
                with st.expander("View OCR deviation details"):
                    st.json(summary_meta.get("ocr_deviation_details", []))

            # Store original for comparison
            if f"original_summary_{selected_base}" not in st.session_state:
                st.session_state[f"original_summary_{selected_base}"] = summary_text

            edited_summary = st.text_area("Edit Summary:", value=summary_text, height=250)

            if st.button("Save Updated Summary"):
                original_summary = st.session_state.get(f"original_summary_{selected_base}", summary_text)

                # Only log if there's an actual change
                if edited_summary != original_summary:
                    try:
                        # Log to DynamoDB BEFORE confirming to user (zero data loss)
                        if audit_logger:
                            audit_logger.log_summary_edit(
                                document_id=selected_base,
                                user_id=current_user,
                                before_summary=original_summary,
                                after_summary=edited_summary
                            )

                        # Save to file
                        with open(os.path.join(SUMMARY_DIR, summary_file), "w") as f:
                            f.write(edited_summary)

                        # Update session state
                        st.session_state[f"original_summary_{selected_base}"] = edited_summary
                        st.success("Summary updated and audit logged successfully!")
                    except Exception as e:
                        st.error(f"Failed to save: {e}. Changes NOT persisted.")
                else:
                    st.info("No changes detected.")
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
                        snomed_code = ""
                        if "SNOMEDCTConcepts" in ent and ent["SNOMEDCTConcepts"]:
                            concept = ent["SNOMEDCTConcepts"][0]
                            snomed_code = concept['Code']
                            st.write(f"**Code:** {concept['Code']}")
                            st.write(f"**Description:** {concept['Description']}")
                            st.write(f"**Confidence:** {round(concept['Score'] * 100, 2)}%")

                        # Track previous status
                        status_key = f"status_{selected_base}_{idx}"
                        prev_status_key = f"prev_status_{selected_base}_{idx}"

                        if prev_status_key not in st.session_state:
                            st.session_state[prev_status_key] = "Pending Review"

                        new_status = st.selectbox(
                            f"Status for {ent['Text']}:",
                            ["Pending Review", "Approved", "Incorrect Code", "Needs Clarification"],
                            key=status_key
                        )

                        # Log status change if different
                        if new_status != st.session_state[prev_status_key]:
                            if audit_logger:
                                try:
                                    audit_logger.log_snomed_status_change(
                                        document_id=selected_base,
                                        user_id=current_user,
                                        entity_text=ent['Text'],
                                        snomed_code=snomed_code,
                                        before_status=st.session_state[prev_status_key],
                                        after_status=new_status
                                    )
                                    st.session_state[prev_status_key] = new_status
                                except Exception as e:
                                    st.error(f"Audit log failed: {e}")
        else:
            st.warning(f"No SNOMED file found for {selected_base}. Did Track A finish processing it?")

    # 4. Global Action Buttons
    st.markdown("---")
    c1, c2, c3 = st.columns(3)

    if c1.button("Approve All & Export to EMIS"):
        try:
            # Count entities for audit
            entities_count = len(snomed_data.get("Entities", [])) if snomed_file else 0

            if audit_logger:
                audit_logger.log_approve_all(
                    document_id=selected_base,
                    user_id=current_user,
                    entities_approved=entities_count
                )
            st.balloons()
            st.success(f"Document {selected_base} validated and sent to system. (Audit logged)")
        except Exception as e:
            st.error(f"Failed to approve: {e}")

    if c2.button("Flag for Specialist Review"):
        try:
            if audit_logger:
                audit_logger.log_flag_for_review(
                    document_id=selected_base,
                    user_id=current_user,
                    reason="Manual flag by clinician"
                )
            st.warning("Document flagged for secondary review. (Audit logged)")
        except Exception as e:
            st.error(f"Failed to flag: {e}")

    if c3.button("Download JSON Audit Trail"):
        try:
            if audit_logger:
                json_export = audit_logger.export_audit_trail_to_json(document_id=selected_base)
                st.download_button(
                    label="Click to Download",
                    data=json_export,
                    file_name=f"{selected_base}_audit_trail.json",
                    mime="application/json"
                )
                st.success("Audit trail ready for download!")
            else:
                st.error("Audit logger not available.")
        except Exception as e:
            st.error(f"Failed to generate audit trail: {e}")

    # 5. Audit Trail Viewer (Sidebar)
    st.sidebar.markdown("---")
    st.sidebar.header("Audit Trail")

    if st.sidebar.button("View Document Audit History"):
        if audit_logger:
            trail = audit_logger.get_audit_trail_by_document(selected_base, limit=20)
            if trail:
                st.sidebar.write(f"**{len(trail)} entries found:**")
                for entry in trail:
                    timestamp = entry['timestamp'][:19].replace('T', ' ')
                    st.sidebar.text(f"{timestamp}\n  {entry['change_type']}\n  by {entry['user_id']}")
            else:
                st.sidebar.info("No audit history for this document.")
        else:
            st.sidebar.error("Audit logger not available.")
