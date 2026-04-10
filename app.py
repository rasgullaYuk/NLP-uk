import json
import os
from datetime import date, datetime, timedelta
from typing import Any, Dict, List

import streamlit as st

from audit_dynamodb import get_audit_logger, get_current_user
from emis_export_integration import export_to_emis
from lambda_confidence_aggregator import DEFAULT_THRESHOLD
from review_interface_utils import (
    ACTION_PRIORITY_OPTIONS,
    SUMMARY_ROLES,
    compute_confidence_bundle,
    confidence_visual,
    discover_document_assets,
    extract_textract_text,
    format_actions_for_text,
    load_all_role_summaries,
    load_snomed_entities,
    normalize_action_items,
    parse_actions_from_text,
    recommendation_text,
    serialize_action_items,
)

# Page configuration
st.set_page_config(page_title="Clinician Review Dashboard", layout="wide")

# Paths
TEXTRACT_DIR = "textract_outputs"
SNOMED_DIR = "track_a_outputs"
SUMMARY_DIR = "track_b_outputs"
REVIEW_DIR = "review_outputs"

DECISION_OPTIONS = ["Pending Review", "Approved", "Rejected"]
CATEGORY_OPTIONS = [
    "Problems/Issues",
    "Medication",
    "Diagnosis",
    "Procedures",
    "Investigations",
    "Uncategorized",
]
ROLE_ACTION_PANEL_TITLE = {
    "clinician": "Clinician Actions",
    "patient": "Patient Instructions",
    "pharmacist": "Pharmacist Actions",
}


@st.cache_resource
def init_audit_logger():
    try:
        return get_audit_logger()
    except Exception as exc:
        st.error(f"Failed to initialize audit logger: {exc}")
        return None


def load_review_state(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_review_state(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _setdefault_state(key: str, value: Any) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


def render_unified_score_card(score: float, threshold: float, system_score: float) -> None:
    visual = confidence_visual(score, threshold=threshold)
    recommendation = recommendation_text(score, threshold=threshold)
    st.markdown(
        f"""
        <div style="
            border: 1px solid {visual["color"]};
            border-left: 10px solid {visual["color"]};
            border-radius: 8px;
            background: {visual["background"]};
            padding: 14px;
            margin: 8px 0 16px 0;
        ">
            <div style="font-size: 1.4rem; font-weight: 700; color: {visual["color"]};">
                {visual["icon"]} Unified Confidence Score: {score:.3f}
            </div>
            <div style="margin-top: 4px; font-weight: 600; color: {visual["color"]};">
                Recommendation: {recommendation}
            </div>
            <div style="margin-top: 6px; color: #334155; font-size: 0.9rem;">
                System score: {system_score:.3f} | Threshold: {threshold:.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _snomed_category_options(current_category: str) -> List[str]:
    options = list(CATEGORY_OPTIONS)
    if current_category and current_category not in options:
        options.append(current_category)
    return options


def _to_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return date.today() + timedelta(days=7)
    return date.today() + timedelta(days=7)


def build_review_payload(
    document_id: str,
    current_user: str,
    role_summaries: Dict[str, Dict[str, Any]],
    snomed_entities: List[Dict[str, Any]],
    confidence_bundle: Dict[str, Any],
    system_score: float,
    effective_score: float,
    threshold: float,
    confidence_mode: str,
) -> Dict[str, Any]:
    summaries_payload: Dict[str, Any] = {}
    role_based_actions_payload: Dict[str, Any] = {}
    for role in SUMMARY_ROLES:
        role_data = role_summaries.get(role, {})
        summary_key = f"{document_id}_{role}_summary_edit"
        actions_key = f"{document_id}_{role}_actions_edit"
        action_items_key = f"{document_id}_{role}_action_items"
        decision_key = f"{document_id}_{role}_decision"
        override_key = f"{document_id}_{role}_override"
        structured_actions = serialize_action_items(
            st.session_state.get(action_items_key, [])
        )
        role_based_actions_payload[role] = structured_actions

        summaries_payload[role] = {
            "role": role,
            "confidence_score": float(role_data.get("confidence_score", 0.0)),
            "original_summary": role_data.get("summary", ""),
            "edited_summary": st.session_state.get(summary_key, ""),
            "original_follow_up_actions": role_data.get("follow_up_actions", []),
            "edited_follow_up_actions": parse_actions_from_text(
                st.session_state.get(actions_key, "")
            ),
            "structured_action_items": structured_actions,
            "decision": st.session_state.get(decision_key, DECISION_OPTIONS[0]),
            "override_confidence_gate": bool(st.session_state.get(override_key, False)),
        }

    snomed_payload: List[Dict[str, Any]] = []
    for entity in snomed_entities:
        entity_id = entity["entity_id"]
        category_key = f"{document_id}_{entity_id}_category"
        code_key = f"{document_id}_{entity_id}_code"
        desc_key = f"{document_id}_{entity_id}_description"
        decision_key = f"{document_id}_{entity_id}_decision"
        override_key = f"{document_id}_{entity_id}_override"

        snomed_payload.append(
            {
                "entity_id": entity_id,
                "text": entity.get("text", ""),
                "confidence": float(entity.get("confidence", 0.0)),
                "original_category": entity.get("category", ""),
                "edited_category": st.session_state.get(category_key, entity.get("category", "")),
                "original_snomed_code": entity.get("snomed_code", ""),
                "edited_snomed_code": st.session_state.get(
                    code_key, entity.get("snomed_code", "")
                ),
                "original_description": entity.get("description", ""),
                "edited_description": st.session_state.get(
                    desc_key, entity.get("description", "")
                ),
                "decision": st.session_state.get(decision_key, DECISION_OPTIONS[0]),
                "override_confidence_gate": bool(st.session_state.get(override_key, False)),
            }
        )

    doc_decision_key = f"{document_id}_document_decision"
    doc_override_gate_key = f"{document_id}_document_override_gate"
    doc_override_score_key = f"{document_id}_override_score"

    return {
        "document_id": document_id,
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "saved_by": current_user,
        "document_review": {
            "decision": st.session_state.get(doc_decision_key, DECISION_OPTIONS[0]),
            "override_confidence_gate": bool(
                st.session_state.get(doc_override_gate_key, False)
            ),
            "confidence_mode": confidence_mode,
            "system_unified_confidence_score": float(system_score),
            "effective_unified_confidence_score": float(effective_score),
            "override_score": (
                float(st.session_state.get(doc_override_score_key, system_score))
                if confidence_mode == "Override unified score"
                else None
            ),
            "threshold": float(threshold),
            "route": "bypass_database" if effective_score >= threshold else "human_review",
            "recommendation": recommendation_text(effective_score, threshold=threshold),
        },
        "confidence_bundle": confidence_bundle,
        "summaries": summaries_payload,
        "role_based_actions": role_based_actions_payload,
        "snomed": {"entities": snomed_payload},
    }


def log_review_changes(
    audit_logger,
    document_id: str,
    current_user: str,
    previous_state: Dict[str, Any],
    current_state: Dict[str, Any],
) -> int:
    change_count = 0

    previous_document = previous_state.get("document_review", {})
    current_document = current_state.get("document_review", {})
    previous_doc_projection = {
        "decision": previous_document.get("decision", "Pending Review"),
        "override_confidence_gate": previous_document.get("override_confidence_gate", False),
        "effective_unified_confidence_score": previous_document.get(
            "effective_unified_confidence_score",
            current_document.get("system_unified_confidence_score", 0.0),
        ),
        "route": previous_document.get("route", "human_review"),
    }
    current_doc_projection = {
        "decision": current_document.get("decision"),
        "override_confidence_gate": current_document.get("override_confidence_gate"),
        "effective_unified_confidence_score": current_document.get(
            "effective_unified_confidence_score"
        ),
        "route": current_document.get("route"),
    }
    if previous_doc_projection != current_doc_projection:
        audit_logger.log_change(
            document_id=document_id,
            user_id=current_user,
            change_type="DOCUMENT_REVIEW_DECISION",
            before_state=previous_doc_projection,
            after_state=current_doc_projection,
            metadata={
                "threshold": current_document.get("threshold"),
                "confidence_mode": current_document.get("confidence_mode"),
                "override_score": current_document.get("override_score"),
            },
        )
        change_count += 1

    previous_summaries = previous_state.get("summaries", {})
    previous_role_based_actions = previous_state.get("role_based_actions", {})
    current_role_based_actions = current_state.get("role_based_actions", {})
    for role, summary_state in current_state.get("summaries", {}).items():
        previous_role = previous_summaries.get(role, {})
        before_summary = previous_role.get(
            "edited_summary", summary_state.get("original_summary", "")
        )
        after_summary = summary_state.get("edited_summary", "")
        if before_summary != after_summary:
            audit_logger.log_change(
                document_id=document_id,
                user_id=current_user,
                change_type="SUMMARY_EDIT",
                before_state={"role": role, "summary": before_summary},
                after_state={"role": role, "summary": after_summary},
                metadata={"field": f"{role}_summary"},
            )
            change_count += 1

        before_actions = previous_role.get(
            "edited_follow_up_actions", summary_state.get("original_follow_up_actions", [])
        )
        after_actions = summary_state.get("edited_follow_up_actions", [])
        if before_actions != after_actions:
            audit_logger.log_change(
                document_id=document_id,
                user_id=current_user,
                change_type="ACTION_PLAN_EDIT",
                before_state={"role": role, "actions": before_actions},
                after_state={"role": role, "actions": after_actions},
                metadata={"field": f"{role}_follow_up_actions"},
            )
            change_count += 1

        before_decision = previous_role.get("decision", DECISION_OPTIONS[0])
        after_decision = summary_state.get("decision", DECISION_OPTIONS[0])
        before_override = bool(previous_role.get("override_confidence_gate", False))
        after_override = bool(summary_state.get("override_confidence_gate", False))
        if before_decision != after_decision or before_override != after_override:
            audit_logger.log_change(
                document_id=document_id,
                user_id=current_user,
                change_type="SUMMARY_REVIEW_DECISION",
                before_state={
                    "role": role,
                    "decision": before_decision,
                    "override_confidence_gate": before_override,
                },
                after_state={
                    "role": role,
                    "decision": after_decision,
                    "override_confidence_gate": after_override,
                },
                metadata={"field": f"{role}_review_decision"},
            )
            change_count += 1

        before_structured_actions = previous_role_based_actions.get(
            role, previous_role.get("structured_action_items", [])
        )
        after_structured_actions = current_role_based_actions.get(role, [])
        if before_structured_actions != after_structured_actions:
            audit_logger.log_change(
                document_id=document_id,
                user_id=current_user,
                change_type="ROLE_ACTION_PLAN_EDIT",
                before_state={"role": role, "actions": before_structured_actions},
                after_state={"role": role, "actions": after_structured_actions},
                metadata={"field": f"{role}_role_based_actions"},
            )
            change_count += 1

    previous_entities_by_id = {
        entry.get("entity_id"): entry
        for entry in previous_state.get("snomed", {}).get("entities", [])
    }
    for entity_state in current_state.get("snomed", {}).get("entities", []):
        entity_id = entity_state["entity_id"]
        previous_entity = previous_entities_by_id.get(entity_id, {})

        before_projection = {
            "category": previous_entity.get(
                "edited_category", entity_state.get("original_category")
            ),
            "snomed_code": previous_entity.get(
                "edited_snomed_code", entity_state.get("original_snomed_code")
            ),
            "description": previous_entity.get(
                "edited_description", entity_state.get("original_description")
            ),
            "decision": previous_entity.get("decision", DECISION_OPTIONS[0]),
            "override_confidence_gate": bool(
                previous_entity.get("override_confidence_gate", False)
            ),
        }
        after_projection = {
            "category": entity_state.get("edited_category"),
            "snomed_code": entity_state.get("edited_snomed_code"),
            "description": entity_state.get("edited_description"),
            "decision": entity_state.get("decision", DECISION_OPTIONS[0]),
            "override_confidence_gate": bool(
                entity_state.get("override_confidence_gate", False)
            ),
        }
        if before_projection != after_projection:
            audit_logger.log_change(
                document_id=document_id,
                user_id=current_user,
                change_type="SNOMED_MAPPING_EDIT",
                before_state=before_projection,
                after_state=after_projection,
                metadata={
                    "entity_id": entity_id,
                    "entity_text": entity_state.get("text"),
                    "confidence": entity_state.get("confidence"),
                },
            )
            change_count += 1

    return change_count


# Initialize app/session
audit_logger = init_audit_logger()
current_user = get_current_user()
os.makedirs(REVIEW_DIR, exist_ok=True)

st.title("Clinician Review & Validation Interface")
st.markdown(
    f"**Logged in as:** `{current_user}` | **Session:** {datetime.now().strftime('%Y-%m-%d %H:%M')}"
)
st.markdown("---")

assets_by_document = discover_document_assets(
    summary_dir=SUMMARY_DIR,
    snomed_dir=SNOMED_DIR,
    textract_dir=TEXTRACT_DIR,
)

if not assets_by_document:
    st.error("No processed documents found. Please run your pipeline first.")
    st.stop()

st.sidebar.header("Document Selection")
selected_document = st.sidebar.selectbox(
    "Choose a document to review:",
    list(assets_by_document.keys()),
)
selected_assets = assets_by_document[selected_document]
review_state_path = os.path.join(REVIEW_DIR, f"{selected_document}_review.json")
saved_review_state = load_review_state(review_state_path)

role_summaries = load_all_role_summaries(selected_assets)
snomed_entities = load_snomed_entities(selected_assets.get("snomed_json"))
source_document_text = extract_textract_text(selected_assets.get("textract_json"))
snomed_codes_for_actions = sorted(
    {
        str(entry.get("snomed_code", "")).strip()
        for entry in snomed_entities
        if str(entry.get("snomed_code", "")).strip()
    }
)

confidence_bundle = compute_confidence_bundle(
    selected_assets, threshold=DEFAULT_THRESHOLD
)
system_unified_confidence = confidence_bundle.get("unified_confidence_score", 0.0)
confidence_threshold = confidence_bundle.get("threshold", DEFAULT_THRESHOLD)
component_scores = confidence_bundle.get("component_scores", {})

# Initialize state for document-level controls
document_review_state = saved_review_state.get("document_review", {})
confidence_mode_key = f"{selected_document}_confidence_mode"
override_score_key = f"{selected_document}_override_score"
document_decision_key = f"{selected_document}_document_decision"
document_override_gate_key = f"{selected_document}_document_override_gate"

_setdefault_state(
    confidence_mode_key,
    "Override unified score"
    if document_review_state.get("override_score") is not None
    else "Use system unified score",
)
_setdefault_state(
    override_score_key,
    float(document_review_state.get("override_score", system_unified_confidence)),
)
_setdefault_state(
    document_decision_key,
    document_review_state.get("decision", DECISION_OPTIONS[0]),
)
_setdefault_state(
    document_override_gate_key,
    bool(document_review_state.get("override_confidence_gate", False)),
)

# Initialize state for summaries
for role in SUMMARY_ROLES:
    role_state = saved_review_state.get("summaries", {}).get(role, {})
    role_data = role_summaries.get(role, {})
    summary_key = f"{selected_document}_{role}_summary_edit"
    actions_key = f"{selected_document}_{role}_actions_edit"
    decision_key = f"{selected_document}_{role}_decision"
    override_key = f"{selected_document}_{role}_override"

    _setdefault_state(
        summary_key,
        role_state.get("edited_summary", role_data.get("summary", "")),
    )
    _setdefault_state(
        actions_key,
        format_actions_for_text(
            role_state.get(
                "edited_follow_up_actions",
                role_data.get("follow_up_actions", []),
            )
        ),
    )
    _setdefault_state(
        decision_key,
        role_state.get("decision", DECISION_OPTIONS[0]),
    )
    _setdefault_state(
        override_key,
        bool(role_state.get("override_confidence_gate", False)),
    )

# Initialize state for structured role-based action plans
saved_role_actions = saved_review_state.get("role_based_actions", {})
for role in SUMMARY_ROLES:
    action_items_key = f"{selected_document}_{role}_action_items"
    role_actions_default = saved_role_actions.get(role)
    if role_actions_default is None:
        role_actions_default = normalize_action_items(
            role_summaries.get(role, {}).get("follow_up_actions", []),
            default_assignee=role.title(),
        )
    _setdefault_state(
        action_items_key,
        [dict(item) for item in role_actions_default],
    )

# Initialize state for SNOMED entity edits
previous_entities_map = {
    entry.get("entity_id"): entry
    for entry in saved_review_state.get("snomed", {}).get("entities", [])
}
for entity in snomed_entities:
    entity_id = entity["entity_id"]
    previous_entity = previous_entities_map.get(entity_id, {})
    category_key = f"{selected_document}_{entity_id}_category"
    code_key = f"{selected_document}_{entity_id}_code"
    desc_key = f"{selected_document}_{entity_id}_description"
    decision_key = f"{selected_document}_{entity_id}_decision"
    override_key = f"{selected_document}_{entity_id}_override"

    _setdefault_state(
        category_key,
        previous_entity.get("edited_category", entity.get("category", "Uncategorized")),
    )
    _setdefault_state(
        code_key,
        previous_entity.get("edited_snomed_code", entity.get("snomed_code", "")),
    )
    _setdefault_state(
        desc_key,
        previous_entity.get("edited_description", entity.get("description", "")),
    )
    _setdefault_state(
        decision_key,
        previous_entity.get("decision", DECISION_OPTIONS[0]),
    )
    _setdefault_state(
        override_key,
        bool(previous_entity.get("override_confidence_gate", False)),
    )

# Top confidence controls
st.subheader("Confidence & Routing Overview")
st.radio(
    "Unified confidence handling",
    ["Use system unified score", "Override unified score"],
    key=confidence_mode_key,
    horizontal=True,
)
st.slider(
    "Override unified confidence score",
    min_value=0.0,
    max_value=1.0,
    value=float(st.session_state[override_score_key]),
    step=0.01,
    key=override_score_key,
    disabled=st.session_state[confidence_mode_key] != "Override unified score",
)

effective_unified_confidence = (
    float(st.session_state[override_score_key])
    if st.session_state[confidence_mode_key] == "Override unified score"
    else float(system_unified_confidence)
)

render_unified_score_card(
    score=effective_unified_confidence,
    threshold=confidence_threshold,
    system_score=system_unified_confidence,
)

with st.expander("Component confidence breakdown"):
    st.write("Textract:", round(component_scores.get("textract", 0.0), 4))
    st.write("Comprehend:", round(component_scores.get("comprehend", 0.0), 4))
    st.write("FAISS:", round(component_scores.get("faiss", 0.0), 4))
    st.write("LLM:", round(component_scores.get("llm_logprobs", 0.0), 4))
    st.write("Weights:", confidence_bundle.get("weights", {}))
    st.write("Route recommendation:", confidence_bundle.get("route", "human_review"))
    st.write(
        "Calculation latency (ms):",
        round(float(confidence_bundle.get("calculation_latency_ms", 0.0)), 3),
    )

st.selectbox("Document review decision", DECISION_OPTIONS, key=document_decision_key)
st.checkbox(
    "Override confidence gate for document-level approval",
    key=document_override_gate_key,
)

if effective_unified_confidence >= confidence_threshold:
    st.success("✅ High confidence document - pre-filled fields ready for quick validation.")
else:
    st.warning("⚠️ Confidence below threshold - manual review required for low-confidence fields.")

st.markdown("---")

# Separation of source view and editable fields
left_col, right_col = st.columns([1, 1.2], gap="large")

with left_col:
    st.subheader("Original Document View")
    if source_document_text:
        st.text_area(
            "Extracted source text (read-only)",
            value=source_document_text,
            height=760,
            disabled=True,
        )
    else:
        st.info("No Textract source text found for this document.")

with right_col:
    st.subheader("Editable AI-Extracted Fields")
    tab_summary, tab_snomed, tab_actions = st.tabs(
        ["Summaries", "SNOMED Mapping", "Role-Based Action Plans"]
    )

    with tab_summary:
        for role in SUMMARY_ROLES:
            role_data = role_summaries.get(role, {})
            role_conf = float(role_data.get("confidence_score", 0.0)) or float(
                system_unified_confidence
            )
            role_visual = confidence_visual(role_conf, threshold=confidence_threshold)
            summary_key = f"{selected_document}_{role}_summary_edit"
            actions_key = f"{selected_document}_{role}_actions_edit"
            decision_key = f"{selected_document}_{role}_decision"
            override_key = f"{selected_document}_{role}_override"

            with st.expander(
                f"{role_visual['icon']} {role.title()} ({role_visual['label']} - {role_conf:.3f})",
                expanded=(role == "clinician"),
            ):
                if role_conf >= confidence_threshold:
                    st.success("✅ High confidence: field pre-filled from AI output.")
                else:
                    st.warning("⚠️ Low confidence: manual review required before approval.")

                st.caption("Original AI-generated summary")
                st.text_area(
                    f"Original {role.title()} Summary (read-only)",
                    value=role_data.get("summary", ""),
                    disabled=True,
                    height=120,
                    key=f"{selected_document}_{role}_original_summary_view",
                )

                st.text_area(
                    f"Editable {role.title()} Summary",
                    key=summary_key,
                    height=150,
                )

                st.caption("Original AI-generated action plan")
                st.text_area(
                    f"Original {role.title()} Actions (read-only)",
                    value=format_actions_for_text(role_data.get("follow_up_actions", [])),
                    disabled=True,
                    height=90,
                    key=f"{selected_document}_{role}_original_actions_view",
                )

                st.text_area(
                    f"Editable {role.title()} Actions (one per line)",
                    key=actions_key,
                    height=110,
                )

                st.selectbox(
                    f"{role.title()} field decision",
                    DECISION_OPTIONS,
                    key=decision_key,
                )
                st.checkbox(
                    f"Override low-confidence gate for {role.title()} fields",
                    key=override_key,
                )

    with tab_snomed:
        if not snomed_entities:
            st.info("No SNOMED entities available for this document.")
        else:
            for entity in snomed_entities:
                entity_id = entity["entity_id"]
                entity_text = entity.get("text", "")
                entity_conf = float(entity.get("confidence", 0.0)) or float(
                    system_unified_confidence
                )
                visual = confidence_visual(entity_conf, threshold=confidence_threshold)

                category_key = f"{selected_document}_{entity_id}_category"
                code_key = f"{selected_document}_{entity_id}_code"
                desc_key = f"{selected_document}_{entity_id}_description"
                decision_key = f"{selected_document}_{entity_id}_decision"
                override_key = f"{selected_document}_{entity_id}_override"

                with st.expander(
                    f"{visual['icon']} {entity_text or entity_id} ({visual['label']} - {entity_conf:.3f})"
                ):
                    if entity_conf >= confidence_threshold:
                        st.success("✅ High confidence mapping pre-filled.")
                    else:
                        st.warning("⚠️ Low confidence mapping: manual verification required.")

                    original_col, edit_col = st.columns(2)

                    with original_col:
                        st.markdown("**Original AI values**")
                        st.write("Category:", entity.get("category", "Uncategorized"))
                        st.write("SNOMED code:", entity.get("snomed_code", ""))
                        st.write("Description:", entity.get("description", ""))

                    with edit_col:
                        st.markdown("**Editable values**")
                        current_category = st.session_state.get(
                            category_key, entity.get("category", "Uncategorized")
                        )
                        category_options = _snomed_category_options(current_category)
                        selected_category = st.selectbox(
                            f"Mapped Category - {entity_id}",
                            category_options,
                            index=category_options.index(current_category),
                            key=category_key,
                        )
                        st.text_input(f"SNOMED Code - {entity_id}", key=code_key)
                        st.text_input(f"Description - {entity_id}", key=desc_key)
                        st.selectbox(
                            f"Field decision - {entity_id}",
                            DECISION_OPTIONS,
                            key=decision_key,
                        )
                        st.checkbox(
                            f"Override low-confidence gate - {entity_id}",
                            key=override_key,
                        )
                        if selected_category:
                            st.caption(f"Current editable category: {selected_category}")

    with tab_actions:
        st.caption(
            "Action panels are fully editable and exported with SNOMED linkage for EMIS integration."
        )
        for role in SUMMARY_ROLES:
            role_data = role_summaries.get(role, {})
            role_conf = float(role_data.get("confidence_score", 0.0)) or float(
                system_unified_confidence
            )
            role_visual = confidence_visual(role_conf, threshold=confidence_threshold)
            action_items_key = f"{selected_document}_{role}_action_items"
            add_action_key = f"{selected_document}_{role}_add_action"
            panel_title = ROLE_ACTION_PANEL_TITLE.get(role, f"{role.title()} Actions")

            with st.expander(
                f"{role_visual['icon']} {panel_title} ({role_visual['label']} - {role_conf:.3f})",
                expanded=(role == "clinician"),
            ):
                if role_conf >= confidence_threshold:
                    st.success("✅ High confidence action panel pre-filled from AI output.")
                else:
                    st.warning("⚠️ Low confidence action panel requires manual review.")

                if st.button(f"Add {panel_title} Item", key=add_action_key):
                    actions = list(st.session_state.get(action_items_key, []))
                    actions.append(
                        {
                            "action_text": "",
                            "due_date": (date.today() + timedelta(days=7)).isoformat(),
                            "priority": "Medium",
                            "assignee": role.title(),
                            "snomed_code": "",
                        }
                    )
                    st.session_state[action_items_key] = actions
                    st.rerun()

                action_rows = list(st.session_state.get(action_items_key, []))
                if not action_rows:
                    st.info("No actions added yet. Use the Add button to create role-specific tasks.")
                remove_index = None

                for idx, action in enumerate(action_rows):
                    text_key = f"{selected_document}_{role}_action_text_{idx}"
                    due_key = f"{selected_document}_{role}_action_due_{idx}"
                    priority_key = f"{selected_document}_{role}_action_priority_{idx}"
                    assignee_key = f"{selected_document}_{role}_action_assignee_{idx}"
                    snomed_key = f"{selected_document}_{role}_action_snomed_{idx}"
                    remove_key = f"{selected_document}_{role}_action_remove_{idx}"

                    priority_default = str(action.get("priority", "Medium")).title()
                    if priority_default not in ACTION_PRIORITY_OPTIONS:
                        priority_default = "Medium"

                    _setdefault_state(text_key, str(action.get("action_text", "")))
                    _setdefault_state(due_key, _to_date(action.get("due_date")))
                    _setdefault_state(priority_key, priority_default)
                    _setdefault_state(
                        assignee_key, str(action.get("assignee", role.title()))
                    )

                    snomed_options = [""] + list(snomed_codes_for_actions)
                    current_snomed = str(action.get("snomed_code", "")).strip()
                    if current_snomed and current_snomed not in snomed_options:
                        snomed_options.append(current_snomed)
                    _setdefault_state(
                        snomed_key,
                        current_snomed if current_snomed in snomed_options else "",
                    )

                    row_cols = st.columns([4, 2, 2, 2, 2, 1])
                    with row_cols[0]:
                        st.text_input(
                            f"{role.title()} Action {idx + 1}",
                            key=text_key,
                            placeholder="Enter action item",
                        )
                    with row_cols[1]:
                        st.date_input("Due Date", key=due_key)
                    with row_cols[2]:
                        st.selectbox(
                            "Priority",
                            options=list(ACTION_PRIORITY_OPTIONS),
                            key=priority_key,
                        )
                    with row_cols[3]:
                        st.text_input("Assignee", key=assignee_key)
                    with row_cols[4]:
                        st.selectbox(
                            "SNOMED",
                            options=snomed_options,
                            key=snomed_key,
                            format_func=lambda value: value if value else "Not linked",
                        )
                    with row_cols[5]:
                        st.markdown("&nbsp;")
                        if st.button("Remove", key=remove_key):
                            remove_index = idx

                    action_rows[idx] = {
                        "action_text": st.session_state.get(text_key, "").strip(),
                        "due_date": _to_date(st.session_state.get(due_key)).isoformat(),
                        "priority": st.session_state.get(priority_key, "Medium"),
                        "assignee": st.session_state.get(assignee_key, "").strip(),
                        "snomed_code": st.session_state.get(snomed_key, "").strip(),
                    }

                if remove_index is not None:
                    action_rows.pop(remove_index)
                    st.session_state[action_items_key] = action_rows
                    st.rerun()

                st.session_state[action_items_key] = action_rows

# Action row
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)

if c1.button("Save Review Edits"):
    current_payload = build_review_payload(
        document_id=selected_document,
        current_user=current_user,
        role_summaries=role_summaries,
        snomed_entities=snomed_entities,
        confidence_bundle=confidence_bundle,
        system_score=system_unified_confidence,
        effective_score=effective_unified_confidence,
        threshold=confidence_threshold,
        confidence_mode=st.session_state[confidence_mode_key],
    )
    try:
        if audit_logger is None:
            raise RuntimeError("Audit logger unavailable. Cannot save without audit logging.")

        changes_logged = log_review_changes(
            audit_logger=audit_logger,
            document_id=selected_document,
            current_user=current_user,
            previous_state=saved_review_state,
            current_state=current_payload,
        )
        save_review_state(review_state_path, current_payload)
        st.success(f"Review saved successfully. Audit entries created: {changes_logged}")
    except Exception as exc:
        st.error(f"Failed to save review edits: {exc}")

if c2.button("Approve All & Export to EMIS"):
    current_payload = build_review_payload(
        document_id=selected_document,
        current_user=current_user,
        role_summaries=role_summaries,
        snomed_entities=snomed_entities,
        confidence_bundle=confidence_bundle,
        system_score=system_unified_confidence,
        effective_score=effective_unified_confidence,
        threshold=confidence_threshold,
        confidence_mode=st.session_state[confidence_mode_key],
    )

    try:
        if audit_logger is None:
            raise RuntimeError("Audit logger unavailable.")

        entities_approved = len(current_payload.get("snomed", {}).get("entities", []))
        audit_logger.log_approve_all(
            document_id=selected_document,
            user_id=current_user,
            entities_approved=entities_approved,
        )

        export_payload = {
            "document_id": selected_document,
            "exported_at": datetime.utcnow().isoformat() + "Z",
            "document_review": current_payload.get("document_review", {}),
            "component_scores": confidence_bundle.get("component_scores", {}),
            "summaries": current_payload.get("summaries", {}),
            "snomed_entities": current_payload.get("snomed", {}).get("entities", []),
            "role_based_actions": current_payload.get("role_based_actions", {}),
        }
        emis_result = export_to_emis(
            document_id=selected_document,
            validated_payload=export_payload,
            user_id=current_user,
            audit_logger=audit_logger,
        )
        export_json = json.dumps(export_payload, indent=2)
        export_key = f"{selected_document}_emis_export_json"
        st.session_state[export_key] = export_json
        if emis_result.get("success"):
            st.success("Document approved and exported to EMIS successfully.")
        else:
            st.warning(
                "Document approved, but EMIS export failed and was queued for retry."
            )
    except Exception as exc:
        st.error(f"Failed to approve/export: {exc}")

export_key = f"{selected_document}_emis_export_json"
if st.session_state.get(export_key):
    c2.download_button(
        label="Download EMIS Export JSON",
        data=st.session_state[export_key],
        file_name=f"{selected_document}_emis_export.json",
        mime="application/json",
    )

if c3.button("Flag for Specialist Review"):
    try:
        if audit_logger:
            audit_logger.log_flag_for_review(
                document_id=selected_document,
                user_id=current_user,
                reason="Manual flag by clinician from enhanced review interface",
            )
        st.warning("Document flagged for specialist review.")
    except Exception as exc:
        st.error(f"Failed to flag for review: {exc}")

if c4.button("Download JSON Audit Trail"):
    try:
        if audit_logger:
            json_export = audit_logger.export_audit_trail_to_json(document_id=selected_document)
            st.download_button(
                label="Click to Download",
                data=json_export,
                file_name=f"{selected_document}_audit_trail.json",
                mime="application/json",
            )
            st.success("Audit trail ready for download.")
        else:
            st.error("Audit logger not available.")
    except Exception as exc:
        st.error(f"Failed to generate audit trail: {exc}")

# Sidebar audit viewer
st.sidebar.markdown("---")
st.sidebar.header("Audit Trail")
if st.sidebar.button("View Document Audit History"):
    if audit_logger:
        history = audit_logger.get_audit_trail_by_document(selected_document, limit=30)
        if history:
            st.sidebar.write(f"**{len(history)} entries found**")
            for entry in history:
                timestamp = entry.get("timestamp", "")[:19].replace("T", " ")
                change_type = entry.get("change_type", "UNKNOWN")
                user = entry.get("user_id", "unknown")
                st.sidebar.text(f"{timestamp}\n  {change_type}\n  by {user}")
        else:
            st.sidebar.info("No audit history for this document.")
    else:
        st.sidebar.error("Audit logger not available.")
