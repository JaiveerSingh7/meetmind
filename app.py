
import json
from copy import deepcopy

import streamlit as st

from core.segmenter import segment_transcript
from flows.meeting_flow import MeetingFlow


st.set_page_config(
    page_title="MeetMind",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 MeetMind")
st.caption(
    "Agentic AI Meeting Assistant | "
    "CrewAI + Qwen3 8B + Human Review"
)

st.divider()


# ------------------------------------------
# HELPERS
# ------------------------------------------

SECTIONS = {
    "decisions": ("Decisions", "D"),
    "action_items": ("Action Items", "A"),
    "discussion_points": ("Discussion Points", "P"),
    "unresolved_items": ("Unresolved Items", "U")
}


def clear_review_state():
    """Remove previous report and review widgets."""
    for key in list(st.session_state.keys()):
        if key.startswith("review_"):
            del st.session_state[key]

    st.session_state.pop("reviewed_report", None)


def get_evidence(segment_ids, segment_map):
    """Retrieve original dialogue using evidence IDs."""
    evidence = []

    for sid in segment_ids:
        segment = segment_map.get(sid)

        if segment is not None:
            speaker = (
                f"{segment.speaker}: "
                if segment.speaker else ""
            )

            evidence.append(
                f"[{sid}] {speaker}{segment.text}"
            )

    return evidence


def get_verification_map(report):
    """Index verification verdicts by claim ID."""
    return {
        item["claim_id"]: item
        for item in report.get("verification", [])
    }


def build_reviewed_report():
    """Collect the human's saved edits and decisions."""
    report = deepcopy(st.session_state.raw_report)

    report["summary"] = st.session_state[
        "review_summary"
    ]

    all_reviewed = True

    for section, (_, prefix) in SECTIONS.items():
        for i, item in enumerate(report[section]):
            claim_id = f"{prefix}{i + 1}"
            base = f"review_{claim_id}"

            # Save edited description or task.
            if section == "action_items":
                item["task"] = st.session_state[
                    f"{base}_text"
                ]
                item["owner"] = (
                    st.session_state[f"{base}_owner"].strip()
                    or None
                )
                item["deadline"] = (
                    st.session_state[f"{base}_deadline"].strip()
                    or None
                )

            elif section == "discussion_points":
                item["summary"] = st.session_state[
                    f"{base}_text"
                ]

            else:
                field = (
                    "description"
                    if section != "discussion_points"
                    else "summary"
                )
                item[field] = st.session_state[
                    f"{base}_text"
                ]

            status = st.session_state[
                f"{base}_status"
            ]

            item["human_review"] = status.lower()

            if status == "Pending":
                all_reviewed = False

    report["review_status"] = (
        "approved" if all_reviewed
        else "pending"
    )

    return report, all_reviewed


# ------------------------------------------
# TRANSCRIPT INPUT
# ------------------------------------------

st.header("1. Meeting Transcript")

uploaded = st.file_uploader(
    "Upload a meeting transcript",
    type=["txt"]
)

pasted = st.text_area(
    "Or paste the transcript",
    height=180,
    placeholder="Rahul: Let's discuss the project..."
)

transcript = pasted

if uploaded is not None:
    try:
        transcript = uploaded.getvalue().decode(
            "utf-8-sig"
        )
        st.caption(
            "Using uploaded file: "
            f"{uploaded.name}"
        )
    except UnicodeDecodeError:
        transcript = ""
        st.error("Please upload a UTF-8 text file.")


# ------------------------------------------
# ANALYSIS
# ------------------------------------------

if st.button(
    "Analyze Meeting",
    type="primary",
    disabled=not transcript.strip()
):

    if len(transcript) > 24000:
        st.error(
            "This version supports up to "
            "24,000 transcript characters."
        )
    else:
        # Prevent an old report from remaining
        # visible if a new analysis fails.
        st.session_state.pop("raw_report", None)
        st.session_state.pop("source_transcript", None)
        clear_review_state()

        with st.spinner(
            "CrewAI agents are analyzing the meeting..."
        ):
            try:
                flow = MeetingFlow()

                report = flow.kickoff(
                    inputs={"transcript": transcript}
                )

                st.session_state.raw_report = report
                st.session_state.source_transcript = (
                    transcript
                )

                st.success(
                    "Multi-agent analysis completed!"
                )

            except Exception as exc:
                st.error(
                    f"Analysis failed: {exc}"
                )


# ------------------------------------------
# RESULTS AND HUMAN REVIEW
# ------------------------------------------

if "raw_report" in st.session_state:

    report = st.session_state.raw_report

    segments = segment_transcript(
        st.session_state.source_transcript
    )

    segment_map = {
        s.id: s for s in segments
    }

    verification_map = get_verification_map(
        report
    )

    st.divider()

    st.header("2. Agent Results")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Decisions",
        len(report["decisions"])
    )

    col2.metric(
        "Action Items",
        len(report["action_items"])
    )

    col3.metric(
        "Verification Results",
        len(report.get("verification", []))
    )

    st.info(
        "AI verification is advisory. "
        "Review every item before exporting."
    )

    # --------------------------------------
    # REVIEW FORM
    # --------------------------------------

    st.header("3. Human Review")

    st.write(
        "Edit the extracted information, inspect "
        "the original transcript evidence, "
        "then approve or reject each item."
    )

    with st.form("human_review_form"):

        st.subheader("Meeting Summary")

        st.text_area(
            "Edit the summary",
            value=report["summary"],
            key="review_summary",
            height=130
        )

        for section, (title, prefix) in SECTIONS.items():

            st.subheader(title)

            items = report.get(section, [])

            if not items:
                st.info(f"No {title.lower()} extracted.")
                continue

            for i, item in enumerate(items):
                claim_id = f"{prefix}{i + 1}"
                base = f"review_{claim_id}"

                with st.container(border=True):

                    st.markdown(f"**{claim_id}**")

                    if section == "action_items":
                        text_value = item["task"]
                    elif section == "discussion_points":
                        text_value = item["summary"]
                    else:
                        text_value = item["description"]

                    st.text_area(
                        "Extracted information",
                        value=text_value,
                        key=f"{base}_text",
                        height=90
                    )

                    if section == "action_items":

                        c1, c2 = st.columns(2)

                        with c1:
                            st.text_input(
                                "Owner",
                                value=item.get("owner") or "",
                                key=f"{base}_owner"
                            )

                        with c2:
                            st.text_input(
                                "Deadline",
                                value=item.get("deadline") or "",
                                key=f"{base}_deadline"
                            )

                    # Original source evidence
                    evidence = get_evidence(
                        item.get(
                            "evidence_segment_ids", []
                        ),
                        segment_map
                    )

                    with st.expander(
                        "View transcript evidence"
                    ):
                        if evidence:
                            for line in evidence:
                                st.write(line)
                        else:
                            st.warning(
                                "No valid evidence available."
                            )

                    # Verifier's decision
                    verdict = verification_map.get(
                        claim_id
                    )

                    if verdict:
                        status = verdict["status"]

                        if status == "supported":
                            st.success(
                                "AI verification: Supported"
                            )

                        elif status == "unsupported":
                            st.error(
                                "AI verification: Unsupported"
                            )

                        else:
                            st.warning(
                                "AI verification: Needs review"
                            )

                        st.caption(
                            verdict.get("explanation", "")
                        )
                    else:
                        st.warning(
                            "No verification verdict available."
                        )

                    # Human decision
                    st.radio(
                        "Human decision",
                        options=[
                            "Pending",
                            "Approve",
                            "Reject"
                        ],
                        key=f"{base}_status",
                        horizontal=True
                    )

        submitted = st.form_submit_button(
            "Save Human Review",
            type="primary"
        )

    # --------------------------------------
    # SAVE REVIEW
    # --------------------------------------

    if submitted:

        reviewed, complete = build_reviewed_report()

        st.session_state.reviewed_report = reviewed

        if complete:
            st.success(
                "All extracted items have been reviewed."
            )
        else:
            st.warning(
                "Review saved. Some items are still pending."
            )

    # --------------------------------------
    # EXPORT
    # --------------------------------------

    if "reviewed_report" in st.session_state:

        reviewed = st.session_state.reviewed_report

        if reviewed["review_status"] == "approved":

            approved = deepcopy(reviewed)

            # Export only approved items.
            for section in SECTIONS:
                approved[section] = [
                    item
                    for item in approved[section]
                    if item.get("human_review") == "approve"
                ]

            approved["review_status"] = "approved"

            st.divider()
            st.header("4. Export Approved Report")

            st.download_button(
                "Download Approved Meeting Report",
                data=json.dumps(
                    approved,
                    indent=2,
                    ensure_ascii=False
                ),
                file_name="meetmind_approved_report.json",
                mime="application/json"
            )

        else:
            st.info(
                "Complete all pending reviews "
                "to enable export."
            )
