
from pathlib import Path

from core.segmenter import segment_transcript

from agents.crewai_decision_agent import (
    run_crewai_decision_agent
)

from agents.crewai_action_agent import (
    run_crewai_action_agent
)

from agents.crewai_minutes_agent import (
    run_crewai_minutes_agent
)

from agents.crewai_verification_agent import (
    run_verification_agent
)


transcript = Path(
    "sample_meeting.txt"
).read_text(encoding="utf-8-sig")

segments = segment_transcript(transcript)

print("\n=== DECISION AGENT ===")
decisions = run_crewai_decision_agent(segments)

print("\n=== ACTION AGENT ===")
actions = run_crewai_action_agent(segments)

print("\n=== MINUTES AGENT ===")
minutes = run_crewai_minutes_agent(segments)

print("\n=== VERIFICATION AGENT ===")

verification = run_verification_agent(
    segments=segments,
    decisions=decisions,
    actions=actions,
    minutes=minutes
)

print("\n=== VERIFICATION RESULTS ===")

print(verification.model_dump_json(indent=2))

print("\n=== SUMMARY ===")

for result in verification.results:
    print(
        f"{result.claim_id}: "
        f"{result.status.upper()}"
    )
    print(f"Reason: {result.explanation}\n")
