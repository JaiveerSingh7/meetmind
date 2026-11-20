
from pathlib import Path

from core.segmenter import segment_transcript
from agents.crewai_decision_agent import (
    run_crewai_decision_agent
)


transcript = Path(
    "sample_meeting.txt"
).read_text(encoding="utf-8-sig")

segments = segment_transcript(transcript)

print("\n=== STARTING CREWAI ===")

result = run_crewai_decision_agent(segments)

print("\n=== FINAL DECISIONS ===")

print(result.model_dump_json(indent=2))

segment_map = {
    segment.id: segment
    for segment in segments
}

for decision in result.decisions:

    print("\nDecision:", decision.description)

    for segment_id in decision.evidence_segment_ids:
        segment = segment_map[segment_id]

        print(
            f"[{segment.id}] "
            f"{segment.speaker or 'Unknown'}: "
            f"{segment.text}"
        )
