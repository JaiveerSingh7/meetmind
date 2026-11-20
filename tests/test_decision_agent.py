
from pathlib import Path

from core.segmenter import segment_transcript
from agents.decision_agent import run_decision_agent


# Read sample meeting
transcript = Path(
    "sample_meeting.txt"
).read_text(encoding="utf-8-sig")

# Create numbered segments
segments = segment_transcript(transcript)

print("\n=== TRANSCRIPT ===")

for segment in segments:
    print(
        f"{segment.id}: "
        f"{segment.speaker}: "
        f"{segment.text}"
    )

# Execute Decision Agent
print("\n=== RUNNING DECISION AGENT ===")

result = run_decision_agent(segments)

print("\n=== EXTRACTED DECISIONS ===")

print(
    result.model_dump_json(indent=2)
)

# Show actual evidence text
segment_map = {
    segment.id: segment
    for segment in segments
}

for decision in result.decisions:

    print("\nDecision:", decision.description)

    print("Evidence:")

    for segment_id in decision.evidence_segment_ids:
        segment = segment_map[segment_id]

        print(
            f"  [{segment.id}] "
            f"{segment.speaker}: "
            f"{segment.text}"
        )
