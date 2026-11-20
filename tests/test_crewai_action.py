
from pathlib import Path

from core.segmenter import segment_transcript
from agents.crewai_action_agent import (
    run_crewai_action_agent
)


transcript = Path(
    "sample_meeting.txt"
).read_text(encoding="utf-8-sig")

segments = segment_transcript(transcript)

print("\n=== STARTING CREWAI ACTION AGENT ===")

result = run_crewai_action_agent(segments)

print("\n=== EXTRACTED ACTION ITEMS ===")

print(result.model_dump_json(indent=2))

segment_map = {
    segment.id: segment
    for segment in segments
}

for action in result.action_items:

    print("\nTask:", action.task)
    print("Owner:", action.owner)
    print("Deadline:", action.deadline)

    print("Evidence:")

    for segment_id in action.evidence_segment_ids:
        segment = segment_map[segment_id]

        print(
            f"[{segment.id}] "
            f"{segment.speaker or 'Unknown'}: "
            f"{segment.text}"
        )
