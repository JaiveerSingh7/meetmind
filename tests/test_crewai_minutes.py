
from pathlib import Path

from core.segmenter import segment_transcript
from agents.crewai_minutes_agent import (
    run_crewai_minutes_agent
)


transcript = Path(
    "sample_meeting.txt"
).read_text(encoding="utf-8-sig")

segments = segment_transcript(transcript)

print("\n=== RUNNING MINUTES AGENT ===")

result = run_crewai_minutes_agent(segments)

print("\n=== GENERATED MEETING MINUTES ===")

print(result.model_dump_json(indent=2))

segment_map = {
    segment.id: segment
    for segment in segments
}

print("\n=== EVIDENCE ===")

for point in result.discussion_points:
    print("\nTopic:", point.topic)
    print("Summary:", point.summary)

    for sid in point.evidence_segment_ids:
        segment = segment_map[sid]
        print(f"[{sid}] {segment.speaker}: {segment.text}")

for item in result.unresolved_items:
    print("\nUnresolved:", item.description)

    for sid in item.evidence_segment_ids:
        segment = segment_map[sid]
        print(f"[{sid}] {segment.speaker}: {segment.text}")
