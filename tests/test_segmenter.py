
from pathlib import Path

from core.segmenter import segment_transcript
from tools.transcript_tools import TranscriptTools


transcript = Path(
    "sample_meeting.txt"
).read_text(encoding="utf-8-sig")

segments = segment_transcript(transcript)

tools = TranscriptTools(segments)

print("\n=== SEGMENTS ===")

for segment in segments:
    print(segment.model_dump())

print("\n=== SEARCH: PostgreSQL ===")

print(
    tools.search_transcript("PostgreSQL")
)

print("\n=== CONTEXT FOR S3 ===")

print(
    tools.get_context("S3")
)
