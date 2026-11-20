from pydantic import BaseModel


class TranscriptSegment(BaseModel):
    id: str
    speaker: str | None = None
    text: str


def segment_transcript(
    transcript: str
) -> list[TranscriptSegment]:
    """
    Splits a meeting transcript into numbered segments.

    Assumes one speaker turn per non-empty line.
    """

    segments = []

    lines = transcript.splitlines()

    for line in lines:
        line = line.strip()

        if not line:
            continue

        speaker = None
        text = line

        if ":" in line:
            possible_speaker, content = line.split(":", 1)

            # Avoid treating long sentences as speaker names.
            if len(possible_speaker.strip()) <= 40:
                speaker = possible_speaker.strip()
                text = content.strip()

        if not text:
            continue

        segment = TranscriptSegment(
            id=f"S{len(segments) + 1}",
            speaker=speaker,
            text=text
        )

        segments.append(segment)

    return segments

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
