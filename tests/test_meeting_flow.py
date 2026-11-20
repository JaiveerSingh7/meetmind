
import json
from pathlib import Path

from flows.meeting_flow import MeetingFlow


transcript = Path(
    "sample_meeting.txt"
).read_text(encoding="utf-8-sig")


print("\n=== STARTING MEETMIND FLOW ===")

flow = MeetingFlow()

report = flow.kickoff(
    inputs={"transcript": transcript}
)

print("\n=== FINAL MEETING REPORT ===")

print(
    json.dumps(report, indent=2)
)

print("\n=== FLOW COMPLETED ===")

print("Decisions:", len(report["decisions"]))
print("Action items:", len(report["action_items"]))
print(
    "Verification results:",
    len(report["verification"])
)
