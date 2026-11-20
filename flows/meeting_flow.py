
from crewai.flow.flow import Flow, start, listen

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


class MeetingFlow(Flow):

    @start()
    def prepare_transcript(self):
        """
        Step 1: Prepare transcript segments.
        """
        transcript = self.state.get("transcript", "")

        if not transcript.strip():
            raise ValueError("Transcript cannot be empty.")

        segments = segment_transcript(transcript)

        self.state["segments"] = segments

        print(
            f"[Flow] Created {len(segments)} segments."
        )

    @listen(prepare_transcript)
    def extract_information(self):
        """
        Step 2: Run the three extraction crews.
        """
        segments = self.state["segments"]

        print("[Flow] Running Decision Agent...")
        decisions = run_crewai_decision_agent(
            segments
        )

        print("[Flow] Running Action Agent...")
        actions = run_crewai_action_agent(
            segments
        )

        print("[Flow] Running Minutes Agent...")
        minutes = run_crewai_minutes_agent(
            segments
        )

        self.state["decisions"] = decisions
        self.state["actions"] = actions
        self.state["minutes"] = minutes

        print("[Flow] Extraction completed.")

    @listen(extract_information)
    def verify_information(self):
        """
        Step 3: Verify all extracted claims.
        """
        print("[Flow] Running Verification Agent...")

        verification = run_verification_agent(
            segments=self.state["segments"],
            decisions=self.state["decisions"],
            actions=self.state["actions"],
            minutes=self.state["minutes"]
        )

        self.state["verification"] = verification

        print("[Flow] Verification completed.")

    @listen(verify_information)
    def generate_report(self):
        """
        Step 4: Combine all results.
        """
        report = {
            "summary": self.state["minutes"].summary,

            "discussion_points": [
                item.model_dump()
                for item in self.state[
                    "minutes"
                ].discussion_points
            ],

            "decisions": [
                item.model_dump()
                for item in self.state[
                    "decisions"
                ].decisions
            ],

            "action_items": [
                item.model_dump()
                for item in self.state[
                    "actions"
                ].action_items
            ],

            "unresolved_items": [
                item.model_dump()
                for item in self.state[
                    "minutes"
                ].unresolved_items
            ],

            "verification": [
                item.model_dump()
                for item in self.state[
                    "verification"
                ].results
            ],

            "review_status": "pending"
        }

        self.state["report"] = report

        print("[Flow] Meeting report generated.")

        return report
