
from core.segmenter import TranscriptSegment


class TranscriptTools:
    def __init__(
        self,
        segments: list[TranscriptSegment]
    ):
        self.segments = segments

        self.segment_map = {
            segment.id: segment
            for segment in segments
        }

    def get_segment(self, segment_id: str):
        """
        Retrieve one segment by ID.
        """
        segment = self.segment_map.get(segment_id)

        if segment is None:
            return None

        return segment.model_dump()

    def search_transcript(self, query: str):
        """
        Search transcript segments using keywords.
        """
        query = query.lower()

        results = []

        for segment in self.segments:
            if query in segment.text.lower():
                results.append(segment.model_dump())

        return results

    def get_context(
        self,
        segment_id: str,
        window: int = 2
    ):
        """
        Retrieve surrounding segments.
        """
        if segment_id not in self.segment_map:
            return []

        index = self.segments.index(
            self.segment_map[segment_id]
        )

        start = max(0, index - window)
        end = min(
            len(self.segments),
            index + window + 1
        )

        return [
            segment.model_dump()
            for segment in self.segments[start:end]
        ]
