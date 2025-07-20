from typing import List, TypedDict, Union


class State(TypedDict):
    student_essay: str
    image_url: str
    task_response: str
    coherence_and_cohesion: str
    lexical_resource: str
    grammatical_range_and_accuracy: str
    estimated_band_score: float
    aggregated_result: str
