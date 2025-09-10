from typing import List, Optional

from pydantic import Field

from studentsimulator.general import Model
from studentsimulator.item import Item
from studentsimulator.skill import Skill


class BehaviorEventCollection(Model):
    """Group of behavior events, typically for an activity or assessment."""

    student_id: int
    timestamp_in_days_since_initialization: int = 0
    behavioral_events: List["BehaviorEvent"] = Field(
        default_factory=list,
        description="List of behavior events (can be BehaviorEvent or any subclass thereof)",
    )
    activity_provider_name: Optional[str] = None

    @property
    def percent_correct(self) -> float:
        """Calculate the average score of all behavior events."""
        if not self.behavioral_events:
            return 0.0
        total_score = sum(
            event.score
            for event in self.behavioral_events
            if (isinstance(event, ItemResponseEvent) and event.score is not None)
        )
        return total_score / len(self.behavioral_events) * 100


class BehaviorEvent(Model):
    student_id: int
    timestamp_in_days_since_initialization: int = 0
    activity_provider_name: Optional[str] = None
    category: Optional[str] = None
    # behavior: "BehaviorRepresentation" = None


class ItemResponseEvent(BehaviorEvent):
    skill: Optional[Skill] = None
    item: Optional[Item] = None
    score: Optional[float] = None
    prob_correct: Optional[float] = None
    feedback_given: Optional[bool] = False
    practice_increment_logit: Optional[float] = 0.0
    category: Optional[str] = "ItemResponse"

    def __str__(self):
        return f"""ItemResponseEvent(
    timestamp={self.timestamp_in_days_since_initialization}
    id={self.id},
    type={str(self.item.__class__.__name__)},
    skill={self.skill.name},
    score={self.score},
    prob_correct={self.prob_correct},
    feedback_given={self.feedback_given},
    practice_increment_logit={self.practice_increment_logit}
    )"""

    @property
    def engagement_object(self) -> Optional[Item]:
        return self.item


# ---------------------------------------------------------------------------
# New event type: WaitEvent
# ---------------------------------------------------------------------------


class WaitEvent(BehaviorEvent):
    """Represents the passage of time with no direct student interaction.

    Storing this event allows us to replay a student's history later and
    reconstruct skill trajectories without recording every single skill state
    snapshot.  The duration (in days) is stored so that forgetting can be
    re-applied deterministically during a replay.
    """

    days_waited: int = 0
    category: str = "Wait"

    def __str__(self):
        return f"""WaitEvent(
    timestamp={self.timestamp_in_days_since_initialization}
    id={self.id},
    days_waited={self.days_waited},
    )"""


class LearningEvent(BehaviorEvent):
    skill: Optional[Skill] = None
    initial_learned: Optional[bool] = None
    final_learned: Optional[bool] = None
    initial_skill_level: Optional[float] = None
    final_skill_level: Optional[float] = None
    probability_of_learning_with_prerequisites: Optional[float] = None
    had_prerequisites: Optional[bool] = None
    category: str = "Learning"

    def __str__(self):
        return f"""LearningEvent(
    timestamp={self.timestamp_in_days_since_initialization}
    id={self.id},
    skill={self.skill.name},
    initial_learned={self.initial_learned},
    final_learned={self.final_learned},
    initial_skill_level={self.initial_skill_level},
    final_skill_level={self.final_skill_level},
    probability_of_learning_with_prerequisites={self.probability_of_learning_with_prerequisites},
    had_prerequisites={self.had_prerequisites},
    activity_provider_name={self.activity_provider_name},
    )"""


# Rebuild models to resolve forward references
BehaviorEventCollection.model_rebuild()
BehaviorEvent.model_rebuild()
ItemResponseEvent.model_rebuild()
WaitEvent.model_rebuild()
LearningEvent.model_rebuild()
