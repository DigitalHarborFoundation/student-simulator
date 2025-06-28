import csv
import os
import random
from typing import Annotated, Any, Dict, List, Optional

from pydantic import Field, validate_arguments

from studentsimulator.activity_provider import FixedFormAssessment, Item
from studentsimulator.general import Model, SkillSpace
from studentsimulator.math import logistic, logit


class SkillState(Model):
    skill_name: str = Field(
        pattern=r"^\w+$", description="Name of the skill, alphanumeric only"
    )
    skill_level: float = Field(
        ge=0.0, le=1.0, description="Current skill level (0.0 to 1.0)"
    )


class Student(Model):
    name: Optional[str] = "None"  # just for use when printing for demos
    skill_space: SkillSpace  # list of skill IDs available to this student
    skill_state: Optional[Dict[str, "SkillState"]] = Field(
        default_factory=dict
    )  # current skill states
    history: Optional[List[Any]] = []

    def print_history(self):
        """Nicely print the student's history."""
        if not self.history:
            print("No history available.")
        print("\n".join(str(event) for event in self.history))

    def __str__(self):
        rep = f"Student(name={self.name}, id={self.id})"
        if self.skill_state is not None:
            for skill in self.skill_state:
                skill_state = self.skill_state[skill]
                rep += "\n" + f"    {skill}: {skill_state.skill_level:.2f}"
        return rep

    def set_skill_values(self, skill_values: Dict[str, float]) -> "Student":
        """Set initial skill levels for the student."""
        for skill_name, level in skill_values.items():
            if self.skill_state is not None and skill_name in self.skill_state:
                self.skill_state[skill_name].skill_level = level
            else:
                # If the skill is not registered, create a new state
                if self.skill_state is None:
                    self.skill_state = {}
                self.skill_state[skill_name] = SkillState(
                    skill_name=skill_name, skill_level=level
                )

    def initialize_skill_values(
        self,
        default_learning_prob: Annotated[
            float, "Chance of learning the concept after a single learning event"
        ] = 0.7,
        practice_count: Annotated[
            int, "Number of practice events to simulate. Default=10."
        ] = 0,
        include_in_history: Annotated[
            bool, "Whether to include practice events in the student history."
        ] = False,
    ):
        pass

    # learning_history: Optional[list[LearningEvent]] = []  # complete history
    # current_skills: Dict[str, "SkillState"] = Field(default_factory=dict)  # current skill states

    # skill_state: Optional[SkillState] = None

    def initialize(self):
        # generates student history and/or skill
        pass

    @validate_arguments
    def take_test(self, test: FixedFormAssessment, timestamp=0) -> None:
        """Simulate taking a test with no formative feedback."""
        responses = []
        for item in test:
            response = self.engage(
                group=test, item=item, feedback=False, timestamp=timestamp
            )
            responses.append(response)
        test_results = BehaviorEventCollection(
            student_id=self.id,
            behavioral_events=responses,
        )
        self.history.append(test_results)

        return test_results

    def engage(self, item: "Item", feedback=False, **kwargs) -> "BehaviorEvent":
        """Simulate engaging with an item, and returning a response."""
        prob_correct = self.get_prob_correct(item)
        correct = 1 if random.random() < prob_correct else 0
        return BehaviorEvent(
            student_id=self.id, engagement_object=item, score=correct, **kwargs
        )

    def get_prob_correct(self, item: "Item") -> float:
        """Calculate probability of correct response based on skill state."""
        skill_level_01 = self.skill_state.get(item.skill.name).skill_level
        skill_level_logit = logit(skill_level_01)

        return item.guess + (1 - item.slip - item.guess) * logistic(
            item.discrimination * (skill_level_logit - item.difficulty)
        )


class BehaviorEventCollection(Model):
    """Group of behavior events, typically for an activity or assessment."""

    student_id: int
    timestamp: int = None
    behavioral_events: List["BehaviorEvent"] = Field(default_factory=list)

    @property
    def percent_correct(self) -> float:
        """Calculate the average score of all behavior events."""
        if not self.behavioral_events:
            return None
        total_score = sum(
            event.score for event in self.behavioral_events if event.score is not None
        )
        return total_score / len(self.behavioral_events) * 100


class BehaviorEvent(Model):
    student_id: int
    timestamp: int = None
    engagement_object: Item
    group: Any = None  # Optional group or activity this behavior is part of
    # behavior: "BehaviorRepresentation" = None
    score: Optional[float] = None

    def __str__(self):
        return (
            f"id={self.id}, type={str(self.engagement_object.__class__.__name__)}, "
            f"score={self.score}, timestamp={self.timestamp}"
        )


Student.model_rebuild()


def prepare_directory(filename: str) -> str:
    """Ensure the directory for the given filename exists."""
    if not filename.endswith(".csv"):
        filename += ".csv"
    dir = os.path.dirname(filename)
    if dir == "":
        # If no directory is specified, use the current directory
        dir = os.getcwd()
    os.makedirs(dir, exist_ok=True)
    return os.path.join(dir, filename)


def save_student_profile_to_csv(students: List[Student], filename: str) -> None:
    """Save student skill states to a CSV file."""

    path = prepare_directory(filename)
    with open(path, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write header
        header = [
            "student_id",
            "student_name",
            "skill_name",
            "skill_level",
            "skill_level_logit",
        ]
        writer.writerow(header)

        # Write each student's skill state
        for student in students:
            for skill_name, skill_state in student.skill_state.items():
                writer.writerow(
                    [
                        student.id,
                        student.name,
                        skill_state.skill_name,
                        skill_state.skill_level,
                        (
                            round(logit(skill_state.skill_level), 4)
                            if skill_state.skill_level is not None
                            else None
                        ),
                    ]
                )


def save_student_activity_to_csv(students: List[Student], filename: str) -> None:
    """Save student activity (behavior events) to a CSV file."""

    path = prepare_directory(filename)
    with open(path, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write header
        header = ["student_id", "timestamp", "engagement_object", "score", "group"]
        writer.writerow(header)

        # Write each student's behavior events
        for student in students:
            for event in student.history:
                if isinstance(event, BehaviorEventCollection):
                    for behavior_event in event.behavioral_events:
                        writer.writerow(
                            [
                                student.id,
                                behavior_event.timestamp,
                                behavior_event.engagement_object.id,
                                behavior_event.score,
                                behavior_event.group.id,
                            ]
                        )
                elif isinstance(event, BehaviorEvent):
                    writer.writerow(
                        [
                            student.id,
                            event.timestamp,
                            event.engagement_object.id,
                            event.score,
                            event.group.id,
                        ]
                    )


#     def perform_activity(self, activity: "Activity") -> None:
#         # get next activity
#         # perform activity to generate behavior
#         # get feedback
#         # update skills based on activity type and feedback
#         pass

#     def get_skill_level(self, skill_id: str) -> float:
#         """Get current proficiency level for a skill."""
#         if skill_id not in self.current_skills:
#             return 0.0
#         return self.current_skills[skill_id].current_proficiency

#     def get_skill_state(self, skill_id: str) -> "SkillState":
#         """Get complete skill state for a skill."""
#         if skill_id not in self.current_skills:
#             self.current_skills[skill_id] = SkillState(skill_id=skill_id)
#         return self.current_skills[skill_id]

#     def apply_intervention(self, intervention: "InterventionEvent", rng: Any) -> bool:
#         """Apply intervention and return whether learning occurred."""
#         if not intervention.target_skill:
#             return False

#         skill_state = self.get_skill_state(intervention.target_skill)

#         # Check if learning occurs (probabilistic)
#         if rng.random() < intervention.learning_probability:
#             # Student learns the skill
#             skill_state.proficiency_state = "baseline"
#             skill_state.baseline_proficiency = intervention.baseline_proficiency
#             skill_state.current_proficiency = intervention.baseline_proficiency

#             # Record learning event
#             event = LearningEvent(
#                 timestamp=intervention.timestamp,
#                 event_type="skill_learned",
#                 skill_changes={
#                     intervention.target_skill: intervention.baseline_proficiency
#                 },
#                 context={"intervention_type": intervention.intervention_type},
#             )
#             self.learning_history.append(event)
#             return True
#         return False

#     def practice_skill(
#         self, skill_id: str, practice_effectiveness: float, timestamp: int
#     ) -> None:
#         """Update skill proficiency through practice."""
#         skill_state = self.get_skill_state(skill_id)

#         if skill_state.proficiency_state == "no_learning":
#             # No effect if skill not yet learned
#             return

#         # Increase proficiency up to maximum
#         old_proficiency = skill_state.current_proficiency
#         skill_state.current_proficiency = min(
#             skill_state.current_proficiency + practice_effectiveness,
#             skill_state.max_proficiency,
#         )

#         # Check if mastery achieved
#         if skill_state.current_proficiency >= skill_state.max_proficiency:
#             skill_state.proficiency_state = "mastery"

#         skill_state.practice_count += 1
#         skill_state.last_practiced = timestamp

#         # Record practice event if there was improvement
#         improvement = skill_state.current_proficiency - old_proficiency
#         if improvement > 0:
#             event = LearningEvent(
#                 timestamp=timestamp,
#                 event_type="skill_practiced",
#                 skill_changes={skill_id: improvement},
#             )
#             self.learning_history.append(event)

#     def record_behavioral_event(self, behavioral_event: "BehavioralEvent") -> None:
#         """Record a behavioral event in learning history."""
#         # Convert behavioral event to internal learning event
#         learning_event = LearningEvent(
#             timestamp=behavioral_event.timestamp,
#             event_type="behavioral_event",
#             context={
#                 "behavior": behavioral_event.behavior.model_dump(),
#                 "feedback": (
#                     behavioral_event.feedback.model_dump()
#                     if behavioral_event.feedback
#                     else None
#                 ),
#                 **behavioral_event.context,
#             },
#         )
#         self.learning_history.append(learning_event)

#     def record_intervention_event(
#         self, intervention_event: "InterventionEvent"
#     ) -> None:
#         """Record an intervention event in learning history."""
#         learning_event = LearningEvent(
#             timestamp=intervention_event.timestamp,
#             event_type="intervention_event",
#             context={
#                 "intervention_type": intervention_event.intervention_type,
#                 "target_skill": intervention_event.target_skill,
#                 "intervention_data": intervention_event.intervention_data,
#             },
#         )
#         self.learning_history.append(learning_event)

#     def record_response(
#         self, timestamp: int, item_id: str, response: int, **context: Any
#     ) -> None:
#         """Record a response event (backward compatibility)."""
#         # Convert to new behavioral event structure
#         behavior = BehaviorRepresentation(
#             behavior_type="item_selection",
#             item_id=item_id,
#             selected_option=str(response),  # Convert response to string option
#         )

#         feedback = None
#         if "intervention_type" in context:
#             # This was an intervention-related response, create minimal feedback
#             feedback = Feedback(
#                 feedback_type="binary",
#                 correct=bool(response),  # Simple assumption: 1=correct, 0=incorrect
#             )

#         behavioral_event = BehavioralEvent(
#             student_id=self.id,
#             timestamp=timestamp,
#             behavior=behavior,
#             feedback=feedback,
#             context=context,
#         )

#         self.record_behavioral_event(behavioral_event)


# class SkillsConfig(BaseModel):
#     skills: List[Skill] = []


# class SkillState(BaseModel):
#     skill_id: str
#     proficiency_state: str = "no_learning"  # "no_learning", "baseline", "mastery"
#     baseline_proficiency: float = 0.0  # proficiency after intervention
#     current_proficiency: float = 0.0  # current proficiency (increases with practice)
#     max_proficiency: float = 1.0  # maximum achievable proficiency
#     practice_count: int = 0
#     last_practiced: Optional[int] = None  # timestamp

#     def get_probability_correct(
#         self, item_guess: float, item_slip: float, mode: str = "hybrid"
#     ) -> float:
#         """Calculate P(correct) based on current proficiency state and item parameters."""
#         if mode == "cdm":
#             # CDM binary model: learned vs not learned
#             if self.current_proficiency > 0.5:  # "Learned" threshold
#                 return 1.0 - item_slip
#             else:
#                 return item_guess
#         else:
#             # Original hybrid model
#             if self.proficiency_state == "no_learning":
#                 return item_guess
#             elif self.proficiency_state == "mastery":
#                 return 1.0 - item_slip
#             else:  # baseline or intermediate
#                 # Linear interpolation between baseline and mastery
#                 progress = min(self.current_proficiency / self.max_proficiency, 1.0)
#                 baseline_prob = self.baseline_proficiency
#                 mastery_prob = 1.0 - item_slip
#                 return baseline_prob + progress * (mastery_prob - baseline_prob)
