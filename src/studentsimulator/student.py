import csv
import os
import random
from typing import Annotated, Any, Dict, List, Optional, Union

from pydantic import Field

from studentsimulator.activity_provider import FixedFormAssessment, Item
from studentsimulator.general import Model, Skill, SkillSpace
from studentsimulator.math import logistic, logit


class SkillState(Model):
    skill_name: str = Field(
        pattern=r"^\w+$", description="Name of the skill, alphanumeric only"
    )
    skill_level: float = Field(
        ge=0.0, le=1.0, description="Current skill level (0.0 to 1.0)"
    )
    learned: bool = Field(
        default=False,
        description="Whether the skill has been learned cognitively. Practice can't increase the skill level if the skill is not learned.",
    )


class StudentHistory(Model):
    """Tracks student's learning and assessment history."""

    student_id: int
    events: List[Any] = Field(default_factory=list)

    def add_event(self, event: Any) -> None:
        """Add an event to the history."""
        self.events.append(event)

    def get_events(self) -> List[Any]:
        """Get all events in the history."""
        return self.events

    def get_assessment_events(self) -> List["BehaviorEventCollection"]:
        """Get all assessment events."""
        return [
            event for event in self.events if isinstance(event, BehaviorEventCollection)
        ]

    def get_behavior_events(self) -> List["BehaviorEvent"]:
        """Get all individual behavior events."""
        behavior_events = []
        for event in self.events:
            if isinstance(event, BehaviorEventCollection):
                behavior_events.extend(event.behavioral_events)
            elif isinstance(event, BehaviorEvent):
                behavior_events.append(event)
        return behavior_events

    def clear(self) -> None:
        """Clear all events from history."""
        self.events.clear()


class Student(Model):
    name: Optional[str] = "None"  # just for use when printing for demos
    skill_space: SkillSpace  # list of skill IDs available to this student
    skill_state: Dict[str, "SkillState"] = Field(
        default_factory=dict
    )  # current skill states
    history: Optional[StudentHistory] = None

    def __init__(self, **data):
        # First call the parent Model.__init__ for validation and ID assignment
        super().__init__(**data)

        # Initialize history if not provided
        if self.history is None:
            self.history = StudentHistory(student_id=self.id)

        # Initialize all skill levels to 0
        if self.skill_space and not self.skill_state:
            self.skill_state = {}
            for skill in self.skill_space.skills:
                self.skill_state[skill.name] = SkillState(
                    skill_name=skill.name, skill_level=0
                )

    def print_history(self):
        """Nicely print the student's history."""
        if not self.history or not self.history.get_events():
            print("No history available.")
        else:
            print("\n".join(str(event) for event in self.history.get_events()))

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
                self.skill_state[skill_name].learned = True
            else:
                # If the skill is not registered, create a new state
                if self.skill_state is None:
                    self.skill_state = {}
                self.skill_state[skill_name] = SkillState(
                    skill_name=skill_name, skill_level=level
                )
        return self

    def initialize_skill_values(
        self,
        practice_count: Annotated[
            Union[int, list[int]],
            "Number of practice events to simulate. If int, use as is. If list of two ints, draw random int between min and max (inclusive). Default=10. Example: practice_count=[3,7] will randomly choose between 3 and 7 (inclusive).",
        ] = 5,
        include_in_history: Annotated[
            bool, "Whether to include practice events in the student history."
        ] = False,
    ):
        """The goal of this method is to initialize a student's skill values "randomly" while
        still respecting prerequisites.

        To do this, we run students through a series of learning and practice events in the
        topological order of the skill space.
        Learning events have a chance of turning "on" a skill by converting a gate from 0 to 1,
        and then practice events increase the skill level.  Practice events are not effective
        if the learning gate is still 0.
        """

        # Iterate through the skills in topological order
        for skill in self.skill_space.skills:
            # Encounter a learning event
            self.learn(skill)

            # Determine the number of practice events for this skill
            if isinstance(practice_count, int):
                n_practice = practice_count
            elif (
                isinstance(practice_count, (list, tuple))
                and len(practice_count) == 2
                and all(isinstance(x, int) for x in practice_count)
            ):
                n_practice = random.randint(
                    practice_count[0], practice_count[1]
                )  # inclusive of both endpoints
            else:
                raise ValueError(
                    "practice_count must be an int or a list/tuple of two ints."
                )

            # Increment through the practice iterations
            # Skill level will only increase if the skill is learned
            for i in range(n_practice):
                self.practice(skill)
        # TODO: include history of learning and practice events
        return self

    def learn(self, skill: Skill, record_event_in_history: bool = False):
        """Learn a skill.
        Learning happens during a 'learning encounter'.
        First we check to see if the student has the necessary prerequisites, if there are any.
        If they do, they learn with p=probability_of_learning_with_prerequisites.
        If they don't, they learn with p=probability_of_learning_without_prerequisites.
        If the skill is learned during this encounter, this sets the gate skill.learn=True,
        which enables practice to be productive and increase the skill level.
        """

        # get random number
        random_number = random.random()
        # Check to see if the skill has prerequisites
        if self.has_prerequisites(skill):
            if random_number < skill.probability_of_learning_with_prerequisites:
                # Set learned to True. If it was already True, we don't change it.
                self.skill_state[skill.name].learned = True
        else:
            if random_number < skill.probability_of_learning_without_prerequisites:
                self.skill_state[skill.name].learned = True

        # If the skill is learned, we set the skill level to the initial skill level
        if self.skill_state[skill.name].learned:
            # If the skill level is less than the initial skill level, increase it to equal the initial skill level
            if self.skill_state[skill.name].skill_level < skill.initial_skill_level:
                self.skill_state[skill.name].skill_level = skill.initial_skill_level

    def practice(self, skill: Skill):
        """Practice a skill.
        Practice happens during a 'practice encounter'.
        Practice increases the skill level in logit space.
        Practice is only effective if the skill is learned.
        """
        if self.skill_state[skill.name].learned:
            self.skill_state[skill.name].skill_level = logistic(
                logit(self.skill_state[skill.name].skill_level)
                + skill.practice_increment_logit
            )
        # Practice should create a BehaviorEvent that's stored in the student's history
        if self.history:
            self.history.add_event(
                ItemResponseEvent(
                    student_id=self.id,
                    item=None,
                    score=None,
                    feedback_given=True,
                    practice_increment_logit=skill.practice_increment_logit,
                )
            )

    def has_prerequisites(self, skill: Skill) -> bool:
        """Check if the student has the prerequisites for a skill."""
        if not skill.prerequisites.parent_names:
            return True
        if skill.prerequisites.dependence_model == "any":
            return any(
                self.skill_state[prerequisite].learned
                for prerequisite in skill.prerequisites.parent_names
            )
        elif skill.prerequisites.dependence_model == "all":
            return all(
                self.skill_state[prerequisite].learned
                for prerequisite in skill.prerequisites.parent_names
            )
        else:
            raise ValueError(
                f"Invalid dependence model: {skill.prerequisites.dependence_model}"
            )

    def take_test(
        self, test: FixedFormAssessment, timestamp=0
    ) -> "BehaviorEventCollection":
        """Simulate taking a test with no formative feedback."""
        responses = []
        for item in test:
            response = self.respond_to_item(
                group=test, item=item, feedback=False, timestamp=timestamp
            )
            responses.append(response)
        test_results = BehaviorEventCollection(
            student_id=self.id,
            behavioral_events=responses,
        )
        if self.history:
            self.history.add_event(test_results)

        return test_results

    def respond_to_item(
        self, item: "Item", feedback=False, **kwargs
    ) -> "BehaviorEvent":
        """Simulate engaging with an item, and returning a response."""
        prob_correct = self.get_prob_correct(item)
        correct = 1 if random.random() < prob_correct else 0
        return ItemResponseEvent(
            student_id=self.id,
            item=item,
            score=correct,
            feedback_given=feedback,
            **kwargs,
        )

    def get_prob_correct(self, item: "Item") -> float:
        """Calculate probability of correct response based on skill state."""

        skill_level = self.skill_state[item.skill.name].skill_level
        skill_level_logit = logit(skill_level)

        return item.guess + (1 - item.slip - item.guess) * logistic(
            item.discrimination * (skill_level_logit - item.difficulty_logit)
        )


class BehaviorEventCollection(Model):
    """Group of behavior events, typically for an activity or assessment."""

    student_id: int
    timestamp: int = 0
    behavioral_events: List["BehaviorEvent"] = Field(
        default_factory=list,
        description="List of behavior events (can be BehaviorEvent or any subclass thereof)",
    )

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
    timestamp: int = 0
    # behavior: "BehaviorRepresentation" = None


class ItemResponseEvent(BehaviorEvent):
    item: Optional[Item] = None
    score: Optional[float] = None
    feedback_given: Optional[bool] = False
    practice_increment_logit: Optional[float] = 0.0

    def __str__(self):
        return (
            f"id={self.id}, type={str(self.item.__class__.__name__)}, "
            f"score={self.score}, timestamp={self.timestamp}"
        )

    @property
    def engagement_object(self) -> Optional[Item]:
        return self.item


class LearningEvent(BehaviorEvent):
    skill: Optional[Skill] = None


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


def save_student_activity_to_csv(
    students: List[Student],
    filename: str,
    include_engagements_without_ids: bool = False,
) -> None:
    """Save student activity (behavior events) to a CSV file."""

    path = prepare_directory(filename)
    with open(path, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write header
        header = [
            "student_id",
            "timestamp",
            "engagement_object_id",
            "score",
            "group_id",
        ]
        writer.writerow(header)

        # Write each student's behavior events
        for student in students:
            if not student.history:
                continue
            for event in student.history.get_events():
                all_events = []
                if isinstance(event, BehaviorEventCollection):
                    for behavior_event in event.behavioral_events:
                        if include_engagements_without_ids:
                            all_events.append(behavior_event)
                        else:
                            if behavior_event.engagement_object is not None:
                                all_events.append(behavior_event)
                elif isinstance(event, BehaviorEvent):
                    if include_engagements_without_ids:
                        all_events.append(event)
                    else:
                        if event.engagement_object is not None:
                            all_events.append(event)
                for behavior_event in all_events:
                    writer.writerow(
                        [
                            student.id,
                            behavior_event.timestamp,
                            behavior_event.engagement_object.id,
                            behavior_event.score
                            if hasattr(behavior_event, "score")
                            else None,
                            event.id if hasattr(event, "id") else None,
                        ]
                    )
