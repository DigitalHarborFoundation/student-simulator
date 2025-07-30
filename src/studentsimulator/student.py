"""Student simulation module."""

import csv
import os
import random
from typing import Annotated, Any, Dict, List, Optional, Tuple, Union

from pydantic import Field
from sklearn.metrics import roc_auc_score

from studentsimulator.general import Model, Skill, SkillSpace
from studentsimulator.item import Item
from studentsimulator.math import logistic, logit

# Constants for train/validation/test splits
TRAIN_SPLIT = 0
VAL_SPLIT = 1
TEST_SPLIT = 2

# Constants for observation status
OBSERVED = 1
UNOBSERVED = 0


class SkillState(Model):
    """Represents the skill state for a single student for a single skill.
    Skills are represented on a [0,1] scale.
    If a skill is 0, the student responds to skill-aligned items at chance level, i.e. with p=guess.
    If a skill is 1, the student responds to skill-aligned items at a high but not necessarily perfect level, i.e. with p=(1-slip).
    """

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


class StudentEventHistory(Model):
    """List of events associated with a student that would either:
    - affect their skill state (learning, practicing, waiting)
    - reveal their skill state (responding to an item, taking an assessment)

    Every entry in the events list is a subclass of BehaivorEvent or BehaviorEventCollection.
    """

    student_id: int
    events: List[Any] = Field(default_factory=list)

    def get_events(self, event_types: List[str] = None) -> List[Any]:
        """Get all events in the history, optionally filtered by event type.

        args:
            event_types: List[str] = None, optional. If provided, only return events of these types.
        """
        if event_types is None:
            return self.events
        else:
            return [
                event for event in self.events if type(event).__name__ in event_types
            ]

    def add_event(self, event: Any) -> None:
        """Add an event to the history."""
        self.events.append(event)

    def clear(self) -> None:
        """Clear all events from history."""
        self.events.clear()


class EndOfDaySkillStates(Model):
    """Stores end-of-day snapshots of a student's skill levels.

    The core data structure is an ordered dictionary where:
     - key: day number since we started recording events for that student
     - value: a dictionary where:
        - key: skill name
        - value: skill level at the end of the day.

    This is typically not used directly, but is updated automatically each
    time the student's StudentEventHistory is updated,
    and is used for post-hoc analysis.

    """

    daily_snapshots: Dict[int, Dict[str, float]] = {}  # day -> {skill_name: level}

    def get_skill_trajectories(
        self, skill_names: Union[str, List[str]] = None
    ) -> List[Tuple[int, float]]:
        """Get time series of (day, skill_level) for a specific skill.

        args:
            skill_name: str or list of str, optional. If provided, return the trajectory for the specified skill(s).
            If not provided, return the trajectory for all skills.

        returns:
            List of tuples (day, skill_level) for the specified skill(s).
        """
        # Make a list of skills we need to return
        all_skills = set()
        for levels in self.daily_snapshots.values():
            all_skills.update(levels.keys())
        if isinstance(skill_names, str):
            skill_set = [skill_names]
            if skill_names not in all_skills:
                raise ValueError(f"Skill {skill_names} not found in daily snapshots")
        elif isinstance(skill_names, list):
            skill_set = skill_names
            for skill in skill_set:
                if skill not in all_skills:
                    raise ValueError(f"Skill {skill} not found in daily snapshots")
        elif skill_names is None:
            skill_set = list(all_skills)
        else:
            raise ValueError(f"Invalid skill name: {skill_names}")

        # Get the trajectory for each skill
        trajectories = {}
        for skill in skill_set:
            trajectories[skill] = [
                (day, levels.get(skill, 0.0))
                for day, levels in sorted(self.daily_snapshots.items())
            ]
        return trajectories


class Student(Model):
    """A student is an entity that can learn skills, practice them, and take assessments.
    Students have a skill state that represents their current proficiency in each skill.
    They also have a history that records their interactions with the system.
    """

    name: str = "Student"  # Made optional with default value
    skill_space: SkillSpace
    event_history: Field(default_factory=StudentEventHistory)  # Renamed from history
    end_of_day_skill_states: Field(
        default_factory=EndOfDaySkillStates
    )  # Renamed from daily_history
    days_since_initialization: int = 0

    def __str__(self):
        rep = f"Student(name={self.name}, id={self.id})"
        if self.skill_state is not None:
            for skill in self.skill_state:
                skill_state = self.skill_state[skill]
                rep += "\n" + f"    {skill}: {skill_state.skill_level:.2f}"
        return rep

    def print_history(self):
        """Nicely print the student's history."""
        if not self.event_history or not self.event_history.get_events():
            print("No history available.")
        else:
            print("\n".join(str(event) for event in self.event_history.get_events()))

    @property
    def skill_state(self) -> Dict[str, SkillState]:
        """Get current skill state. Maintained for backward compatibility."""
        return self.end_of_day_skill_states.current_skill_state

    ##### Methods for handling student actions #####

    @staticmethod
    def _apply_practice(level: float, skill: Skill) -> float:
        """Return new level after one practice increment (if learned)."""
        new_logit = logit(level) + skill.practice_increment_logit
        return logistic(new_logit)

    @staticmethod
    def _apply_forgetting(level: float, skill: Skill, days: int) -> float:
        """Return new level after forgetting for *days* days."""
        if days <= 0:
            return level
        new_logit = logit(level) - skill.decay_logit * days
        min_logit = logit(0.01)
        return logistic(max(min_logit, new_logit))

    def practice(self, skill: Skill, item: Optional[Item] = None):
        """Practice a skill with recursive transfer to all ancestor skills.

        This method simulates a student practicing a specific skill, with the option
        to use a particular item. When practice occurs on a learned skill, it increases
        the skill level and also provides transfer effects to all ancestor skills
        (parents, grandparents, etc.) with diminishing returns.

        The transfer effect is based on research showing that learning in one domain
        can benefit related domains, particularly when there are hierarchical
        relationships. The diminishing returns (exponential decay with depth) reflects
        that transfer effects are strongest for immediate prerequisites and weaker
        for more distant ancestors.

        References:
        - Barnett, S. M., & Ceci, S. J. (2002). When and where do we apply what we learn?
        - Singley, M. K., & Anderson, J. R. (1989). The transfer of cognitive skill.

        Args:
            skill: The skill being practiced
            item: Optional item used for practice (affects response accuracy)
        """

        ## Generate behavior
        if item is not None:
            response = self._respond_to_item(item)
            score = response.score
            prob_correct = response.prob_correct
        else:
            # No item provided, so no score or prob_correct
            score = None
            prob_correct = None

        ## Update skill states
        if self.skill_state[skill.name].learned:
            self.skill_state[skill.name].skill_level = self._apply_practice(
                self.skill_state[skill.name].skill_level, skill
            )
            self._update_ancestor_skills(skill, skill.practice_increment_logit, depth=1)

        ## Record event

        # get response to item if provided

        if self.event_history:
            self.event_history.add_event(
                ItemResponseEvent(
                    student_id=self.id,
                    item=item,
                    score=score,
                    prob_correct=prob_correct,
                    feedback_given=True,
                    practice_increment_logit=skill.practice_increment_logit,
                )
            )

    def wait(
        self,
        seconds: int = 0,
        minutes: int = 0,
        hours: int = 0,
        days: int = 0,
        weeks: int = 0,
        months: int = 0,
    ):
        """Wait for a period of time, applying forgetting to all learned skills."""
        # check that all are non-negative
        if (
            seconds < 0
            or minutes < 0
            or hours < 0
            or days < 0
            or weeks < 0
            or months < 0
        ):
            raise ValueError("All time units must be non-negative")

        # check that all are float or convertable to float
        if not all(
            isinstance(x, (int, float))
            for x in [seconds, minutes, hours, days, weeks, months]
        ):
            raise ValueError("All time units must be integers or floats")

        # Calculate total days to wait
        total_days = int(
            days
            + (weeks * 7)
            + (months * 30)
            + (hours / 24)
            + (minutes / (24 * 60))
            + (seconds / (24 * 60 * 60))
        )

        # ------------------------------------------------------------------
        # Record a WaitEvent *before* applying forgetting so that the event
        # timestamp represents the moment immediately preceding the wait.
        # ------------------------------------------------------------------
        if total_days > 0 and self.event_history is not None:
            self.event_history.add_event(
                WaitEvent(
                    student_id=self.id,
                    timestamp_in_days_since_initialization=self.days_since_initialization,
                    days_waited=total_days,
                )
            )

        # Apply forgetting day by day and record daily snapshots
        if total_days > 0:
            for day_offset in range(1, total_days + 1):
                # Apply one day of forgetting to all learned skills
                for skill in self.skill_space.skills:
                    self.forget(skill, 1)  # Apply 1 day of forgetting

                # Update day counter
                self.days_since_initialization += 1

                # Record daily snapshot after forgetting
                self._record_daily_snapshot()
        else:
            # No time passed, but still update the day counter if needed
            self.days_since_initialization += total_days

        return self

    def forget(self, skill: Skill, time_days: int, rate: Optional[float] = None):
        """Apply forgetting to a skill based on time elapsed.

        Uses exponential decay on the logit scale, following Ebbinghaus forgetting curve.
        Only affects learned skills - unlearned skills don't decay.

        Args:
            skill: The skill to apply forgetting to
            time_days: Number of days since last practice/learning
            rate: Forgetting rate in logit units per day. If None, uses skill.decay_logit
        """
        if not self.skill_state[skill.name].learned:
            return  # Unlearned skills don't decay

        if rate is None:
            rate = skill.decay_logit

        if time_days <= 0 or rate <= 0:
            return  # No time passed or no decay rate

        self.skill_state[skill.name].skill_level = self._apply_forgetting(
            self.skill_state[skill.name].skill_level, skill, time_days
        )

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
            if (
                self.skill_state[skill.name].skill_level
                < skill.initial_skill_level_after_learning
            ):
                self.skill_state[
                    skill.name
                ].skill_level = skill.initial_skill_level_after_learning

        if record_event_in_history:
            self.event_history.add_event(
                LearningEvent(
                    student_id=self.id,
                    skill=skill,
                    timestamp_in_days_since_initialization=self.days_since_initialization,
                )
            )

    def _respond_to_item(
        self, item: "Item", feedback=False, **kwargs
    ) -> "BehaviorEvent":
        """Engage with an item, and return a response."""
        prob_correct = self.get_prob_correct(item)
        correct = 1 if random.random() < prob_correct else 0
        return ItemResponseEvent(
            student_id=self.id,
            item=item,
            score=correct,
            prob_correct=prob_correct,
            feedback_given=feedback,
            **kwargs,
        )

    ##### Methods for initializing student skill state #####

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

    ##### Methods for updating student skill state after actions #####

    def _record_event(self, event: Any):
        """Add an event to the history and update daily snapshot if appropriate."""

        # Add event to history
        self.event_history.add_event(event)
        # Update end-of-day skill states
        self._update_end_of_day_skill_states(event)

    def _update_end_of_day_skill_states(self, event: Any):
        """Update the end-of-day skill states based on the event."""

        # Update daily snapshot if appropriate
        # TODO

    def _update_ancestor_skills(self, skill: Skill, base_increment: float, depth: int):
        transfer_factor = 0.3  # 30% of the practice effect transfers to each level
        if skill.prerequisites and skill.prerequisites.parent_names:
            for parent_name in skill.prerequisites.parent_names:
                if (
                    parent_name in self.skill_state
                    and self.skill_state[parent_name].learned
                ):
                    parent_skill = self.skill_space.get_skill(parent_name)
                    transfer_increment = base_increment * (transfer_factor**depth)
                    # before = self.skill_state[parent_name].skill_level
                    self.skill_state[parent_name].skill_level = logistic(
                        logit(self.skill_state[parent_name].skill_level)
                        + transfer_increment
                    )
                    # after = self.skill_state[parent_name].skill_level

                    self._update_ancestor_skills(
                        parent_skill, base_increment, depth + 1
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

    def _get_prob_correct(self, item: "Item") -> float:
        """Calculate probability of correct response based on skill state."""

        skill_level = self.skill_state[item.skill.name].skill_level
        skill_level_logit = logit(skill_level)

        return item.guess + (1 - item.slip - item.guess) * logistic(
            item.discrimination * (skill_level_logit - item.difficulty_logit)
        )


class BehaviorEventCollection(Model):
    """Group of behavior events, typically for an activity or assessment."""

    student_id: int
    timestamp_in_days_since_initialization: int = 0
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
    timestamp_in_days_since_initialization: int = 0
    # behavior: "BehaviorRepresentation" = None


class ItemResponseEvent(BehaviorEvent):
    item: Optional[Item] = None
    score: Optional[float] = None
    prob_correct: Optional[float] = None
    feedback_given: Optional[bool] = False
    practice_increment_logit: Optional[float] = 0.0

    def __str__(self):
        return f"""ItemResponseEvent(
    timestamp={self.timestamp_in_days_since_initialization}
    id={self.id},
    type={str(self.item.__class__.__name__)},
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
    )"""


Student.model_rebuild()


def calculate_auc(y_true: List[float], y_pred: List[float]) -> Optional[float]:
    """Calculate AUC score for binary classification.

    Args:
        y_true: List of actual outcomes (0 or 1)
        y_pred: List of predicted probabilities

    Returns:
        AUC score or None if calculation fails
    """
    try:
        return roc_auc_score(y_true, y_pred)
    except (ValueError, TypeError):
        return None


def prepare_directory(filename: str) -> str:
    """Ensure the directory for the given filename exists and return the full path."""
    # Ensure .csv extension
    if not filename.endswith(".csv"):
        filename += ".csv"

    # Get the directory path
    directory = os.path.dirname(filename)

    # If no directory specified, use current directory
    if directory == "":
        return filename

    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Return the full path
    return filename


def save_student_profile_to_csv(students: List[Student], filename: str) -> None:
    """Save student skill states to a CSV file."""

    path = prepare_directory(filename)
    with open(path, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write header
        header = [
            "student_id",
            "student_name",
            "skill_id",
            "skill_name",
            "skill_level",
            "skill_level_logit",
        ]
        writer.writerow(header)

        # Write each student's skill state
        for student in students:
            for skill_name, skill_state in student.skill_state.items():
                # Get the skill ID from the skill space
                skill_id = None
                if student.skill_space:
                    try:
                        skill = student.skill_space.get_skill(skill_name)
                        skill_id = skill.id if skill else None
                    except ValueError:
                        # Skill not found in skill space
                        skill_id = None

                writer.writerow(
                    [
                        student.id,
                        student.name,
                        skill_id,
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
    train_val_test_split: Optional[Tuple[float, float, float]] = None,
    observation_rate: float = 1.0,
) -> None:
    """Save student activity (behavior events) to a CSV file."""

    # Validate observation rate
    if not 0.0 <= observation_rate <= 1.0:
        raise ValueError("observation_rate must be between 0.0 and 1.0")

    # Create train/validation/test split if requested
    student_splits = {}
    if train_val_test_split is not None:
        train_pct, val_pct, test_pct = train_val_test_split
        if abs(train_pct + val_pct + test_pct - 1.0) > 1e-6:
            raise ValueError("train_val_test_split percentages must sum to 1.0")

        # Shuffle students for random split
        student_ids = [student.id for student in students]
        random.shuffle(student_ids)

        # Calculate split points
        n_students = len(student_ids)
        train_end = int(n_students * train_pct)
        val_end = train_end + int(n_students * val_pct)

        # Assign splits
        for i, student_id in enumerate(student_ids):
            if i < train_end:
                student_splits[student_id] = 0  # train
            elif i < val_end:
                student_splits[student_id] = 1  # validation
            else:
                student_splits[student_id] = 2  # test

    # Collect data for AUC calculation
    y_true = []
    y_pred = []

    path = prepare_directory(filename)
    with open(path, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write header
        header = [
            "studentid",
            "timeid",
            "itemid",
            "skillid",
            "response",
            "prob_correct",
            "groupid",
        ]
        if train_val_test_split is not None:
            header.append("train_val_test")
        if observation_rate < 1.0:
            header.append("observed")
        writer.writerow(header)

        # Write each student's behavior events
        for student in students:
            if not student.event_history:
                continue
            for event in student.event_history.get_events():
                all_events = []
                if isinstance(event, BehaviorEventCollection):
                    for behavior_event in event.behavioral_events:
                        if include_engagements_without_ids:
                            # add no matter what
                            all_events.append(behavior_event)
                        else:
                            # add only if engagement_object is not None
                            if (
                                isinstance(behavior_event, ItemResponseEvent)
                                and behavior_event.engagement_object is not None
                            ):
                                all_events.append(behavior_event)
                elif isinstance(event, BehaviorEvent):
                    if include_engagements_without_ids:
                        all_events.append(event)
                    else:
                        if (
                            isinstance(event, ItemResponseEvent)
                            and event.engagement_object is not None
                        ):
                            all_events.append(event)
                for behavior_event in all_events:
                    # Get engagement_object_id safely
                    engagement_id = None
                    skill_id = None
                    if (
                        isinstance(behavior_event, ItemResponseEvent)
                        and behavior_event.engagement_object is not None
                    ):
                        engagement_id = behavior_event.engagement_object.id
                        # Get skill ID from the item
                        if hasattr(behavior_event.engagement_object, "skill"):
                            skill_id = behavior_event.engagement_object.skill.id

                    # Collect data for AUC calculation
                    if (
                        hasattr(behavior_event, "score")
                        and hasattr(behavior_event, "prob_correct")
                        and behavior_event.score is not None
                        and behavior_event.prob_correct is not None
                    ):
                        y_true.append(float(behavior_event.score))
                        y_pred.append(float(behavior_event.prob_correct))

                    row = [
                        student.id,
                        behavior_event.timestamp_in_days_since_initialization,
                        engagement_id,
                        skill_id,
                        behavior_event.score
                        if hasattr(behavior_event, "score")
                        else None,
                        round(behavior_event.prob_correct, 4)
                        if hasattr(behavior_event, "prob_correct")
                        and behavior_event.prob_correct is not None
                        else None,
                        event.id if hasattr(event, "id") else None,
                    ]
                    if train_val_test_split is not None:
                        row.append(student_splits.get(student.id, 0))
                    if observation_rate < 1.0:
                        # Randomly determine if this event is observed
                        is_observed = random.random() < observation_rate
                        row.append(1 if is_observed else 0)
                    writer.writerow(row)

    # Calculate and report AUC
    print(f"Debug: Collected {len(y_true)} responses for AUC calculation")
    if y_true and y_pred:
        auc_score = calculate_auc(y_true, y_pred)
        if auc_score is not None:
            print(f"AUC Score: {auc_score:.4f} (based on {len(y_true)} responses)")
        else:
            print("Could not calculate AUC score")
    else:
        print("No valid data for AUC calculation")
