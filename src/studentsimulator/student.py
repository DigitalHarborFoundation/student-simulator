"""Student simulation module."""

import random
from typing import Annotated, Dict, Optional, Union

from pydantic import Field, validate_call

from studentsimulator.event import (
    BehaviorEvent,
    ItemResponseEvent,
    LearningEvent,
    WaitEvent,
)
from studentsimulator.general import Model, validate_int_list
from studentsimulator.item import Item
from studentsimulator.math import logistic, logit
from studentsimulator.skill import Skill, SkillSpace, SkillState, StudentSkills

# Constants for train/validation/test splits
TRAIN_SPLIT = 0
VAL_SPLIT = 1
TEST_SPLIT = 2

# Constants for observation status
OBSERVED = 1
UNOBSERVED = 0


class Student(Model):
    """A student is an entity that can learn skills, practice them, and take assessments.
    Students have a skill state that represents their current proficiency in each skill.
    They also have a history that records their interactions with the system.
    """

    name: str = "Student"  # Made optional with default value
    skill_space: SkillSpace = Field(default_factory=SkillSpace)
    skills: Optional[StudentSkills] = Field(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize skills with the skill_space
        if self.skills is None:
            self.skills = StudentSkills(skill_space=self.skill_space)

    def __str__(self):
        rep = f"Student(name={self.name}, id={self.id})"
        skills = self.skills.print_skill_states()
        rep += "\n" + skills
        return rep

    def print_event_history(self):
        self.skills.print_event_history()
        return self

    def print_daily_history(self):
        self.skills.print_daily_history()
        return self

    @property
    def skill_state(self) -> Dict[str, SkillState]:
        """Get current skill state. Maintained for backward compatibility."""
        return self.skills.end_of_day_skill_states.current_skill_states

    @property
    def days_since_initialization(self) -> int:
        """Get the number of days since initialization."""
        return max(self.skills.end_of_day_skill_states.daily_skill_states.keys())

    def set_skill_values(self, skill_values: Dict[str, float]) -> "Student":
        """Set initial skill levels for the student and mark them as learned."""
        for skill_name, level in skill_values.items():
            self.skills.set_skill_level(skill_name, level)
            self.skills[skill_name].learned = True
        return self

    @validate_int_list(field_name="practice_count")
    def randomly_initialize_skills(
        self,
        practice_count: Union[int, list[int]] = [1, 20],
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
        for skill in self.skills:
            # Encounter a learning event
            self.learn(skill)
            # Increment through the practice iterations
            # Skill level will only increase if the skill is successfully learned
            n_practice = random.randint(practice_count[0], practice_count[1])
            for i in range(n_practice):
                self.practice(skill)

        return self

    @validate_call()
    def practice(
        self,
        skill: Annotated[Skill, "The skill to practice"],
        item: Annotated[Optional[Item], "The item to practice on, if provided."] = None,
    ):
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
            activity_provider_name = response.activity_provider_name
        else:
            # No item provided, so no score or prob_correct
            score = None
            prob_correct = None
            activity_provider_name = None

        ## Record event
        self.skills.record_event(
            self,
            ItemResponseEvent(
                student_id=self.id,
                timestamp_in_days_since_initialization=self.days_since_initialization,
                skill=skill,
                item=item,
                score=score,
                prob_correct=prob_correct,
                feedback_given=True,
                activity_provider_name=activity_provider_name,
            ),
        )

        return self

    @validate_call()
    def wait(
        self,
        days: Annotated[float, Field(default=0, ge=0)],
        weeks: Annotated[float, Field(default=0, ge=0)],
        months: Annotated[float, Field(default=0, ge=0)],
    ):
        """Wait for a period of time, applying forgetting to all learned skills."""

        # Calculate total days to wait
        total_days = int(days + (weeks * 7) + (months * 30))

        self.skills.record_event(
            self,
            WaitEvent(
                student_id=self.id,
                timestamp_in_days_since_initialization=self.days_since_initialization,
                days_waited=total_days,
            ),
        )
        return self

    @validate_call()
    def learn(
        self,
        skill: Annotated[Skill, "The skill to learn"],
        activity_provider_name: Optional[str] = None,
        probability_of_learning_with_prerequisites: Optional[float] = None,
        probability_of_learning_without_prerequisites: Optional[float] = None,
    ):
        """Learn a skill.
        Learning happens during a 'learning encounter'.
        First we check to see if the student has the necessary prerequisites, if there are any.
        If they do, they learn with p=probability_of_learning_with_prerequisites.
        If they don't, they learn with p=probability_of_learning_without_prerequisites.
        If the skill is learned during this encounter, this sets the gate skill.learn=True,
        which enables practice to be productive and increase the skill level.
        """
        had_prerequisites = self.skills.has_prerequisites(skill)
        initial_learned = self.skills.is_learned(skill)
        # get random number
        random_number = random.random()

        # Override defaults if provided
        if probability_of_learning_with_prerequisites is None:
            probability_of_learning_with_prerequisites = (
                skill.probability_of_learning_with_prerequisites
            )
        if probability_of_learning_without_prerequisites is None:
            probability_of_learning_without_prerequisites = (
                skill.probability_of_learning_without_prerequisites
            )

        # Check to see if the skill has prerequisites
        if had_prerequisites:
            learned = random_number < probability_of_learning_with_prerequisites
        else:
            learned = random_number < probability_of_learning_without_prerequisites

        self.skills.record_event(
            self,
            LearningEvent(
                student_id=self.id,
                skill=skill,
                timestamp_in_days_since_initialization=self.days_since_initialization,
                had_prerequisites=had_prerequisites,
                initial_learned=initial_learned,
                final_learned=(initial_learned | learned),
                activity_provider_name=activity_provider_name,
            ),
        )

    @validate_call()
    def _get_prob_correct(
        self, item: Annotated[Item, "The item to respond to"]
    ) -> float:
        """Calculate probability of correct response based on skill state."""

        skill_level = self.skills[item.skill.name].skill_level
        skill_level_logit = logit(skill_level)

        return item.guess + (1 - item.slip - item.guess) * logistic(
            item.discrimination * (skill_level_logit - item.difficulty_logit)
        )

    @validate_call()
    def _respond_to_item(
        self,
        item: Annotated[Item, "The item to respond to"],
        feedback: Annotated[bool, "Whether to give feedback"] = False,
    ) -> BehaviorEvent:
        """Engage with an item, and return a response."""
        prob_correct = self._get_prob_correct(item)
        correct = 1 if random.random() < prob_correct else 0
        return ItemResponseEvent(
            student_id=self.id,
            item=item,
            skill=item.skill,
            score=correct,
            prob_correct=prob_correct,
            feedback_given=feedback,
            activity_provider_name=item.activity_provider_name,
        )


Student.model_rebuild()
