"""This class maintains objects that are created by an Activity Provider
including an Item, an ItemPool, and a FixedFormAssessment."""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import model_validator

from studentsimulator.general import Model, Skill, SkillSpace


class ActivityProvider(Model):
    """The ActivityProvider creates learning experiences for students.
    The generator takes the perspective of a "provider" of learning or assessment experiences.
    This encapsulates logic for things like:
    - a lesson, video, or other instructional content ('gating event')
    - a tutoring session (interleaved instruction and practice)
    - formative practice with feedback (practice leading to learning)
    - summative assessment without feedback (practice not learning to learning)

    Notably, the ActivityProvider is *stateless*. It does not keep a
    memory of past interactions with students. Rather, each interaction accepts
    a student's history and generates a new event based on that history.

    Pragmatically, this design decision allows the 'single source of truth' for
    student history to be the student object itself.

    A single ActivityProvider may or may not have visibility into the full
    history of a student's learning.
    """

    skill_space: SkillSpace = None  # skills that this provider can generate items for
    item_pools: Dict[str, "ItemPool"] = {}

    def construct_item_pool(
        self, name: str, skills: List[Skill], n_items_per_skill: int, **kwargs: Any
    ) -> "ItemPool":
        """Construct an item pool with a specified number of items for the given skills.

        Additional kwargs are passed to the Item constructor.
        """
        item_pool = []
        skills = self.validate_skill_list(skills)
        for skill in skills:
            for _ in range(n_items_per_skill):
                item = Item(skill=skill, **kwargs)
                item_pool.append(item)
        item_pool = ItemPool(name=name, items=item_pool)
        self.item_pools[name] = item_pool
        return item_pool

    def register_skills(self, skill_space: SkillSpace) -> None:
        """Register a list of skills with the provider."""
        self.skill_space = skill_space

    def get_skill(self, skill_id: str | Skill) -> Optional[Skill]:
        """Get a skill by its ID."""
        for skill in self.skill_space.skills:
            if skill.name == skill:
                return skill

        return None

    def validate_skill_list(
        self, skills: Union[List[Skill], List[str], SkillSpace]
    ) -> list[Skill]:
        """Validate and convert a list of skills to Skill objects."""
        # if it's a SkillSpace, return the skills directly
        print("HELP1", skills)
        if isinstance(skills, SkillSpace):
            return skills.skills
        elif isinstance(skills, list):
            skill_list = []
            for skill in skills:
                if isinstance(skill, Skill):
                    skill_list.append(skill)
                elif isinstance(skill, str):
                    skill_obj = self.skill_space.get_skill(skill)
                    if skill_obj is None:
                        raise ValueError(f"Skill not found: {skill}")
                    skill_list.append(skill_obj)
                elif isinstance(skill, dict):
                    # If it's a dict, retrieve it by name
                    skill_name = skill.get("name")
                    if skill_name is None:
                        raise ValueError("Skill dict must have a 'name' key.")
                    skill_obj = self.skill_space.get_skill(skill_name)
                    if skill_obj is None:
                        raise ValueError(f"Skill not found: {skill_name}")
                    skill_list.append(skill_obj)
                else:
                    raise ValueError(
                        "Skills must be a list of Skill objects or strings."
                    )
            print("RETURNING", len(skill_list))
            return skill_list
        else:
            raise ValueError(
                "Skills must be a SkillSpace or a list of Skill objects or strings."
            )

    def generate_fixed_form_assessment(
        self,
        item_pool: Union[str, "ItemPool"],
        n_items: int,
        skills: Union[list[Skill], list[str], SkillSpace] = None,
    ) -> "FixedFormAssessment":
        """Generate a random assessment with a specified number of items."""
        # Create a list of skill names that we can use for filtering
        print("HELP0", skills)
        skills_to_use = self.validate_skill_list(skills) if skills else []
        print("HELP3", len(skills_to_use))
        # Filter items in the item pool based on the skills to use

        if isinstance(item_pool, str):
            item_pool = self.item_pools.get(item_pool)
        valid_items = []
        for item in item_pool:
            if item.skill.name in [s.name for s in skills_to_use]:
                valid_items.append(item)

        # Randomly select n items from the filtered item pool, without replacement
        if len(valid_items) < n_items:
            raise ValueError(
                f"Not enough items in the pool for the requested number of items. "
                f"Requested {n_items}, valid items: {len(valid_items)}."
            )
        selected_items = random.sample(valid_items, n_items)

        return FixedFormAssessment(items=selected_items)


class ItemOption(Model):
    """An option for an item, typically used in multiple-choice questions."""

    label: str  # e.g., "A", "B", "C", "D"
    text: Optional[str]  # the text of the option
    is_target: bool = False  # whether this option is the correct answer
    misconception: Optional[
        str
    ] = None  # optional misconception this option is sensitive to

    @model_validator(mode="after")
    def check_target(self) -> "ItemOption":
        """Ensure is_target is False if misconception is not None."""
        if self.is_target and self.misconception is not None:
            raise ValueError("Target options cannot have a misconception.")
        return self


class ItemPool(Model):
    """A collection of items that can be used in assessments.
    This is typically used to generate assessments with a fixed number of items
    from a pool of available items.
    """

    name: str
    items: List["Item"] = []  # list of items in the pool

    def __str__(self) -> str:
        return f"Item Pool Name: {self.name}" + "\n".join(str(i) for i in self.items)

    def __iter__(self):
        return iter(self.items)


class Item(Model):
    """An item is a question or task that a student can attempt. It is associated with a skill
    and has parameters that affect how the student responds to it.
    This is used in both formative (practice) and summative (test) assessments.

    The item parameters are assuming an IRT-like model with an additional 'slip' parameter.

    p(correct | student, item) = guess + (1 - guess - slip) * sigmoid(discrimination * (skill_level - difficulty))

    All of these are properties of the item itself except for skill_level, which is a property of the student.
    """

    skill: Skill
    difficulty: float | Tuple[float, float] = 0.0
    guess: float | Tuple[float, float] = 0.2
    slip: float | Tuple[float, float] = 0.1
    discrimination: float | Tuple[float, float] = 1.0
    practice_effectiveness: float = 0.0
    options: List["ItemOption"] = []
    bug_map: Dict[str, int] = {}  # misconception -> distractor mapping

    text: Optional[str] = None  # text of the item (question or task)

    @model_validator(mode="after")
    def generate_options(self) -> "Item":
        """Generate options if not provided."""
        if not self.options:
            target_index = random.randint(0, 3)
            self.options = [
                ItemOption(
                    label=chr(65 + i),
                    text=f"Option {i + 1}",
                    is_target=(i == target_index),
                )
                for i in range(4)  # Default to 4 options A, B, C, D
            ]
        if isinstance(self.guess, tuple):
            self.guess = round(random.uniform(self.guess[0], self.guess[1]), 3)
        if isinstance(self.slip, tuple):
            self.slip = round(random.uniform(self.slip[0], self.slip[1]), 3)
        if isinstance(self.discrimination, tuple):
            self.discrimination = round(
                random.uniform(self.discrimination[0], self.discrimination[1]), 3
            )
        if isinstance(self.difficulty, tuple):
            self.difficulty = round(
                random.uniform(self.difficulty[0], self.difficulty[1]), 3
            )

        return self

    def __str__(self):
        return (
            f"Item id={self.id}\n"
            f"alignment={self.skill.name}\n"
            f"options={[option.label + ('*' if option.is_target else '') for option in self.options]})\n"
            f"difficulty={self.difficulty}\n"
            f"guess={self.guess}\n"
            f"slip={self.slip}\n"
            f"discrimination={self.discrimination}\n"
        )


class FixedFormAssessment(Model):
    """A fixed-form assessment is a set of items that are presented to the student
    in a fixed order. This is typically used for summative assessments where the
    items are not adaptive and the student does not receive feedback on their responses.
    """

    items: Optional[List[Item]]  # list of item IDs in the assessment

    def __iter__(self):
        return iter(self.items)

    # def generate_activity(self, student_history: Any) -> Any:
    #     """The generator looks at the student's history and returns
    #     (possibly adaptively) a new activity.

    #     The student interacts with the activity, and the behavior
    #     is recorded by the student object and tagged with the source.

    #     Possible return types include:
    #     - `Instruction`: a lesson, video, or other instructional content
    #     - `PracticeInstance`: a practice item with feedback
    #     - `TestQuestionInstance`: a summative assessment item without feedback

    #     These can be composed into more complex activities. For example, a tutoring
    #     session could have interleaved Instruction and Practice items, or a lesson
    #     may have instruction followed by practice, and finally a formative assessment.

    #     Args:
    #         student_history: The student's history of interactions, skills, and events.
    #     """


# class Practice(Event):
#     pass

# class Intervention(Event):
#     """Represents an intervention event where the student is given some
#     additional support or instruction to help them learn a skill."""

#     intervention_id: str  # intervention is linked to skill


# class Feedback(BaseModel):
#     """Evaluation of the student's behavior"""

#     feedback_type: str  # "binary", "score", "rubric", "qualitative"
#     correct: Optional[bool] = None  # for binary feedback
#     score: Optional[float] = None  # for numeric scores
#     max_score: Optional[float] = None  # scale information
#     feedback_text: Optional[str] = None  # qualitative feedback
#     rubric_scores: Optional[Dict[str, float]] = None  # detailed rubric


# class InterventionEvent(BaseModel):
#     """Something that happens TO the student"""

#     student_id: str
#     timestamp: int
#     intervention_type: str  # "lesson", "video", "hint", "tutoring", "feedback"
#     target_skill: Optional[str] = None
#     target_misconception: Optional[str] = None
#     intervention_data: Dict[str, Any] = {}  # intervention-specific parameters
#     learning_probability: float = 0.8  # probability student learns the skill
#     baseline_proficiency: float = 0.6  # proficiency level if learning occurs


# class BehavioralEvent(BaseModel):
#     """Something the student DOES, with potential feedback"""

#     student_id: str
#     timestamp: int
#     behavior: BehaviorRepresentation
#     feedback: Optional[Feedback] = None
#     context: Dict[str, Any] = {}  # additional metadata
