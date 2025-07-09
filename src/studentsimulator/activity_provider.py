"""This class maintains objects that are created by an Activity Provider
including an Item, an ItemPool, and a FixedFormAssessment."""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import Field, model_validator

from studentsimulator.general import Model, Skill, SkillSpace


class ActivityProvider(Model):
    """The ActivityProvider creates learning experiences for students.
    The generator takes the perspective of a "provider" of learning or assessment experiences.
    This encapsulates logic for things like:
    - a lesson, video, or other instructional content ('gating event')
    - a tutoring session (interleaved instruction and practice)
    - formative practice with feedback (practice leading to learning)
    - summative assessment without feedback (practice not leading to learning)

    Notably, the ActivityProvider is *stateless*. It does not keep a
    memory of past interactions with students. Rather, each interaction accepts
    a student's history and generates a new event based on that history.

    Pragmatically, this design decision allows the 'single source of truth' for
    student history to be the student object itself.

    A single ActivityProvider may or may not have visibility into the full
    history of a student's learning.
    """

    skill_space: SkillSpace = SkillSpace(
        skills=[]
    )  # skills that this provider can generate items for
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

    def get_skill(self, skill_id: Union[str, Skill]) -> Optional[Skill]:
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
            return skill_list
        else:
            raise ValueError(
                "Skills must be a SkillSpace or a list of Skill objects or strings."
            )

    def generate_fixed_form_assessment(
        self,
        item_pool: Union[str, "ItemPool"],
        n_items: int,
        skills: Union[list[Skill], list[str], SkillSpace] = [],
    ) -> "FixedFormAssessment":
        """Generate a random assessment with a specified number of items.

        Args:
            item_pool: The name of the item pool to use, or the item pool object itself.
            n_items: The number of items to include in the assessment.
            skills: The skills to include in the assessment. If None, all skills in the skill space will be used.
        """

        # Create a list of skill names that we can use for filtering
        # if skills is an empty list, use all skills in the skill space
        skills_to_use = (
            self.validate_skill_list(skills) if skills else self.skill_space.skills
        )
        # Filter items in the item pool based on the skills to use

        if isinstance(item_pool, str):
            item_pool = self.item_pools[item_pool]
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

    difficulty_logit: float = Field(
        default=0.0, ge=-4.0, le=4.0, description="Difficulty parameter (logit scale)"
    )
    difficulty_logit_range: Optional[Tuple[float, float]] = Field(
        default=None, description="Range for difficulty parameter [min, max]"
    )

    guess: float = Field(
        default=0.2,
        ge=0.0,
        le=0.5,
        description="Guess parameter (probability of correct response by guessing)",
    )
    guess_range: Optional[Tuple[float, float]] = Field(
        default=None, description="Range for guess parameter [min, max]"
    )

    slip: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Slip parameter (probability of incorrect response when should be correct)",
    )
    slip_range: Optional[Tuple[float, float]] = Field(
        default=None, description="Range for slip parameter [min, max]"
    )

    discrimination: float = Field(
        default=1.0,
        ge=0.1,
        le=3.0,
        description="Discrimination parameter (slope of item characteristic curve)",
    )
    discrimination_range: Optional[Tuple[float, float]] = Field(
        default=None, description="Range for discrimination parameter [min, max]"
    )

    practice_effectiveness_logit: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Effectiveness of this item for skill practice",
    )
    practice_effectiveness_logit_range: Optional[Tuple[float, float]] = Field(
        default=None, description="Range for practice effectiveness [min, max]"
    )

    options: List["ItemOption"] = []
    bug_map: Dict[str, int] = {}  # misconception -> distractor mapping

    text: Optional[str] = None  # text of the item (question or task)

    @model_validator(mode="after")
    def validate_parameters(self) -> "Item":
        """Validate that only one of each parameter/range pair is specified."""

        # Validate ranges if provided
        if self.difficulty_logit_range is not None:
            if not (
                -4.0
                <= self.difficulty_logit_range[0]
                <= self.difficulty_logit_range[1]
                <= 4.0
            ):
                raise ValueError(
                    "difficulty_logit_range must be [min, max] where -4.0 <= min <= max <= 4.0"
                )

        if self.guess_range is not None:
            if not (0.0 <= self.guess_range[0] <= self.guess_range[1] <= 0.5):
                raise ValueError(
                    "guess_range must be [min, max] where 0.0 <= min <= max <= 0.5"
                )

        if self.slip_range is not None:
            if not (0.0 <= self.slip_range[0] <= self.slip_range[1] <= 0.5):
                raise ValueError(
                    "slip_range must be [min, max] where 0.0 <= min <= max <= 0.5"
                )

        if self.discrimination_range is not None:
            if not (
                0.1
                <= self.discrimination_range[0]
                <= self.discrimination_range[1]
                <= 3.0
            ):
                raise ValueError(
                    "discrimination_range must be [min, max] where 0.1 <= min <= max <= 3.0"
                )

        if self.practice_effectiveness_logit_range is not None:
            if not (
                0.0
                <= self.practice_effectiveness_logit_range[0]
                <= self.practice_effectiveness_logit_range[1]
                <= 1.0
            ):
                raise ValueError(
                    "practice_effectiveness_logit_range must be [min, max] where 0.0 <= min <= max <= 1.0"
                )

        return self

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

        # Generate random values from ranges if specified
        if self.difficulty_logit_range is not None:
            self.difficulty_logit = round(
                random.uniform(
                    self.difficulty_logit_range[0], self.difficulty_logit_range[1]
                ),
                3,
            )

        if self.guess_range is not None:
            self.guess = round(
                random.uniform(self.guess_range[0], self.guess_range[1]), 3
            )

        if self.slip_range is not None:
            self.slip = round(random.uniform(self.slip_range[0], self.slip_range[1]), 3)
        elif self.slip is None:
            self.slip = 0.1  # Default value

        if self.discrimination_range is not None:
            self.discrimination = round(
                random.uniform(
                    self.discrimination_range[0], self.discrimination_range[1]
                ),
                3,
            )

        if self.practice_effectiveness_logit_range is not None:
            self.practice_effectiveness_logit = round(
                random.uniform(
                    self.practice_effectiveness_logit_range[0],
                    self.practice_effectiveness_logit_range[1],
                ),
                3,
            )

        return self

    def __str__(self):
        return (
            f"Item id={self.id}\n"
            f"alignment={self.skill.name}\n"
            f"options={[option.label + ('*' if option.is_target else '') for option in self.options]})\n"
            f"difficulty={self.difficulty_logit}\n"
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
        # If self.items is None, return an empty iterator; else, iterate over items
        return iter(self.items or [])
