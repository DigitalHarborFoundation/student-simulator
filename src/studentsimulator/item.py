from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Dict, List, Optional, Tuple

from pydantic import Field, model_validator

from studentsimulator.general import Model
from studentsimulator.skill import Skill


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


class ItemPool(Model, Sequence):
    """A collection of items that can be used in assessments.
    This is typically used to generate assessments with a fixed number of items
    from a pool of available items.

    The ItemPool implements the sequence protocol, so you can:
    - Get length: len(pool)
    - Iterate: for item in pool
    - Index: pool[i]
    - Slice: pool[i:j]
    """

    name: str
    items: List["Item"] = []  # list of items in the pool

    def __str__(self) -> str:
        return f"Item Pool Name: {self.name}" + "\n".join(str(i) for i in self.items)

    def __iter__(self):
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index):
        """Support indexed access (pool[i]) and slicing (pool[i:j]).

        Args:
            index: An integer index or slice object.

        Returns:
            Item: A single item if indexed, or a list of items if sliced.

        Raises:
            IndexError: If index is out of range.
            TypeError: If index is not an integer or slice.
        """
        return self.items[index]


class Item(Model):
    """An item is a question or task that a student can attempt. It is associated with a skill
    and has parameters that affect how the student responds to it.
    This is used in both formative (practice) and summative (test) assessments.

    The item parameters are assuming an IRT-like model with an additional 'slip' parameter.

    p(correct | student, item) = guess + (1 - guess - slip) * sigmoid(discrimination * (skill_level - difficulty))

    All of these are properties of the item itself except for skill_level, which is a property of the student.
    """

    skill: Skill
    activity_provider_name: Optional[str] = None
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
