from __future__ import annotations

from typing import ClassVar, Literal, Optional

import networkx as nx
from pydantic import BaseModel, Field, model_validator


class Model(BaseModel):
    # one counter per subclass, created automatically
    _counter: ClassVar[int] = 0  # ensure base class can be instantiated

    id: int = -1  # not required—filled in later

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._counter = 0  # fresh counter for this subclass
        # PEP 487 guarantees this runs once per subclass
        # [oai_citation:1‡peps.python.org](https://peps.python.org/pep-0487/?utm_source=chatgpt.com)

    def __init__(self, **data):
        super().__init__(**data)  # normal Pydantic validation
        if self.id < 0:  # assign only if caller didn’t supply one
            cls = self.__class__
            self.id = cls._counter
            cls._counter += 1


class PrerequisiteStructure(Model):
    parent_names: list[str] = []
    dependence_model: Literal["any", "all"] = "all"


class SkillSpace(Model):
    """A SkillSpace is the collection of all skills that a student can learn.
    The validation methods ensure that skills are unique, prerequisites are valid, and
    the skills are topologically sorted based on their prerequisites."""

    skills: list[Skill] = []

    @model_validator(mode="after")
    def guarantee_unique(self):
        # Check for duplicate skill names
        skill_names = [s.name for s in self.skills]
        if len(skill_names) != len(set(skill_names)):
            # Find the duplicate names
            seen = set()
            duplicates = set()
            for name in skill_names:
                if name in seen:
                    duplicates.add(name)
                seen.add(name)
            raise ValueError(
                f"Skill names must be unique. Found duplicates: {duplicates}"
            )

        # Check for duplicate skill codes (if codes are provided)
        skill_codes = [s.code for s in self.skills if s.code not in (None, "", "None")]
        if len(skill_codes) != len(set(skill_codes)):
            # Find the duplicate codes
            seen = set()
            duplicates = set()
            for code in skill_codes:
                if code in seen:
                    duplicates.add(code)
                seen.add(code)
            raise ValueError(
                f"Skill codes must be unique. Found duplicates: {duplicates}"
            )

        return self

    @model_validator(mode="after")
    def topological_sort(self):
        # Reorder the skills using topological sort
        # Do a topological sort on the skills
        # Make string representation with all nodes for error printing in assertion
        G = nx.DiGraph()
        for skill in self.skills:
            # skill.prerequisites may be None, so check before accessing parent_names
            if skill.prerequisites is not None and hasattr(
                skill.prerequisites, "parent_names"
            ):
                for parent_name in skill.prerequisites.parent_names:
                    G.add_edge(parent_name, skill.name)

        # Add all skills as nodes to ensure they're included
        for skill in self.skills:
            G.add_node(skill.name)

        graph_string = f"Nodes: {list(G.nodes)}, Edges: {list(G.edges)}"
        assert nx.is_directed_acyclic_graph(
            G
        ), f"The set of metric dependencies must be acyclic! You have cyclical dependencies. {graph_string}"

        topological_order = list(nx.topological_sort(G))

        # Sort skills array in the same order as topological_order
        # Skills without prerequisites will be at the beginning
        self.skills = [self.get_skill(skill_name) for skill_name in topological_order]
        return self

    def get_skill(self, skill_name):
        matching_skills = [s for s in self.skills if s.name == skill_name]
        if len(matching_skills) > 1:
            raise ValueError(
                f"Skill names must be unique. Got a duplicate of {skill_name}."
            )
        elif len(matching_skills) == 0:
            raise ValueError(f"Requested skill was not found: {skill_name}")
        else:
            return matching_skills[0]


class Skill(Model):
    name: str = Field(..., description="Primary identifier.")
    code: Optional[str] = None
    description: Optional[str] = None
    prerequisites: PrerequisiteStructure = Field(
        default_factory=lambda: PrerequisiteStructure(parent_names=[]),
        description="Prerequisites for the skill.",
    )
    decay_logit: float = Field(
        ge=0.0,
        le=1.0,
        description="Natural temporal decay of the skill, representing how quickly it is forgotten, in days.",
        default=0.01,
    )
    practice_increment_logit: float = Field(
        ge=0.0,
        le=1.0,
        description="Increment to the skill level after one practice encounter.",
        default=0.1,
    )
    probability_of_learning_without_prerequisites: float = Field(
        ge=0.0,
        le=1.0,
        description="Baseline probability of the skill being learned without any prerequisites.",
        default=0.1,
    )
    probability_of_learning_with_prerequisites: float = Field(
        ge=0.0,
        le=1.0,
        description="Probability of the skill being learned with prerequisites.",
        default=0.9,
    )
    initial_skill_level: float = Field(
        ge=0.0,
        le=1.0,
        description="Initial skill level, representing the initial skill level after one learning encounter.",
        default=0.5,
    )

    def __init__(self, **data):
        # Delegate normal validation / ID assignment to the parent class first.
        super().__init__(**data)

        if self.code in (None, "None", ""):
            # The ID is unique per subclass thanks to the auto-incrementing
            # logic in `Model.__init__`, so we can safely derive a distinct
            # code from it.
            self.code = str(self.id)

        if not self.name:
            # Mirror the code when no explicit name is given.  This keeps the
            # original behaviour (name == id) while ensuring uniqueness.
            self.name = self.code

    def __eq__(self, other):
        """Name is a unique identifier since skills are unique by name."""
        if isinstance(other, Skill):
            return self.name == other.name

    @property
    def parents(self) -> list[str]:
        """Backward compatibility: return parent_names from prerequisites."""
        return self.prerequisites.parent_names

    @model_validator(mode="after")
    def validate(self):
        # Ensure that the skill has a name
        if not self.name:
            self.name = str(self.id)
            self.code = str(self.id)

        if self.prerequisites.parent_names:
            # Ensure you don't have the same name or code as a parent
            for parent in self.prerequisites.parent_names:
                if parent == self.name:
                    raise ValueError(
                        f"Skill name '{self.name}' cannot be the same as a parent skill name."
                    )
                if parent == self.code:
                    raise ValueError(
                        f"Skill code '{self.code}' cannot be the same as a parent skill code."
                    )
            if len(self.prerequisites.parent_names) != len(
                set(self.prerequisites.parent_names)
            ):
                raise ValueError("Parent skills must be unique.")

        # Ensure that the probability of learning with prerequisites is greater than the probability of learning without prerequisites
        if (
            self.probability_of_learning_with_prerequisites
            < self.probability_of_learning_without_prerequisites
        ):
            raise ValueError(
                "The probability of learning with prerequisites must be greater than the probability of learning without prerequisites."
            )

        return self


Skill.model_rebuild()


class Misconception(Model):
    parent_skill: Skill = Field(
        ..., description="The skill that this misconception is about."
    )
    description: Optional[str] = Field(
        ..., description="A description of the misconception."
    )
