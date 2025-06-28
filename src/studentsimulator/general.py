from __future__ import annotations

from typing import ClassVar, List, Literal, Optional

import networkx as nx
from pydantic import BaseModel, Field, model_validator


class Model(BaseModel):
    # one counter per subclass, created automatically
    _counter: ClassVar[int] = 0  # ensure base class can be instantiated

    id: Optional[int] = None  # not required—filled in later

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._counter = 0  # fresh counter for this subclass   ──┐
        # PEP 487 guarantees this runs once │
        # per subclass                      │
        #                                   │
        #                                   ▼
        # [oai_citation:1‡peps.python.org](https://peps.python.org/pep-0487/?utm_source=chatgpt.com)

    def __init__(self, **data):
        super().__init__(**data)  # normal Pydantic validation
        if self.id is None:  # assign only if caller didn’t supply one
            cls = self.__class__
            self.id = cls._counter
            cls._counter += 1


class PrerequisiteStructure(Model):
    parent_names: list[str] = []
    dependence_model: Literal["", "any", "all"] = ""
    prior: float = Field(
        ge=0.0,
        le=1.0,
        description="Probability that the student can learn the skill without having the prerequisites.",
        default=0.0,
    )


class SkillSpace(Model):
    skills: list[Skill] = []

    @model_validator(mode="after")
    def guarantee_unique(self):
        skill_names = set([s.name for s in self.skills])
        for skill_name in skill_names:
            # get_skill will validate that there is just 1
            _ = self.get_skill(skill_name=skill_name)
        return self

    @model_validator(mode="after")
    def topological_sort(self):
        # Reorder the skills using aopological sort
        # Do a topological sort on the skills
        # Make string representation with all nodes for error printing in assertion
        G = nx.DiGraph()
        for skill in self.skills:
            for parent_name in skill.prerequisite.parent_names:
                G.add_edge(parent_name, skill.name)
        assert nx.is_directed_acyclic_graph(
            G
        ), "The set of metric dependencies must be acyclic! You have cyclical dependencies. {graph_string}"
        topological_order = list(nx.topological_sort(G))
        # self.skills = sorted(self.skills, key=topological_order)
        # sort skills array in the same order as  using the skill_name
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
    _all_skills: ClassVar[List["Skill"]] = []

    name: str = ""  # Primary identifier
    code: Optional[str] = None
    description: Optional[str] = None
    prerequisite: Optional[PrerequisiteStructure] = Field(
        default_factory=PrerequisiteStructure
    )
    decay: float = 0.0

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

        # Finally register the skill instance.
        Skill._all_skills.append(self)

    def __eq__(self, other):
        """Name is a unique identifier since skills are unique by name."""
        if isinstance(other, Skill):
            return self.name == other.name

    @model_validator(mode="after")
    def validate(self):
        # Ensure that the skill has a name
        if not self.name:
            self.name = str(self.id)
            self.code = str(self.id)

        if self.prerequisite.parent_names:
            # Ensure you don't have the same name or code as a parent
            for parent in self.prerequisite.parent_names:
                if parent == self.name:
                    raise ValueError(
                        f"Skill name '{self.name}' cannot be the same as a parent skill name."
                    )
                if parent == self.code:
                    raise ValueError(
                        f"Skill code '{self.code}' cannot be the same as a parent skill code."
                    )
            if len(self.prerequisite.parent_names) != len(
                set(self.prerequisite.parent_names)
            ):
                raise ValueError("Parent skills must be unique.")

        # Ensure that the code is unique across all skills
        if self.code not in (None, "", "None"):
            all_skills = Skill._all_skills
            if any(
                (skill.code == self.code and skill.id != self.id)
                for skill in all_skills
                if skill.code is not None
            ):
                raise ValueError(
                    f"Skill code '{self.code}' must be unique across all skills."
                )
        # Ensure that

        return self


Skill.model_rebuild()


class Misconception(Model):
    parent_skill: Skill = None
