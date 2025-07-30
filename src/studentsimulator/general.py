from __future__ import annotations

from typing import ClassVar, Literal, Optional

import matplotlib.pyplot as plt
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
    def validate_and_sort(self):
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

    def save_dependency_diagram(self, filename: str = "skillspace.png"):
        """Save a PNG diagram of the skill dependency structure (roots at top, leaves at bottom)."""
        G = nx.DiGraph()
        for skill in self.skills:
            G.add_node(skill.name)
            if skill.prerequisites is not None and hasattr(
                skill.prerequisites, "parent_names"
            ):
                for parent in skill.prerequisites.parent_names:
                    G.add_edge(parent, skill.name)
        plt.figure(figsize=(10, 7))
        try:
            # Try to use Graphviz for hierarchical layout
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except ImportError:
            try:
                pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
            except ImportError:
                print(
                    "Warning: pygraphviz or pydot not installed. Falling back to spring layout. For best results, install pygraphviz or pydot."
                )
                pos = nx.spring_layout(G, seed=42)
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color="lightblue",
            edge_color="gray",
            node_size=2000,
            font_size=10,
            font_weight="bold",
            arrowsize=20,
        )
        plt.title("Skill Dependency Structure (Hierarchical)")
        try:
            plt.tight_layout()
        except Exception:
            pass  # Ignore tight_layout errors
        plt.savefig(filename)
        plt.close()

    def plot_skill_mastery(self, students, filename="skill_mastery.png"):
        """
        Plot the skill dependency graph with node size proportional to the number of students who have each skill,
        and edge width proportional to the number of students who have both prerequisite and dependent skills.
        """
        from collections import defaultdict

        import matplotlib

        matplotlib.use("Agg")  # Ensure non-interactive backend
        import matplotlib.pyplot as plt
        import networkx as nx

        print("Generating skill mastery plot...")
        G = nx.DiGraph()
        for skill in self.skills:
            G.add_node(skill.name)
            if skill.prerequisites is not None and hasattr(
                skill.prerequisites, "parent_names"
            ):
                for parent in skill.prerequisites.parent_names:
                    G.add_edge(parent, skill.name)

        # Count students who have each skill (learned=True)
        skill_counts = defaultdict(int)
        for student in students:
            for skill_name, skill_state in student.skill_state.items():
                if getattr(skill_state, "learned", False):
                    skill_counts[skill_name] += 1

        # Count students who have both skills for each edge
        edge_counts = defaultdict(int)
        for student in students:
            learned_skills = {
                k
                for k, v in student.skill_state.items()
                if getattr(v, "learned", False)
            }
            for u, v in G.edges():
                if u in learned_skills and v in learned_skills:
                    edge_counts[(u, v)] += 1

        # Normalize node sizes and edge widths
        min_node_size = 300
        max_node_size = 3000
        min_edge_width = 1
        max_edge_width = 10
        if skill_counts:
            max_count = max(skill_counts.values())
        else:
            max_count = 1
        if edge_counts:
            max_edge_count = max(edge_counts.values())
        else:
            max_edge_count = 1
        node_sizes = [
            min_node_size
            + (max_node_size - min_node_size) * (skill_counts.get(n, 0) / max_count)
            for n in G.nodes()
        ]
        edge_widths = [
            min_edge_width
            + (max_edge_width - min_edge_width)
            * (edge_counts.get((u, v), 0) / max_edge_count)
            for u, v in G.edges()
        ]

        # Layout
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except Exception as e:
            print(
                "Warning: Could not use Graphviz layout for skill mastery plot. Reason:",
                e,
            )
            print(
                "Falling back to hierarchical layout. For best results, install Graphviz (dot)."
            )
            # Use hierarchical layout as fallback
            pos = nx.kamada_kawai_layout(G)
            # Adjust positions to be more hierarchical
            # Get topological sort to determine levels
            try:
                topo_order = list(nx.topological_sort(G))
                # Create a hierarchical layout manually
                levels = {}
                for node in topo_order:
                    # Find the level of this node (max level of parents + 1)
                    parent_levels = [
                        levels.get(pred, 0) for pred in G.predecessors(node)
                    ]
                    level = max(parent_levels) + 1 if parent_levels else 0
                    levels[node] = level

                # Position nodes by level
                max_level = max(levels.values()) if levels else 0
                for node, level in levels.items():
                    # Y position: higher level = higher Y (top of plot)
                    y = 1.0 - (level / max_level) if max_level > 0 else 0.5
                    # X position: distribute nodes at same level horizontally
                    nodes_at_level = [
                        n for n, level1 in levels.items() if level1 == level
                    ]
                    if len(nodes_at_level) > 1:
                        idx = nodes_at_level.index(node)
                        x = (idx - (len(nodes_at_level) - 1) / 2) / max(
                            len(nodes_at_level), 1
                        )
                    else:
                        x = 0
                    pos[node] = (x, y)
            except Exception:
                # If topological sort fails, use spring layout as last resort
                pos = nx.spring_layout(G, seed=42)
        nx.draw(
            G,
            pos,
            with_labels=False,  # We'll add custom labels
            node_color="lightblue",
            edge_color="gray",
            node_size=node_sizes,
            width=edge_widths,
            arrowsize=20,
        )

        # Add custom labels with skill name and count
        for node, (x, y) in pos.items():
            count = skill_counts.get(node, 0)
            plt.text(
                x,
                y,
                f"{node}\n({count})",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        plt.title("Skill Mastery and Dependencies")
        try:
            plt.tight_layout()
        except Exception:
            pass  # Ignore tight_layout errors
        plt.savefig(filename)
        plt.close()


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
    initial_skill_level_after_learning: float = Field(
        ge=0.0,
        le=1.0,
        description="Initial skill level, representing the initial skill level after one learning encounter.",
        default=0.5,
    )

    @model_validator(mode="before")
    def ensure_prerequisite_structure(cls, values):
        prereq = values.get("prerequisites")
        if prereq is not None and isinstance(prereq, dict):
            values["prerequisites"] = PrerequisiteStructure(**prereq)
        return values

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
