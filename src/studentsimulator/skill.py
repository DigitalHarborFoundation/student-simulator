import copy
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
from pydantic import Field, model_validator

from studentsimulator.general import Model
from studentsimulator.math import logistic, logit

if TYPE_CHECKING:
    from studentsimulator.event import ItemResponseEvent, LearningEvent, WaitEvent


class PrerequisiteStructure(Model):
    parent_names: list[str] = []
    dependence_model: Literal["any", "all"] = "all"


class Skill(Model):
    """A skill is a cognitive ability that a student can learn.
    This class encapsulates the properties of a particular skill,
    but not its instantiation in a particular student.
    """

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


class SkillState(Model):
    """Represents the skill state for a single student for a single skill at a single point in time.
    Skills are represented on a [0,1] scale.
    If a skill is 0, the student responds to skill-aligned items at chance level, i.e. with p=guess.
    If a skill is 1, the student responds to skill-aligned items at a high but not necessarily perfect level, i.e. with p=(1-slip).
    """

    skill: Skill
    skill_level: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Current skill level (0.0 to 1.0)"
    )
    learned: bool = Field(
        default=False,
        description="Whether the skill has been learned cognitively. Practice can't increase the skill level if the skill is not learned.",
    )


class SkillSpace(Model):
    """A SkillSpace is the collection of all skills that a student can learn.
    The validation methods ensure that skills are unique, prerequisites are valid, and
    the skills are topologically sorted based on their prerequisites."""

    skills: list["Skill"] = []

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

    skill_space: SkillSpace = Field(default_factory=SkillSpace)
    daily_skill_states: OrderedDict[int, Dict[str, SkillState]] = {
        0: {}
    }  # day -> {skill_name: level}

    # @model_validator(mode="after")
    # def set_placeholder_skill_states(self):
    #     self.daily_skill_states[0] = SkillState()

    @property
    def current_skill_levels(self) -> Dict[str, float]:
        """Get the current skill state for all skills."""
        if not self.daily_skill_states:
            return {}
        else:
            # Convert SkillState objects to skill levels (floats)
            current_states = self.daily_skill_states[
                max(self.daily_skill_states.keys())
            ]
            return {
                skill_name: skill_state.skill_level
                for skill_name, skill_state in current_states.items()
            }

    @property
    def current_skill_states(self) -> Dict[str, SkillState]:
        """Get the current skill state for all skills."""
        if not self.daily_skill_states:
            return {}
        else:
            return self.daily_skill_states[max(self.daily_skill_states.keys())]

    @model_validator(mode="after")
    def initialize_skills(self):
        """Initialize skills for day 0 based on the skill space that was provided to the Student constructor."""
        if 0 in self.daily_skill_states and not self.daily_skill_states[0]:
            # Initialize empty skill states for all skills in skill space
            for skill in self.skill_space.skills:
                self.daily_skill_states[0][skill.name] = SkillState(
                    skill=skill, skill_level=0.0, learned=False
                )
        return self

    def __getitem__(self, skill_name: str) -> SkillState:
        """Get the skill state for a skill."""
        return self.current_skill_states[skill_name]

    def _apply_practice(self, skill_state: "SkillState", student) -> float:
        """Return new level after one practice increment (if learned)."""
        if not skill_state.learned:
            # practice is not effective if the skill is not learned
            return skill_state.skill_level
        else:
            # if the skill is learned, the practice increment is applied
            new_logit = (
                logit(skill_state.skill_level)
                + skill_state.skill.practice_increment_logit
            )
            new_level = logistic(new_logit)

            # Apply transfer to ancestor skills
            self._update_ancestor_skills(
                student,
                skill_state.skill,
                skill_state.skill.practice_increment_logit,
                1,
            )

            return new_level

    @staticmethod
    def _apply_forgetting(level: float, skill: "Skill", days: int) -> float:
        """Return new level after forgetting for *days* days."""
        if days <= 0:
            return level
        new_logit = logit(level) - skill.decay_logit * days
        min_logit = logit(0.01)
        return logistic(max(min_logit, new_logit))

    def _update_ancestor_skills(
        self, student, skill: "Skill", base_increment: float, depth: int
    ):
        transfer_factor = 0.3  # 30% of the practice effect transfers to each level
        if skill.prerequisites and skill.prerequisites.parent_names:
            for parent_name in skill.prerequisites.parent_names:
                if (
                    parent_name
                    in student.skills.end_of_day_skill_states.current_skill_states
                    and student.skills.end_of_day_skill_states.current_skill_states[
                        parent_name
                    ].learned
                ):
                    parent_skill = self.skill_space.get_skill(parent_name)
                    transfer_increment = base_increment * (transfer_factor**depth)
                    student.skills.end_of_day_skill_states.current_skill_states[
                        parent_name
                    ].skill_level = logistic(
                        logit(
                            student.skills.end_of_day_skill_states.current_skill_states[
                                parent_name
                            ].skill_level
                        )
                        + transfer_increment
                    )

                    self._update_ancestor_skills(
                        student, parent_skill, base_increment, depth + 1
                    )

    def get_skill_states(self) -> dict[str, SkillState]:
        """Get the skill states for all skills."""
        return self.current_skill_states

    def get_skill_level_for_single_skill(self, skill_name: str) -> float:
        """Get the skill level for a skill."""
        return self.current_skill_states[skill_name].skill_level

    def get_skill_state_for_single_skill(self, skill_name: str) -> SkillState:
        """Get the skill state for a skill."""
        return self.current_skill_states[skill_name]

    def print_skill_states(self) -> str:
        """Print the skill states for the student."""
        print("debugging")
        return "\t" + "\n\t".join(
            [
                f"{skill}: {level:.2f}"
                for skill, level in self.current_skill_levels.items()
            ]
        )

    def update_skill_state(self, skill_name: str, level: float) -> None:
        """Update the skill state for a skill."""
        self.daily_skill_states[max(self.daily_skill_states.keys())][
            skill_name
        ].skill_level = level
        return self

    def update_skill_states_after_event(
        self, student, event: Union["LearningEvent", "ItemResponseEvent", "WaitEvent"]
    ) -> None:
        """Update the skill states based on the event.
        Recall that the daily_skill_states is an ordered dictionary
        where the last entry is our current day.

        If the event is one of these, update the current day's skills:
        - LearningEvent
        - ItemResponseEvent

        If the event is one of these, update the previous day's skills:
        - WaitEvent
        """

        if event.__class__.__name__ == "LearningEvent":
            # The skill is now learned if it was already learned or if it was learned in this event
            skill_is_learned = (
                student.skills[event.skill.name].learned | event.final_learned
            )
            if skill_is_learned:
                # If the skill level is less than the initial skill level, increase it to equal the initial skill level
                if (
                    student.skills[event.skill.name].skill_level
                    < event.skill.initial_skill_level_after_learning
                ):
                    student.skills[event.skill.name].learned = True
                    student.skills[
                        event.skill.name
                    ].skill_level = event.skill.initial_skill_level_after_learning

        if event.__class__.__name__ == "WaitEvent":
            # For waiting, we add new days to the end of the daily_skill_states dictionary,
            # and when we do so, we apply forgetting to all skills to update the skill levels.
            days_waited = event.days_waited
            current_day = (
                max(self.daily_skill_states.keys()) if self.daily_skill_states else 0
            )

            # Add new days and apply forgetting
            for day in range(current_day + 1, current_day + days_waited + 1):
                # Start with the previous day's skill states
                if day - 1 in self.daily_skill_states:
                    previous_states = copy.deepcopy(self.daily_skill_states[day - 1])
                else:
                    previous_states = {}

                # Apply forgetting to all skills
                for skill_name, skill_state in previous_states.items():
                    # Apply forgetting for 1 day
                    new_level = self._apply_forgetting(
                        level=skill_state.skill_level, skill=skill_state.skill, days=1
                    )
                    previous_states[skill_name].skill_level = new_level
                # Store the new day's skill states
                self.daily_skill_states[day] = previous_states

        if event.__class__.__name__ == "ItemResponseEvent":
            student.skills[event.skill.name].skill_level = self._apply_practice(
                skill_state=student.skills[event.skill.name], student=student
            )

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
        for levels in self.daily_skill_states.values():
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
                for day, levels in sorted(self.daily_skill_states.items())
            ]
        return trajectories


class StudentSkills(Model):
    """List of events associated with a student that would either:
    - affect their skill state (learning, practicing, waiting)
    - reveal their skill state (responding to an item, taking an assessment)

    Every entry in the events list is a subclass of BehaivorEvent or BehaviorEventCollection.
    """

    skill_space: SkillSpace = Field(default_factory=SkillSpace)
    events: List[Any] = Field(default_factory=list)
    end_of_day_skill_states: EndOfDaySkillStates = Field(
        default_factory=EndOfDaySkillStates
    )

    def __init__(self, **data):
        super().__init__(**data)
        self.end_of_day_skill_states = EndOfDaySkillStates(skill_space=self.skill_space)

    def __iter__(self):
        """Allow iteration over skills in the skill space."""
        return iter(self.skill_space.skills)

    def __getitem__(self, skill_name: str) -> SkillState:
        """Get the skill state for a skill."""
        return self.end_of_day_skill_states.get_skill_state_for_single_skill(
            skill_name=skill_name
        )

    def print_skill_states(self) -> str:
        """Print the skill states for the student."""
        return self.end_of_day_skill_states.print_skill_states()

    def print_event_history(self) -> str:
        """Print the event history for the student."""
        return self.end_of_day_skill_states.print_event_history()

    def print_daily_history(self) -> str:
        """Print the daily history for the student."""
        return self.end_of_day_skill_states.print_daily_history()

    def set_skill_level(self, skill_name: str, level: float) -> None:
        """Set the skill level for a skill.
        If the skill is not registered, create a new state.
        """
        self.end_of_day_skill_states.update_skill_state(skill_name, level)
        return self

    def get_skill_states(self) -> dict[str, SkillState]:
        """Get the skill states for all skills."""
        return self.end_of_day_skill_states.get_skill_states()

    def get_individual_events(self, event_types: List[str] = None) -> List[Any]:
        """Get all individual events from the history, optionally filtered by event type.

        args:
            event_types: List[str] = None, optional. If provided, only return events of these types.
        """
        flattened_events = []
        for event in self.events:
            if type(event).__name__ == "BehaviorEventCollection":
                flattened_events.extend(event.behavioral_events)
            else:
                flattened_events.append(event)

        # Filter by event type if provided
        if event_types is not None:
            flattened_events = [
                event
                for event in flattened_events
                if type(event).__name__ in event_types
            ]

        return flattened_events

    def record_event(self, student, event: Any) -> None:
        """Add an event to the history."""
        self.events.append(event)

        # Update end-of-day skill states
        self.end_of_day_skill_states.update_skill_states_after_event(student, event)

    def clear(self) -> None:
        """Clear all events from history."""
        self.events.clear()

    ##### Methods for initializing student skill state #####

    ##### Methods for updating student skill state after actions #####

    def is_learned(self, skill: "Skill") -> bool:
        """Check if the student has learned a skill."""
        return self[skill.name].learned

    def has_prerequisites(self, skill: "Skill") -> bool:
        """Check if the student has the prerequisites for a skill."""
        if not skill.prerequisites.parent_names:
            return True
        if skill.prerequisites.dependence_model == "any":
            return any(
                self.skill_state[
                    prerequisite
                ].learned  # skill_state is not defined in this scope
                for prerequisite in skill.prerequisites.parent_names
            )
        elif skill.prerequisites.dependence_model == "all":
            return all(
                self.end_of_day_skill_states[
                    prerequisite
                ].learned  # skill_state is not defined in this scope
                for prerequisite in skill.prerequisites.parent_names
            )
        else:
            raise ValueError(
                f"Invalid dependence model: {skill.prerequisites.dependence_model}"
            )


Skill.model_rebuild()


class Misconception(Model):
    parent_skill: Skill = Field(
        ..., description="The skill that this misconception is about."
    )
    description: Optional[str] = Field(
        ..., description="A description of the misconception."
    )
