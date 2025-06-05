"""
Student Response Simulation Framework - Main API
"""

from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from .config import ItemsConfig, PopulationConfig, SimulationConfig, SkillsConfig


class Sim:
    """Fluent API for student response simulation."""

    def __init__(self, seed: int = 42):
        self.config = SimulationConfig()
        self.config.rng.seed = seed
        self._rng = np.random.default_rng(seed)

    def skills(self, skills_data: List[Dict[str, Any]]) -> "Sim":
        """Configure skills for the simulation."""
        from .config import Skill

        skills_list = [Skill.model_validate(skill) for skill in skills_data]
        self.config.skills = SkillsConfig(skills=skills_list)
        return self

    def items(self, items_data: List[Dict[str, Any]]) -> "Sim":
        """Configure items for the simulation."""
        from .config import Item

        items_list = [Item.model_validate(item) for item in items_data]
        self.config.items = ItemsConfig(items=items_list)
        return self

    def population(self, population_data: List[Dict[str, Any]]) -> "Sim":
        """Configure population for the simulation."""
        from .config import Student

        # Handle both old group format and new student format
        students_list = []
        for item in population_data:
            if "name" in item and "size" in item:
                # Old group format - generate individual students
                for i in range(item["size"]):
                    student = Student(
                        id=f"stu{i+1}", initial_theta=item.get("theta_shift", 0.0)
                    )
                    students_list.append(student)
            else:
                # New student format
                student = Student.model_validate(item)
                students_list.append(student)

        self.config.population = PopulationConfig(students=students_list)
        return self

    def learning(self, mode: str = "hybrid", **kwargs: Any) -> "Sim":
        """Configure learning parameters with psychometric model mode."""
        self.config.learning.mode = mode

        # Apply mode-specific defaults
        mode_config = self.config.learning.get_mode_config()
        for key, value in mode_config.items():
            if hasattr(self.config.learning, key):
                setattr(self.config.learning, key, value)

        # Apply any user overrides
        for key, value in kwargs.items():
            if hasattr(self.config.learning, key):
                setattr(self.config.learning, key, value)

        # Auto-configure items and interventions based on mode
        self._apply_mode_to_items_and_interventions(mode_config)

        return self

    def _apply_mode_to_items_and_interventions(
        self, mode_config: Dict[str, Any]
    ) -> None:
        """Apply mode configuration to items and default intervention parameters."""
        # Update default item parameters based on mode
        default_g = mode_config.get("default_guess_parameter", 0.2)
        default_s = mode_config.get("default_slip_parameter", 0.1)
        default_effectiveness = mode_config.get("default_practice_effectiveness", 0.2)

        # Apply to existing items if they don't have custom values
        for item in self.config.items.items:
            if item.g == 0.2:  # Default value, update it
                item.g = default_g
            if item.s == 0.1:  # Default value, update it
                item.s = default_s
            if (
                not hasattr(item, "practice_effectiveness")
                or item.practice_effectiveness == 0.0
            ):
                item.practice_effectiveness = default_effectiveness

    def interventions(self, interventions_data: Dict[str, Any]) -> "Sim":
        """Configure interventions."""
        self.config.interventions = interventions_data
        return self

    def helpers(self, helpers_data: Dict[str, Any]) -> "Sim":
        """Configure helper settings like scheduling."""
        self.config.helpers.scheduler = helpers_data.get("scheduling")
        return self

    def run(
        self, event_stream: List[Dict[str, Any]], return_config: bool = False
    ) -> Union[Tuple[pd.DataFrame, Any], Tuple[pd.DataFrame, Any, Dict[str, Any]]]:
        """Run the simulation with the given event stream."""
        # Generate initial latent state
        latent_state = generate_latent(self.config.model_dump())

        # Process each event in the stream
        responses = []
        for event in event_stream:
            # Update state based on event (handles interventions and practice)
            latent_state = transition(latent_state, event)

            # Find the item for this event
            item_data = None
            for item in self.config.items.items:
                if item.id == event["item_id"]:
                    item_data = item.model_dump()
                    break

            if item_data and event.get("observed", 1):
                # Generate response
                response = emit_response(latent_state, item_data)
                responses.append(
                    {
                        "student_id": event["student_id"],
                        "item_id": event["item_id"],
                        "time": event["time"],
                        "response": response,
                    }
                )

                # Record the response in student's learning history
                student_id = event["student_id"]
                if (
                    "students" in latent_state
                    and student_id in latent_state["students"]
                ):
                    student = latent_state["students"][student_id]
                    student.record_response(
                        timestamp=event["time"],
                        item_id=event["item_id"],
                        response=response,
                        intervention_type=event.get("intervention_type"),
                        **event.get("context", {}),
                    )

        # Convert to DataFrame
        response_log = pd.DataFrame(responses)

        if return_config:
            return response_log, latent_state, self.config.model_dump()
        else:
            return response_log, latent_state

    def next_item_for(self, student_id: str, current_state: Dict[str, Any]) -> str:
        """Get next item for adaptive scheduling."""
        # Simple implementation: return first item for now
        # TODO: Use student_id and current_state for adaptive selection
        if self.config.items.items:
            return self.config.items.items[0].id
        return "item1"


def generate_latent(config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate initial latent state from config."""
    # Initialize students from population config
    students = {}
    rng = np.random.default_rng(config.get("rng", {}).get("seed", 42))

    for student_data in config.get("population", {}).get("students", []):
        student_id = student_data["id"]
        # Create student object with initial skills
        from .config import SkillState, Student

        student = Student.model_validate(student_data)

        # Initialize skills in "no_learning" state
        for skill in config.get("skills", {}).get("skills", []):
            skill_id = skill["id"]
            if skill_id not in student.current_skills:
                student.current_skills[skill_id] = SkillState(
                    skill_id=skill_id,
                    proficiency_state="no_learning",
                    current_proficiency=0.0,
                    baseline_proficiency=0.0,
                    max_proficiency=1.0,
                )

        students[student_id] = student

    return {"students": students, "rng": rng, "config": config}


def transition(state: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
    """Update state based on event using hybrid learning model."""
    student_id = event["student_id"]
    timestamp = event.get("time", 0)

    students = state.get("students", {})
    rng = state.get("rng", np.random.default_rng())
    config = state.get("config", {})

    # Get mode-specific defaults
    learning_config = config.get("learning", {})
    mode_defaults = {
        "learning_probability": learning_config.get(
            "default_learning_probability", 0.8
        ),
        "baseline_proficiency": learning_config.get(
            "default_baseline_proficiency", 0.6
        ),
        "practice_effectiveness": learning_config.get(
            "default_practice_effectiveness", 0.2
        ),
    }

    if student_id in students:
        student = students[student_id]

        # Handle interventions (probabilistic learning)
        intervention_type = event.get("intervention_type")
        context = event.get("context", {})

        if intervention_type == "skill_boost":
            # Legacy compatibility - convert to new intervention format
            target_skill = context.get("target_skill")
            boost = context.get("boost", 0.0)
            if target_skill:
                from .config import InterventionEvent

                # For CDM mode, adjust learning probability based on prerequisites
                learning_prob = mode_defaults["learning_probability"]
                if config.get("learning", {}).get("mode") == "cdm":
                    # Find the skill definition to check prerequisites
                    skill_def = None
                    for skill_data in config.get("skills", {}).get("skills", []):
                        if skill_data.get("id") == target_skill:
                            from .config import Skill

                            skill_def = Skill.model_validate(skill_data)
                            break

                    if skill_def and skill_def.parents:
                        # Get current proficiency of parent skills
                        parent_states = {}
                        for parent_id in skill_def.parents:
                            parent_skill = student.get_skill_state(parent_id)
                            parent_states[parent_id] = parent_skill.current_proficiency

                        # Adjust learning probability based on prerequisites
                        learning_prob = skill_def.get_cdm_learning_probability(
                            parent_states, learning_prob
                        )

                intervention = InterventionEvent(
                    student_id=student_id,
                    timestamp=timestamp,
                    intervention_type="skill_boost",
                    target_skill=target_skill,
                    learning_probability=learning_prob,
                    baseline_proficiency=min(
                        boost, mode_defaults["baseline_proficiency"]
                    ),
                )
                student.apply_intervention(intervention, rng)

        # Apply practice effects when student responds to items
        if event.get("observed", True) and "item_id" in event:
            # For now, practice all skills associated with the student
            # In full implementation, would look up item's specific skills
            for skill_id in student.current_skills:
                student.practice_skill(
                    skill_id,
                    practice_effectiveness=mode_defaults["practice_effectiveness"],
                    timestamp=timestamp,
                )

    return state


def emit_response(state: Dict[str, Any], item: Dict[str, Any]) -> int:
    """Generate response using hybrid BKT/CDM/IRT model."""
    students = state.get("students", {})
    rng = state.get("rng", np.random.default_rng())
    config = state.get("config", {})

    if not students:
        # Fallback for legacy state structure
        return int(rng.random() < item.get("g", 0.2))

    # Use first student (in real implementation, would specify which student)
    first_student = next(iter(students.values()))

    # Get item parameters
    guess_param = item.get("g", 0.2)
    slip_param = item.get("s", 0.1)

    # Get learning mode
    learning_mode = config.get("learning", {}).get("mode", "hybrid")

    # Calculate probability based on skills required by item
    item_skills = item.get("skills", [])
    q_vector = item.get("q_vector", {})

    skill_probabilities = []

    if item_skills:
        # Use direct skill objects
        for skill_dict in item_skills:
            skill_id = skill_dict.get("id")
            if skill_id:
                skill_state = first_student.get_skill_state(skill_id)
                prob = skill_state.get_probability_correct(
                    guess_param, slip_param, learning_mode
                )
                skill_probabilities.append(prob)
    elif q_vector:
        # Use legacy q_vector approach
        for skill_id, required in q_vector.items():
            if required:
                skill_state = first_student.get_skill_state(skill_id)
                prob = skill_state.get_probability_correct(
                    guess_param, slip_param, learning_mode
                )
                skill_probabilities.append(prob)

    if not skill_probabilities:
        # No skills specified, use guess parameter
        prob_correct = guess_param
    else:
        # For CDM: items should target exactly one skill (no combination needed)
        if learning_mode == "cdm":
            prob_correct = skill_probabilities[0]  # Use first (should be only) skill
        else:
            # Combine probabilities (using geometric mean for multiple skills)
            prob_correct = np.prod(skill_probabilities) ** (
                1.0 / len(skill_probabilities)
            )

    # Generate response
    return int(rng.random() < prob_correct)
