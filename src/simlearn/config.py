from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class RNGConfig(BaseModel):
    seed: int = 42


class Skill(BaseModel):
    id: str
    code: Optional[str] = None  # external standard code (e.g., "CCSS.MATH.5.NBT.A.1")
    description: Optional[str] = None
    domain: Optional[str] = None  # subject area or domain
    type: str = "continuous"  # continuous, binary, ordinal
    parents: List[str] = []  # prerequisite skill IDs
    practice_gain: float = 0.0
    decay: float = 0.0

    def get_cdm_learning_probability(
        self, parent_states: Dict[str, float], base_learning_prob: float = 0.3
    ) -> float:
        """Get learning probability based on prerequisite mastery (for CDM mode)"""
        if not self.parents:
            return base_learning_prob

        # Calculate how many prerequisites are "mastered" (proficiency > threshold)
        mastery_threshold = 0.5
        met_prereqs = sum(
            1 for p in self.parents if parent_states.get(p, 0.0) > mastery_threshold
        )
        total_prereqs = len(self.parents)

        if met_prereqs == 0:
            return 0.05  # Very low chance without prerequisites
        elif met_prereqs == total_prereqs:
            return base_learning_prob  # Full chance with all prerequisites
        else:
            # Partial prerequisites - reduced probability
            prereq_ratio = met_prereqs / total_prereqs
            return base_learning_prob * (0.3 + 0.7 * prereq_ratio)


class SkillsConfig(BaseModel):
    skills: List[Skill] = []


class Misconception(BaseModel):
    id: str
    parent_skill: str
    pi: float


class Item(BaseModel):
    id: str
    title: Optional[str] = None
    skills: List[Skill] = []  # skills this item measures
    q_vector: Dict[str, int] = {}  # legacy support - skill_id -> binary requirement
    g: float = 0.2  # guess parameter
    s: float = 0.1  # slip parameter
    a: float = 1.0  # discrimination (unused in hybrid model)
    b: float = 0.0  # difficulty (unused in hybrid model)
    practice_effectiveness: float = 0.1  # how much this item helps when practiced
    options: List[Any] = []  # multiple choice options
    bug_map: Dict[str, int] = {}  # misconception -> distractor mapping


class ItemsConfig(BaseModel):
    items: List[Item] = []


class SkillState(BaseModel):
    skill_id: str
    proficiency_state: str = "no_learning"  # "no_learning", "baseline", "mastery"
    baseline_proficiency: float = 0.0  # proficiency after intervention
    current_proficiency: float = 0.0  # current proficiency (increases with practice)
    max_proficiency: float = 1.0  # maximum achievable proficiency
    practice_count: int = 0
    last_practiced: Optional[int] = None  # timestamp

    def get_probability_correct(
        self, item_guess: float, item_slip: float, mode: str = "hybrid"
    ) -> float:
        """Calculate P(correct) based on current proficiency state and item parameters."""
        if mode == "cdm":
            # CDM binary model: learned vs not learned
            if self.current_proficiency > 0.5:  # "Learned" threshold
                return 1.0 - item_slip
            else:
                return item_guess
        else:
            # Original hybrid model
            if self.proficiency_state == "no_learning":
                return item_guess
            elif self.proficiency_state == "mastery":
                return 1.0 - item_slip
            else:  # baseline or intermediate
                # Linear interpolation between baseline and mastery
                progress = min(self.current_proficiency / self.max_proficiency, 1.0)
                baseline_prob = self.baseline_proficiency
                mastery_prob = 1.0 - item_slip
                return baseline_prob + progress * (mastery_prob - baseline_prob)


class BehaviorRepresentation(BaseModel):
    """What the student actually did"""

    behavior_type: str  # "item_selection", "text_response", "interaction", etc.
    item_id: Optional[str] = None  # for item-based behaviors
    selected_option: Optional[str] = None  # for multiple choice
    text_response: Optional[str] = None  # for open-ended responses
    interaction_data: Optional[Dict[str, Any]] = None  # for complex interactions


class Feedback(BaseModel):
    """Evaluation of the student's behavior"""

    feedback_type: str  # "binary", "score", "rubric", "qualitative"
    correct: Optional[bool] = None  # for binary feedback
    score: Optional[float] = None  # for numeric scores
    max_score: Optional[float] = None  # scale information
    feedback_text: Optional[str] = None  # qualitative feedback
    rubric_scores: Optional[Dict[str, float]] = None  # detailed rubric


class InterventionEvent(BaseModel):
    """Something that happens TO the student"""

    student_id: str
    timestamp: int
    intervention_type: str  # "lesson", "video", "hint", "tutoring", "feedback"
    target_skill: Optional[str] = None
    target_misconception: Optional[str] = None
    intervention_data: Dict[str, Any] = {}  # intervention-specific parameters
    learning_probability: float = 0.8  # probability student learns the skill
    baseline_proficiency: float = 0.6  # proficiency level if learning occurs


class BehavioralEvent(BaseModel):
    """Something the student DOES, with potential feedback"""

    student_id: str
    timestamp: int
    behavior: BehaviorRepresentation
    feedback: Optional[Feedback] = None
    context: Dict[str, Any] = {}  # additional metadata


class LearningEvent(BaseModel):
    """Internal learning state changes (for student history)"""

    timestamp: int
    event_type: str  # "skill_update", "misconception_change", etc.
    skill_changes: Dict[str, float] = {}  # skill_id -> change amount
    misconception_changes: Dict[str, bool] = {}  # misconception_id -> active
    context: Dict[str, Any] = {}


class Student(BaseModel):
    id: str
    name: Optional[str] = None
    initial_theta: float = 0.0  # base ability
    current_skills: Dict[str, SkillState] = {}  # skill_id -> current state
    misconceptions: Dict[str, bool] = {}  # misconception_id -> active
    learning_history: List[LearningEvent] = []  # complete history
    metadata: Dict[str, Any] = {}  # custom attributes

    def get_skill_level(self, skill_id: str) -> float:
        """Get current proficiency level for a skill."""
        if skill_id not in self.current_skills:
            return 0.0
        return self.current_skills[skill_id].current_proficiency

    def get_skill_state(self, skill_id: str) -> "SkillState":
        """Get complete skill state for a skill."""
        if skill_id not in self.current_skills:
            self.current_skills[skill_id] = SkillState(skill_id=skill_id)
        return self.current_skills[skill_id]

    def apply_intervention(self, intervention: "InterventionEvent", rng: Any) -> bool:
        """Apply intervention and return whether learning occurred."""
        if not intervention.target_skill:
            return False

        skill_state = self.get_skill_state(intervention.target_skill)

        # Check if learning occurs (probabilistic)
        if rng.random() < intervention.learning_probability:
            # Student learns the skill
            skill_state.proficiency_state = "baseline"
            skill_state.baseline_proficiency = intervention.baseline_proficiency
            skill_state.current_proficiency = intervention.baseline_proficiency

            # Record learning event
            event = LearningEvent(
                timestamp=intervention.timestamp,
                event_type="skill_learned",
                skill_changes={
                    intervention.target_skill: intervention.baseline_proficiency
                },
                context={"intervention_type": intervention.intervention_type},
            )
            self.learning_history.append(event)
            return True
        return False

    def practice_skill(
        self, skill_id: str, practice_effectiveness: float, timestamp: int
    ) -> None:
        """Update skill proficiency through practice."""
        skill_state = self.get_skill_state(skill_id)

        if skill_state.proficiency_state == "no_learning":
            # No effect if skill not yet learned
            return

        # Increase proficiency up to maximum
        old_proficiency = skill_state.current_proficiency
        skill_state.current_proficiency = min(
            skill_state.current_proficiency + practice_effectiveness,
            skill_state.max_proficiency,
        )

        # Check if mastery achieved
        if skill_state.current_proficiency >= skill_state.max_proficiency:
            skill_state.proficiency_state = "mastery"

        skill_state.practice_count += 1
        skill_state.last_practiced = timestamp

        # Record practice event if there was improvement
        improvement = skill_state.current_proficiency - old_proficiency
        if improvement > 0:
            event = LearningEvent(
                timestamp=timestamp,
                event_type="skill_practiced",
                skill_changes={skill_id: improvement},
            )
            self.learning_history.append(event)

    def record_behavioral_event(self, behavioral_event: "BehavioralEvent") -> None:
        """Record a behavioral event in learning history."""
        # Convert behavioral event to internal learning event
        learning_event = LearningEvent(
            timestamp=behavioral_event.timestamp,
            event_type="behavioral_event",
            context={
                "behavior": behavioral_event.behavior.model_dump(),
                "feedback": (
                    behavioral_event.feedback.model_dump()
                    if behavioral_event.feedback
                    else None
                ),
                **behavioral_event.context,
            },
        )
        self.learning_history.append(learning_event)

    def record_intervention_event(
        self, intervention_event: "InterventionEvent"
    ) -> None:
        """Record an intervention event in learning history."""
        learning_event = LearningEvent(
            timestamp=intervention_event.timestamp,
            event_type="intervention_event",
            context={
                "intervention_type": intervention_event.intervention_type,
                "target_skill": intervention_event.target_skill,
                "intervention_data": intervention_event.intervention_data,
            },
        )
        self.learning_history.append(learning_event)

    def record_response(
        self, timestamp: int, item_id: str, response: int, **context: Any
    ) -> None:
        """Record a response event (backward compatibility)."""
        # Convert to new behavioral event structure
        behavior = BehaviorRepresentation(
            behavior_type="item_selection",
            item_id=item_id,
            selected_option=str(response),  # Convert response to string option
        )

        feedback = None
        if "intervention_type" in context:
            # This was an intervention-related response, create minimal feedback
            feedback = Feedback(
                feedback_type="binary",
                correct=bool(response),  # Simple assumption: 1=correct, 0=incorrect
            )

        behavioral_event = BehavioralEvent(
            student_id=self.id,
            timestamp=timestamp,
            behavior=behavior,
            feedback=feedback,
            context=context,
        )

        self.record_behavioral_event(behavioral_event)


class PopulationConfig(BaseModel):
    students: List[Student] = []


class LearningConfig(BaseModel):
    mode: str = "hybrid"  # "bkt", "pfa", "irt", "hybrid"
    global_kappa: float = 0.0
    interference: bool = False

    # Mode-specific defaults
    default_learning_probability: float = 0.8
    default_baseline_proficiency: float = 0.6
    default_practice_effectiveness: float = 0.2
    default_guess_parameter: float = 0.2
    default_slip_parameter: float = 0.1

    def get_mode_config(self) -> Dict[str, Any]:
        """Get parameter configuration for the specified psychometric model mode."""
        mode_configs = {
            "bkt": {
                "description": "Bayesian Knowledge Tracing - Binary learning states",
                "default_learning_probability": 0.3,
                "default_baseline_proficiency": 1.0,  # Jump to mastery
                "default_practice_effectiveness": 1.0,  # Binary transitions
                "default_guess_parameter": 0.25,
                "default_slip_parameter": 0.05,
                "learning_characteristics": "Probabilistic skill acquisition, binary states",
            },
            "pfa": {
                "description": "Performance Factor Analysis - Gradual improvement",
                "default_learning_probability": 0.9,  # Learning usually occurs
                "default_baseline_proficiency": 0.4,  # Moderate starting point
                "default_practice_effectiveness": 0.15,  # Gradual improvement
                "default_guess_parameter": 0.15,
                "default_slip_parameter": 0.08,
                "learning_characteristics": "Practice-driven gradual improvement",
            },
            "irt": {
                "description": "Item Response Theory - No learning, fixed ability",
                "default_learning_probability": 0.0,  # No learning
                "default_baseline_proficiency": 0.0,  # Not applicable
                "default_practice_effectiveness": 0.0,  # No practice effects
                "default_guess_parameter": 0.2,
                "default_slip_parameter": 0.1,
                "learning_characteristics": "Static ability, no learning dynamics",
            },
            "hybrid": {
                "description": "Hybrid Model - All psychometric features enabled",
                "default_learning_probability": 0.7,
                "default_baseline_proficiency": 0.6,
                "default_practice_effectiveness": 0.25,
                "default_guess_parameter": 0.18,
                "default_slip_parameter": 0.07,
                "learning_characteristics": "Combines BKT, PFA, and IRT features",
            },
            "cdm": {
                "description": "Dynamic Cognitive Diagnostic Model - Binary skills with prerequisites",
                "default_learning_probability": 0.3,  # BKT-style learning
                "default_baseline_proficiency": 0.7,  # When skill first learned
                "default_practice_effectiveness": 0.8,  # High practice effects for mastery
                "default_guess_parameter": 0.25,
                "default_slip_parameter": 0.05,
                "learning_characteristics": "Binary skill states with prerequisite networks",
            },
        }

        return mode_configs.get(self.mode, mode_configs["hybrid"])


class InterventionSpec(BaseModel):
    target_skill: Optional[str] = None
    target_bug: Optional[str] = None
    action: str
    success_prob: float = 1.0
    delta_pi: Optional[float] = None
    delta_theta: Optional[float] = None


class MissingnessConfig(BaseModel):
    mode: str = "MCAR"
    p_global: float = 0.0
    skill_rates: Dict[str, float] = {}
    student_sd: float = 0.0


class OutputConfig(BaseModel):
    dir: str = "output"
    format: str = "csv"
    save_latent: bool = True


class HelperConfig(BaseModel):
    scheduler: Optional[Dict[str, Any]] = None


# Union type for all external events that can be processed by the simulation
SimulationEvent = Union[InterventionEvent, BehavioralEvent]


class SimulationConfig(BaseModel):
    rng: RNGConfig = RNGConfig()
    skills: SkillsConfig = SkillsConfig()
    misconceptions: List[Misconception] = []
    items: ItemsConfig = ItemsConfig()
    population: PopulationConfig = PopulationConfig()
    learning: LearningConfig = LearningConfig()
    interventions: Dict[str, InterventionSpec] = {}
    missingness: MissingnessConfig = MissingnessConfig()
    output: OutputConfig = OutputConfig()
    helpers: HelperConfig = HelperConfig()
