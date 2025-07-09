# Student Simulator: Educational Response Simulation Framework

**Student Simulator** is a comprehensive Python framework for simulating realistic student-item interactions across diverse educational paradigms. It enables researchers and developers to generate interpretable, configurable, and reproducible student behavior data for testing educational technologies, adaptive systems, and learning analytics.

## 🎯 Key Features

### Unified Psychometric Modeling
- **Multiple Psychometric Models**: BKT, PFA, IRT, CDM, and Hybrid approaches
- **Parameter-driven switching**: Change models with a single configuration
- **Mathematical rigor**: Models reduce to established psychometric approaches
- **Comparative studies**: Test different learning theories within one framework

### Dynamic Learning Engine
- **Interventions**: Lessons, hints, videos, tutoring with probabilistic effectiveness
- **Practice effects**: Gradual skill improvement through item responses
- **Prerequisites**: CDM-style skill dependencies affect learning probability
- **Learning history**: Complete trace of student progression and skill development

### Rich Event Architecture
- **Event-driven design**: All interactions flow through structured event streams
- **Two-tier events**: Interventions (done TO students) vs Behaviors (done BY students)
- **Contextual metadata**: Rich context tracking for realistic simulations
- **Missing data patterns**: MCAR, skill-dependent, and student-dependent dropout

### Production-Ready Framework
- **Type-safe**: Full Pydantic validation with strong typing
- **Reproducible**: Deterministic random generation with configurable seeds
- **Scalable**: Efficient NumPy backend with optional GPU acceleration
- **Clean architecture**: Separated concerns with StudentHistory, LearningEngine, and AssessmentEngine

## 🚀 Quick Start

### Installation

Development:
```bash
pip install -e '.[dev]'
pytest tests/
```

### Basic Usage

```python
from studentsimulator.general import Skill, SkillSpace
from studentsimulator.student import Student
from studentsimulator.activity_provider import ActivityProvider

# Define skills with prerequisites
skills = [
    Skill(name="counting", prerequisites={"parent_names": []}),
    Skill(name="addition", prerequisites={"parent_names": ["counting"]}),
    Skill(name="multiplication", prerequisites={"parent_names": ["addition"]})
]

skill_space = SkillSpace(skills=skills)

# Create students with different skill levels
student1 = Student(name="alice", skill_space=skill_space)
student2 = Student(name="bob", skill_space=skill_space).set_skill_values({
    "counting": 0.8,
    "addition": 0.6
})

# Create random student with practice-based initialization
student3 = Student(name="charlie", skill_space=skill_space).initialize_skill_values(
    practice_count=[3, 10]  # Random practice between 3-10 sessions
)

# Set up activity provider and assessments
provider = ActivityProvider()
provider.register_skills(skill_space)

item_pool = provider.construct_item_pool(
    name="math_pool",
    skills=skill_space.skills,
    n_items_per_skill=20,
    difficulty_logit_range=(-2, 2),
    guess_range=(0.1, 0.3),
    slip_range=(0.01, 0.2),
    discrimination=1.0
)

assessment = provider.generate_fixed_form_assessment(
    n_items=10,
    item_pool=item_pool,
    skills=skill_space
)

# Students take the assessment
for student in [student1, student2, student3]:
    results = student.take_test(assessment, timestamp=1)
    print(f"{student.name}: {results.percent_correct:.1f}% correct")

# Save results to CSV
from studentsimulator.student import save_student_profile_to_csv, save_student_activity_to_csv

save_student_profile_to_csv([student1, student2, student3], "students.csv")
save_student_activity_to_csv([student1, student2, student3], "activity.csv")
```

## 📊 Recent Improvements

### Consistent Naming Conventions
- **Unified terminology**: All code now uses `prerequisites.parent_names` instead of mixed `parents`/`prerequisites`
- **Backward compatibility**: `skill.parents` property provides seamless migration path
- **Fixed typos**: Corrected `discrimination` parameter naming throughout

### Clean Architecture
- **StudentHistory extraction**: Separated event tracking from core student behavior
- **Single responsibility**: Each class has a clear, focused purpose
- **Better testability**: Components can be tested in isolation

### Enhanced Event System
- **ItemResponseEvent**: Structured events for item interactions with proper typing
- **LearningEvent**: Dedicated events for learning activities
- **Rich metadata**: Events include feedback flags, practice increments, and timestamps

### Robust Data Export
- **CSV headers**: Updated to use `engagement_object_id` and `group_id` for clarity
- **Null safety**: Proper handling of optional fields and missing data
- **Event filtering**: Support for different event types in exports

## 🔬 Research Applications

### Educational Technology Testing
```python
# Test adaptive tutoring system
events = generate_adaptive_sequence(student_model, difficulty_progression)
response_log, _ = sim.run(events)
evaluate_learning_efficiency(response_log)
```

### Learning Analytics Validation
```python
# Generate ground-truth data for algorithm validation
sim.learning(mode="bkt").run(standard_assessment_events)
validate_bkt_parameter_estimation(response_log, true_parameters)
```

### Intervention Effectiveness Studies
```python
# Compare intervention strategies
control_sim = Sim().learning(mode="pfa", learning_probability=0.1)
treatment_sim = Sim().learning(mode="pfa", learning_probability=0.3)

control_results = control_sim.run(intervention_events)
treatment_results = treatment_sim.run(intervention_events)
```

### Missing Data Pattern Analysis
```python
# Simulate realistic dropout patterns
sim.missingness(mode="skill_dependent", skill_rates={"advanced_math": 0.3})
incomplete_data = sim.run(assessment_events)
test_inference_robustness(incomplete_data)
```

## 🏗️ Architecture

### Event-Driven Design
All interactions flow through structured events:

```python
# Intervention event (done TO student)
intervention = {
    "student_id": "alice",
    "time": 10,
    "intervention_type": "video_lesson",
    "context": {"target_skill": "algebra", "duration": 300}
}

# Behavioral event (done BY student)
behavior = {
    "student_id": "alice",
    "time": 15,
    "item_id": "algebra_problem_1",
    "observed": True
}
```

### Three-Layer API

**1. Fluent API** - High-level, chainable interface:
```python
sim = (Sim(seed=42)
       .skills(skill_definitions)
       .items(item_bank)
       .population(student_cohort)
       .learning(mode="hybrid")
       .run(event_stream))
```

**2. Hook API** - Low-level functions for custom workflows:
```python
from simlearn.api import generate_latent, transition, emit_response

state = generate_latent(config)
state = transition(state, event)
response = emit_response(state, item)
```

**3. CLI** - Command-line interface:
```bash
simulate --config experiment.yaml --events student_interactions.yaml --output results/
```

### Data Models

**Student**: Individual learners with skill progression
```python
student.skill_state  # Dict[skill_name, SkillState]
student.history.get_events()  # Complete trace of learning events
student.history.get_assessment_events()  # Assessment-specific events
```

**Skill**: Competencies with prerequisite relationships
```python
skill.prerequisites.parent_names  # List of prerequisite skill names
skill.parents  # Backward compatibility property
skill.practice_gain  # Learning rate parameters
skill.get_cdm_learning_probability(parent_states)  # CDM prerequisite logic
```

**Item**: Assessment items with psychometric parameters
```python
item.skills  # List of required skills
item.g, item.s  # Guess and slip parameters (IRT)
item.practice_effectiveness  # Contribution to skill development
```

## 📈 Advanced Features

### Model Reduction Properties
The unified framework mathematically reduces to standard models:

```python
# Demonstrate BKT equivalence
hybrid_sim = Sim().learning(mode="hybrid", practice_effectiveness=1.0)
bkt_sim = Sim().learning(mode="bkt")
# Both produce equivalent binary learning behavior

# Show PFA characteristics
pfa_sim = Sim().learning(mode="pfa", learning_probability=0.9)
# Produces gradual improvement curves matching PFA theory
```

### Prerequisite Networks
Model complex skill dependencies:

```python
skills = [
    Skill(name="arithmetic", prerequisites={"parent_names": []}),
    Skill(name="algebra", prerequisites={"parent_names": ["arithmetic"]}),
    Skill(name="calculus", prerequisites={"parent_names": ["algebra"]}),
    Skill(name="statistics", prerequisites={"parent_names": ["arithmetic"]}),  # Alternative path
    Skill(name="data_science", prerequisites={"parent_names": ["statistics", "algebra"]})  # Convergent
]
```

### Rich Intervention Types
Multiple intervention modalities:

```python
interventions = {
    "video_lesson": {"effectiveness": 0.7, "duration": 300},
    "worked_example": {"effectiveness": 0.5, "cognitive_load": "medium"},
    "peer_tutoring": {"effectiveness": 0.6, "social_factor": True},
    "adaptive_hint": {"effectiveness": 0.4, "contextual": True}
}
```

## 🧪 Testing and Validation

Student Simulator includes comprehensive test suites validating:

- **Psychometric model equivalence**: Mathematical reduction properties
- **Learning progression**: Realistic skill development patterns
- **Event architecture**: Two-tier intervention/behavioral system
- **Prerequisite logic**: CDM-style dependency handling
- **Statistical properties**: Response distributions match theory

```bash
# Run full test suite
pytest tests/ -v

# Test specific psychometric models
pytest tests/test_psychometric_model_modes.py -v

# Validate CDM prerequisite behavior
pytest tests/test_cdm_mode.py -v
```

## 📚 Documentation

- **[API Reference](docs/)**: Complete function and class documentation
- **[User Guide](docs/user_guide.md)**: Detailed tutorials and examples
- **[Developer Guide](CLAUDE.md)**: Architecture and contribution guidelines
- **[Research Examples](examples/)**: Common research workflow patterns

## 🤝 Contributing

We welcome contributions! Please see our [contribution guidelines](CONTRIBUTING.md) for:

- Code style and testing requirements
- Issue reporting and feature requests
- Development environment setup
- Pull request process

## 📄 License

Student Simulator is released under the [MIT License](LICENSE).

## 🔗 Citation

If you use Student Simulator in your research, please cite:

```bibtex
@software{studentsimulator2024,
  title={Student Simulator: A Unified Framework for Educational Response Simulation},
  author={Your Name and Contributors},
  year={2024},
  url={https://github.com/your-org/student-simulator}
}
```

## 🎪 Examples

Check out the [examples directory](examples/) for:
- **Basic simulation**: Simple BKT-style learning
- **Adaptive testing**: CAT item selection algorithms
- **Intervention studies**: Comparing teaching strategies
- **Missing data**: Realistic dropout pattern simulation
- **Multi-skill assessments**: Complex prerequisite networks

---

**Ready to simulate learning?** Start with our [Quick Start Guide](docs/quickstart.md) or explore the [example notebooks](examples/notebooks/).
