# SimLearn: Student Response Simulation Framework

**SimLearn** is a comprehensive Python framework for simulating realistic student-item interactions across diverse educational paradigms. It enables researchers and developers to generate interpretable, configurable, and reproducible student behavior data for testing educational technologies, adaptive systems, and learning analytics.

## 🎯 Key Features

### Unified Psychometric Modeling
- **5 Psychometric Models**: BKT, PFA, IRT, CDM, and Hybrid
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
- **Three APIs**: Fluent (high-level), Hook (low-level), CLI (command-line)
- **Type-safe**: Full Pydantic validation with strong typing
- **Reproducible**: Deterministic random generation with configurable seeds
- **Scalable**: Efficient NumPy backend with optional GPU acceleration

## 🚀 Quick Start

### Installation

Development:
```bash
pip install -e '.[dev]'
pytest tests/
```



```bash
# Install from source
git clone https://github.com/your-org/simlearn.git
cd simlearn
pip install -e .

# Or install from PyPI (when available)
pip install simlearn
```

### Basic Usage

```python
from simlearn import Sim

# Define skills with prerequisites
skills = [
    {"id": "counting", "parents": []},
    {"id": "addition", "parents": ["counting"]},
    {"id": "multiplication", "parents": ["addition"]}
]

# Create items aligned to skills
items = [
    {"id": "count_objects", "skills": [{"id": "counting"}], "g": 0.3, "s": 0.05},
    {"id": "simple_addition", "skills": [{"id": "addition"}], "g": 0.25, "s": 0.05},
    {"id": "times_tables", "skills": [{"id": "multiplication"}], "g": 0.2, "s": 0.05}
]

# Create students
students = [{"id": "student_1"}, {"id": "student_2"}]

# Configure simulation
sim = (Sim(seed=42)
       .skills(skills)
       .items(items)
       .population(students)
       .learning(mode="cdm"))  # CDM mode with prerequisites

# Define learning events
events = [
    # Learning intervention
    {
        "student_id": "student_1",
        "time": 1,
        "item_id": "count_objects",
        "observed": True,
        "intervention_type": "skill_boost",
        "context": {"target_skill": "counting", "boost": 0.8}
    },
    # Practice response
    {
        "student_id": "student_1",
        "time": 2,
        "item_id": "count_objects",
        "observed": True
    }
]

# Run simulation
response_log, latent_state = sim.run(events)
print(response_log)
```

## 📊 Psychometric Models

SimLearn supports five major psychometric modeling paradigms:

### 1. Bayesian Knowledge Tracing (BKT)
- **Binary learning states**: Skills are either learned or not learned
- **Probabilistic acquisition**: Learning occurs with some probability per practice
- **Sudden mastery**: Performance jumps from guess rate to mastery level

```python
sim.learning(mode="bkt")
# Characteristics: learning_probability=0.3, practice_effectiveness=1.0 (binary jump)
```

### 2. Performance Factor Analysis (PFA)
- **Gradual improvement**: Skills improve incrementally with practice
- **Practice-driven**: More practice leads to better performance
- **Smooth learning curves**: Continuous proficiency growth

```python
sim.learning(mode="pfa")
# Characteristics: learning_probability=0.9, practice_effectiveness=0.15 (gradual)
```

### 3. Item Response Theory (IRT)
- **Static ability**: No learning occurs, fixed student ability
- **Consistent performance**: Responses based solely on initial proficiency
- **No practice effects**: Performance remains stable over time

```python
sim.learning(mode="irt")
# Characteristics: learning_probability=0.0, practice_effectiveness=0.0 (no learning)
```

### 4. Cognitive Diagnostic Models (CDM)
- **Prerequisite dependencies**: Learning probability depends on prior skills
- **Binary skill states**: Skills are mastered or not mastered
- **Hierarchical learning**: Must learn prerequisites before advanced skills

```python
sim.learning(mode="cdm")
# Characteristics: prerequisite-aware learning, binary response model
```

### 5. Hybrid Model
- **All features enabled**: Combines aspects of all psychometric traditions
- **Flexible parameters**: Fine-tune to match specific learning scenarios
- **Research comparisons**: Baseline for comparing other models

```python
sim.learning(mode="hybrid")
# Characteristics: Full model with learning, practice, and skill progression
```

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
student.current_skills  # Dict[skill_id, SkillState]
student.learning_history  # Complete trace of learning events
student.misconceptions  # Active misconceptions affecting responses
```

**Skill**: Competencies with prerequisite relationships
```python
skill.parents  # List of prerequisite skill IDs
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
    {"id": "arithmetic", "parents": []},
    {"id": "algebra", "parents": ["arithmetic"]},
    {"id": "calculus", "parents": ["algebra"]},
    {"id": "statistics", "parents": ["arithmetic"]},  # Alternative path
    {"id": "data_science", "parents": ["statistics", "algebra"]}  # Convergent
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

SimLearn includes comprehensive test suites validating:

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

SimLearn is released under the [MIT License](LICENSE).

## 🔗 Citation

If you use SimLearn in your research, please cite:

```bibtex
@software{simlearn2024,
  title={SimLearn: A Unified Framework for Educational Response Simulation},
  author={Your Name and Contributors},
  year={2024},
  url={https://github.com/your-org/simlearn}
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
