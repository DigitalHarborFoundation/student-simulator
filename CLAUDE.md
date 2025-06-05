# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SimLearn** is a Student Response Simulation Framework for educational data research. It simulates realistic student-item interactions across diverse educational paradigms including IRT, CDM/BKT, learning curves, misconceptions, and adaptive testing.

## Core Architecture

### Event-Driven Design
- **Principle**: All interaction = event stream in → event stream out
- **Events** contain: `student_id`, `time`, `item_id`, `observed`, `intervention_type`, `context`
- **Processing**: Events flow through `transition()` → `emit_response()` → learning history

### Three-Layer API Structure
1. **Fluent API** (`simlearn.api.Sim`): High-level chainable interface
   ```python
   Sim(seed=42).skills(skills).items(items).population(students).run(events)
   ```

2. **Hook API** (`generate_latent`, `transition`, `emit_response`): Low-level functions for custom workflows

3. **CLI** (`simulate --config config.yaml --events events.yaml`): Command-line interface

### Core Data Models (Pydantic-based)

**Student Model**: Individual learners with complete history tracking
- `current_skills`: Dict of skill states with levels and practice counts
- `learning_history`: Complete trace of responses, interventions, skill changes
- `misconceptions`: Active misconceptions affecting responses

**Skill Model**: Competencies with prerequisite relationships
- `parents`: List of prerequisite skill IDs (forms DAG)
- `code`: External standard alignment (e.g., "CCSS.MATH.5.NBT.A.1")
- `practice_gain`, `decay`: Learning curve parameters

**Item Model**: Assessment items with skill alignments
- `skills`: Direct list of Skill objects (strong typing)
- IRT parameters: `g` (guess), `s` (slip), `a` (discrimination), `b` (difficulty)

### Psychometric Model Support
The framework implements a **unified hybrid model** that can emulate multiple psychometric traditions through parameter tuning:

**Supported Models:**
- **BKT (Bayesian Knowledge Tracing)**: Binary learning states, probabilistic skill acquisition
- **PFA (Performance Factor Analysis)**: Gradual improvement through practice
- **IRT (Item Response Theory)**: Static ability-based responses, no learning
- **CDM (Cognitive Diagnostic Models)**: Binary skills with prerequisite relationships
- **Hybrid**: Full model combining all psychometric features

**Model Switching:**
```python
sim.learning(mode="bkt")    # Binary states, probabilistic learning
sim.learning(mode="pfa")    # Gradual practice-driven improvement
sim.learning(mode="irt")    # No learning, fixed ability
sim.learning(mode="cdm")    # Prerequisites determine learning probability
sim.learning(mode="hybrid") # All features enabled
```

**Three Proficiency States (Hybrid Model):**
1. **No Learning**: P(correct) = guess parameter (g)
2. **Baseline**: P(correct) = baseline proficiency, improves with practice
3. **Mastery**: P(correct) = 1 - slip parameter (maximum proficiency)

**Learning Progression:**
- **Intervention** → Probabilistic skill acquisition (learning_probability) → Baseline proficiency
- **Practice** → Gradual improvement (practice_effectiveness) → Mastery (capped at max)
- **Prerequisites** (CDM mode): Learning probability depends on parent skill mastery

**Model Reduction Properties:**
- **Reduces to BKT** when: `practice_effectiveness ≈ 1.0`, binary transitions
- **Reduces to PFA** when: `learning_probability ≈ 1.0`, gradual practice gains
- **Reduces to IRT** when: `learning_probability = 0.0`, no practice effects
- **CDM behavior** when: prerequisite-aware learning, binary response model

## Development Commands

### Environment Setup
```bash
# Always use virtual environment
source .venv/bin/activate

# Install package and dependencies
pip install -e .
pip install -e ".[dev]"  # for development tools
```

### Testing
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_learning_journey.py -v

# Run single test with output
python -m pytest tests/test_learning_journey.py::test_student_learning_journey_with_prerequisites -v -s

# Run learning journey demo directly
python tests/test_learning_journey.py
```

### Linting and Type Checking
```bash
# Type checking (strict mode enabled)
mypy src/

# Code formatting
black src/ tests/
isort src/ tests/
ruff src/ tests/
```

### CLI Usage
```bash
# Run simulation
simulate --config examples/basic.yaml --events events.yaml

# With overrides
simulate --config examples/basic.yaml --events events.yaml --override learning.global_kappa=0.4
```

## Package Management

**Critical**: Never use `pip install package` directly. Always:
1. Add dependency to `pyproject.toml`
2. Run `pip install -e .` to install from requirements

This maintains reproducible dependency management.

## Key Implementation Notes

### Hybrid Learning Model Implementation
**SkillState** tracks three proficiency states with automatic P(correct) calculation:
```python
skill_state.get_probability_correct(item_guess, item_slip)
# Returns: guess | interpolated_progress | (1-slip)
```

**Student Methods** for learning progression:
- `apply_intervention()`: Probabilistic skill acquisition
- `practice_skill()`: Incremental proficiency improvement
- Complete learning history tracking with event types

### TDD Approach
This codebase was built using Test-Driven Development. Key test files:
- `test_hybrid_learning_model.py`: Complete BKT/CDM/IRT model validation
- `test_psychometric_model_modes.py`: All 5 psychometric model modes with behavioral validation
- `test_cdm_mode.py`: Dynamic CDM with prerequisite-aware learning probability
- `test_bkt_reduction.py`: Mathematical equivalence demonstration (hybrid → BKT)
- `test_event_architecture.py`: Two-tier event system (interventions vs behaviors)
- `test_learning_journey.py`: End-to-end learning progression scenarios

### Learning Simulation Flow
1. `generate_latent()`: Initialize students with "no_learning" skill states
2. For each event: `transition()` applies interventions (probabilistic) and practice effects
3. `emit_response()` uses hybrid model: skill_state.get_probability_correct()
4. Complete learning history automatically recorded in Student objects

### Event Architecture
**Two Event Types:**
- **InterventionEvent**: Things done TO students (lessons, videos, hints)
- **BehavioralEvent**: Things students DO (responses, interactions) with rich feedback

### Configuration Structure
- **Skills**: Prerequisite DAG, standard alignments, learning parameters
- **Items**: Direct Skill objects, guess/slip parameters, practice_effectiveness
- **Students**: Individual learners with proficiency states and learning history
- **Events**: Drive learning progression with interventions and practice
