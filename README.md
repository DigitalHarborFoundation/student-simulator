# Student Simulator

A Python framework for simulating student learning and assessment interactions. Generate interpretable, configurable student behavior data for educational research, adaptive systems, and learning analytics.

## 🚀 Quick Start

```bash
pip install -e .
```

```python
from studentsimulator.skill import Skill, SkillSpace
from studentsimulator.student import Student
from studentsimulator.activity_provider import ActivityProvider

# Define skills with prerequisites and learning parameters
skills = [
    Skill(
        name="counting",
        prerequisites={"parent_names": []},
        practice_increment_logit=0.1,
        initial_skill_level_after_learning=0.5
    ),
    Skill(
        name="addition",
        prerequisites={"parent_names": ["counting"]},
    ),
    Skill(
        name="multiplication",
        prerequisites={"parent_names": ["addition"]},
    )
]

skill_space = SkillSpace(skills=skills)

# Create students
student = Student(name="alice", skill_space=skill_space)
student2 = Student(skill_space=skill_space).randomly_initialize_skills(practice_count=[3, 10])

# Create assessment
provider = ActivityProvider(name='MyActivityProvider')
provider.register_skills(skill_space)
item_pool = provider.construct_item_pool(
    name="math_pool", skills=skill_space.skills, n_items_per_skill=20
)
assessment = provider.generate_fixed_form_assessment(n_items=10, item_pool=item_pool, skills=skill_space)

# Students take assessment
result = student.take_test(assessment, timestamp=1)
print(f"{student.name}: {result.percent_correct:.1f}% correct")

# Save results
from studentsimulator.io import save_student_daily_skill_states_to_csv, save_student_events_to_csv
save_student_daily_skill_states_to_csv([student, student2], "students.csv")
# Just save student events as seen by a particular activity provider.
save_student_events_to_csv([student, student2], "activity.csv", activity_provider_name='MyActivityProvider')
```

## 📚 Documentation

- **[Getting Started](docs/getting-started.md)** - Basic concepts and simple examples
- **[Advanced Simulation](docs/advanced-simulation.md)** - Large-scale simulations with complex skill hierarchies
- **[API Reference](docs/)** - Complete documentation

> **Note**: The API has been updated to support skill transfer effects and a dual-history system. Import `Skill` and `SkillSpace` from `studentsimulator.skill` instead of `studentsimulator.general`.

## 🎯 Key Features

- **Skill-based learning** with prerequisite relationships and skill transfer effects
- **Dual-history system** for efficient event tracking and daily skill snapshots
- **Realistic learning parameters** based on educational research
- **Event-driven architecture** for rich interaction tracking
- **Visualization tools** for skill dependencies and mastery
- **CSV export** for analysis in other tools
- **Skill transfer modeling** with diminishing returns to ancestor skills

## 🔬 Use Cases

- **Educational research** - Study learning progression patterns and skill transfer effects
- **Assessment design** - Test item validity and difficulty
- **Adaptive systems** - Generate training data for ML models
- **Curriculum planning** - Understand skill dependencies and transfer relationships

## 🧠 Skill Transfer

The simulator models skill transfer effects where practicing a skill also benefits its prerequisite skills with diminishing returns. This is based on educational research showing that learning in one domain can benefit related domains, particularly when there are hierarchical relationships.

```python
# Practice multiplication (benefits multiplication, addition, and counting)
student.practice(skill_space.get_skill("multiplication"))

# Check that all skills improved with diminishing effects
print(f"Multiplication: {student.skills['multiplication'].skill_level}")
print(f"Addition: {student.skills['addition'].skill_level}")
print(f"Counting: {student.skills['counting'].skill_level}")
```

## 🧪 Testing

```bash
pytest tests/
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
