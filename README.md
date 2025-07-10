# Student Simulator

A Python framework for simulating student learning and assessment interactions. Generate interpretable, configurable student behavior data for educational research, adaptive systems, and learning analytics.

## 🚀 Quick Start

```bash
pip install -e .
```

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

# Create students
student = Student(name="alice", skill_space=skill_space)
student2 = Student(skill_space=skill_space).initialize_skill_values(practice_count=[3, 10])

# Create assessment
provider = ActivityProvider()
provider.register_skills(skill_space)
item_pool = provider.construct_item_pool(
    name="math_pool", skills=skill_space.skills, n_items_per_skill=20
)
assessment = provider.generate_fixed_form_assessment(n_items=10, item_pool=item_pool, skills=skill_space)

# Students take assessment
result = student.take_test(assessment, timestamp=1)
print(f"{student.name}: {result.percent_correct:.1f}% correct")

# Save results
from studentsimulator.student import save_student_profile_to_csv, save_student_activity_to_csv
save_student_profile_to_csv([student, student2], "students.csv")
save_student_activity_to_csv([student, student2], "activity.csv")
```

## 📚 Documentation

- **[Getting Started](docs/getting-started.md)** - Basic concepts and simple examples
- **[Advanced Simulation](docs/advanced-simulation.md)** - Large-scale simulations with complex skill hierarchies
- **[API Reference](docs/)** - Complete documentation

## 🎯 Key Features

- **Skill-based learning** with prerequisite relationships
- **Multiple psychometric models** (BKT, PFA, IRT, CDM)
- **Realistic learning parameters** based on educational research
- **Event-driven architecture** for rich interaction tracking
- **Visualization tools** for skill dependencies and mastery
- **CSV export** for analysis in other tools

## 🔬 Use Cases

- **Educational research** - Study learning progression patterns
- **Assessment design** - Test item validity and difficulty
- **Adaptive systems** - Generate training data for ML models
- **Curriculum planning** - Understand skill dependencies

## 🧪 Testing

```bash
pytest tests/
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
