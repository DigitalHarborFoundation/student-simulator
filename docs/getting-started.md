# Getting Started with Student Simulator

This guide will walk you through the basic concepts and show you how to create a simple simulation.

## Core Concepts

The Student Simulator models how students learn skills over time. Here are the key components:

- **Skill**: A learning objective (e.g., "addition within 20")
- **SkillSpace**: A collection of skills with prerequisite relationships
- **Student**: A learner with skill levels that change through practice and assessment
- **ActivityProvider**: Creates assessments and learning activities
- **Item**: A question or problem that tests specific skills

## Simple Example

Let's create a basic simulation with 4 math skills:

```python
from studentsimulator.student import Student, save_student_profile_to_csv, save_student_activity_to_csv
from studentsimulator.activity_provider import ActivityProvider
from studentsimulator.general import SkillSpace

# Define skills with prerequisites
skills = [
    {
        "name": "number_recognition",
        "description": "Recognize and write numerals 0-20",
    },
    {
        "name": "place_value_ones",
        "description": "Understand place value of ones",
        "prerequisites": {"parent_names": ["number_recognition"]},
    },
    {
        "name": "place_value_tens",
        "description": "Understand place value of tens",
        "prerequisites": {"parent_names": ["place_value_ones"]},
    },
    {
        "name": "addition_no_carry",
        "description": "Add within 20 without regrouping",
        "prerequisites": {"parent_names": ["place_value_tens"]},
    },
]

# Create skill space
skill_space = SkillSpace(skills=skills)

# Create students with different initial skill levels
## By default, skills set to 0
student1 = Student(name="Alice", skill_space=skill_space)
## Setting skills manually
student2 = Student(skill_space=skill_space).set_skill_values({
    "number_recognition": 0.5,
    "place_value_ones": 0.3
})
## Seetting skills randomly, but obeying prerequisite relationships
student3 = Student(skill_space=skill_space).initialize_skill_values(practice_count=[1, 9])

# Create assessments
activity_provider = ActivityProvider()
activity_provider.register_skills(skill_space)

item_pool = activity_provider.construct_item_pool(
    name="basic_math_pool",
    skills=skill_space.skills,
    n_items_per_skill=20,
    difficulty_logit_range=(-2, 2),
    guess_range=(0.1, 0.3),
    slip_range=(0.01, 0.2),
)

test = activity_provider.generate_fixed_form_assessment(
    n_items=20,
    item_pool=item_pool,
    skills=skill_space
)

# Students take the test
for student in [student1, student2, student3]:
    test_result = student.take_test(test, timestamp=1)

# Save results
save_student_profile_to_csv(students=[student1, student2, student3], filename="students.csv")
save_student_activity_to_csv(students=[student1, student2, student3], filename="student_activity.csv")
```

## Key Parameters Explained

### Skill Parameters
- `prerequisites`: Skills that must be learned first
- `probability_of_learning_without_prerequisites`: Chance of learning without prerequisites
- `probability_of_learning_with_prerequisites`: Chance of learning with prerequisites
- `decay_logit`: How quickly the skill is forgotten (0.01 = slow, 0.1 = fast)

### Assessment Parameters
- `n_items_per_skill`: Number of questions per skill in the item pool
- `difficulty_logit_range`: Range of question difficulties
- `guess_range`: Probability of guessing correctly
- `slip_range`: Probability of making careless errors

## Next Steps

- See [Advanced Simulation](advanced-simulation.md) for larger-scale simulations
- Check the API documentation for detailed parameter descriptions
- Explore the generated CSV files to analyze student performance
