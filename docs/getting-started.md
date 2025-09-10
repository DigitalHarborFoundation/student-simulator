# Getting Started with Student Simulator

This guide will walk you through the basic concepts and show you how to create a simple simulation.

## Core Concepts

The Student Simulator models how students learn skills over time. Here are the key components:

- **Skill**: A learning objective (e.g., "addition within 20")
- **SkillSpace**: A collection of skills with prerequisite relationships
- **Student**: A learner with skill levels that change through practice and assessment
- **ActivityProvider**: Creates assessments and learning activities
- **Item**: A question or problem that tests specific skills

## Simple Example: Manual Student Creation

Let's create a basic simulation with 4 math skills and create students manually:

```python
from studentsimulator.student import Student
from studentsimulator.io import save_student_daily_skill_states_to_csv, save_student_events_to_csv
from studentsimulator.activity_provider import ActivityProvider
from studentsimulator.skill import Skill, SkillSpace

# Define skills with prerequisites
skills = [
    Skill(
        name="number_recognition",
        code="CCSS.MATH.K.CC.A.3",
        description="Recognize and write numerals 0-20",
        decay_logit=0.01,
        initial_skill_level_after_learning=0.3,
    ),
    Skill(
        name="place_value_ones",
        code="CCSS.MATH.1.NBT.A.1",
        description="Understand place value of ones",
        prerequisites={"parent_names": ["number_recognition"]},
        decay_logit=0.02,
        initial_skill_level_after_learning=0.35,
    ),
    Skill(
        name="place_value_tens",
        code="CCSS.MATH.1.NBT.A.2",
        description="Understand place value of tens",
        prerequisites={"parent_names": ["place_value_ones"]},
        decay_logit=0.02,
        initial_skill_level_after_learning=0.3,
    ),
    Skill(
        name="addition_no_carry",
        code="CCSS.MATH.1.OA.A.1",
        description="Add within 20 without regrouping",
        prerequisites={"parent_names": ["place_value_tens"]},
        decay_logit=0.03,
        initial_skill_level_after_learning=0.25,
    ),
]

# Create skill space
skill_space = SkillSpace(skills=skills)

# Create students with different initial skill levels
## By default, skills start at minimum level (0.01)
student1 = Student(name="Alice", skill_space=skill_space)

## Setting skills manually
student2 = Student(skill_space=skill_space).set_skill_values({
    "number_recognition": 0.5,
    "place_value_ones": 0.3
})

## Setting skills with more advanced knowledge
student3 = Student(skill_space=skill_space).set_skill_values({
    "number_recognition": 0.8,
    "place_value_ones": 0.6,
    "place_value_tens": 0.4,
    "addition_no_carry": 0.9,
})

## Randomly initialize skills while respecting prerequisites
student4 = Student(skill_space=skill_space).randomly_initialize_skills(practice_count=[1, 9])

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
    discrimination_range=(1.0, 1.0),
)

test = activity_provider.generate_fixed_form_assessment(
    n_items=20,
    item_pool=item_pool,
    skills=skill_space
)

# Students take the test
for student in [student1, student2, student3, student4]:
    test_result = activity_provider.administer_fixed_form_assessment(student, test)

# Save results
save_student_daily_skill_states_to_csv(students=[student1, student2, student3, student4], filename="students_daily_skill_states.csv")
save_student_events_to_csv(students=[student1, student2, student3, student4], filename="student_events.csv")
```

## Group Student Creation

For larger simulations, you can create groups of students with random skill levels:

```python
from studentsimulator.factory import create_random_students

# Create 100 students with random skill levels
students = create_random_students(
    skill_space=skill_space,
    n_students=100,
    practice_count=[5, 20]  # Random practice sessions between 5-20
)

# Take assessments
for student in students:
    test_result = activity_provider.administer_fixed_form_assessment(student, test)

# Save with train/validation/test split for machine learning
save_student_events_to_csv(
    students=students,
    filename="student_events_with_split.csv",
    train_val_test_split=(0.7, 0.15, 0.15),  # 70% train, 15% validation, 15% test
    observation_rate=0.9  # 90% of events are observed
)
```

## Key Parameters Explained

### Skill Parameters
- `prerequisites`: Skills that must be learned first
- `probability_of_learning_without_prerequisites`: Chance of learning without prerequisites
- `probability_of_learning_with_prerequisites`: Chance of learning with prerequisites
- `decay_logit`: How quickly the skill is forgotten (0.01 = slow, 0.1 = fast)
- `initial_skill_level_after_learning`: Skill level after first learning event
- `practice_increment_logit`: How much skill increases with each practice session

### Assessment Parameters
- `n_items_per_skill`: Number of questions per skill in the item pool
- `difficulty_logit_range`: Range of question difficulties
- `guess_range`: Probability of guessing correctly
- `slip_range`: Probability of making careless errors
- `discrimination_range`: How well items discriminate between skill levels

## Next Steps

- See [Advanced Simulation](advanced-simulation.md) for larger-scale simulations with learning dynamics
- Check the API documentation for detailed parameter descriptions
- Explore the generated CSV files to analyze student performance
