# Advanced Simulation Guide

This guide shows how to create large-scale simulations with complex skill hierarchies, learning dynamics, and visualization, based on educational research and standards.

## Complex Skill Structure with Learning Dynamics

Here's how to create a comprehensive simulation with 12 math skills following Common Core State Standards, including learning lessons and practice over time:

```python
import time
from studentsimulator.activity_provider import ActivityProvider
from studentsimulator.skill import Skill, SkillSpace
from studentsimulator.factory import create_random_students
from studentsimulator.io import save_student_daily_skill_states_to_csv, save_student_events_to_csv
from studentsimulator.analytics import plot_skill_mastery, plot_skill_trajectory

# Define comprehensive skill hierarchy
skills = [
    # Foundation Level - Number Sense (K-1)
    Skill(
        name="number_recognition",
        code="CCSS.MATH.K.CC.A.3",
        description="Recognize and write numerals 0-20",
        decay_logit=0.01,
        probability_of_learning_without_prerequisites=0.8,
        probability_of_learning_with_prerequisites=0.98,
        practice_increment_logit=0.15,
        initial_skill_level_after_learning=0.3,
    ),
    Skill(
        name="counting_sequence",
        code="CCSS.MATH.K.CC.A.1",
        description="Count to 100 by ones and tens",
        prerequisites={"parent_names": ["number_recognition"], "dependence_model": "all"},
        decay_logit=0.01,
        probability_of_learning_without_prerequisites=0.2,
        probability_of_learning_with_prerequisites=0.95,
        practice_increment_logit=0.12,
        initial_skill_level_after_learning=0.4,
    ),

    # Place Value Foundation (1-2)
    Skill(
        name="place_value_ones",
        code="CCSS.MATH.1.NBT.A.1",
        description="Understand place value of ones",
        prerequisites={"parent_names": ["counting_sequence"], "dependence_model": "all"},
        decay_logit=0.02,
        probability_of_learning_without_prerequisites=0.1,
        probability_of_learning_with_prerequisites=0.92,
        practice_increment_logit=0.14,
        initial_skill_level_after_learning=0.35,
    ),
    Skill(
        name="place_value_tens",
        code="CCSS.MATH.1.NBT.A.2",
        description="Understand place value of tens",
        prerequisites={"parent_names": ["place_value_ones"], "dependence_model": "all"},
        decay_logit=0.02,
        probability_of_learning_without_prerequisites=0.05,
        probability_of_learning_with_prerequisites=0.88,
        practice_increment_logit=0.13,
        initial_skill_level_after_learning=0.3,
    ),

    # Basic Operations (1-2)
    Skill(
        name="addition_within_20",
        code="CCSS.MATH.1.OA.A.1",
        description="Add within 20 using strategies",
        prerequisites={"parent_names": ["counting_sequence", "place_value_ones"], "dependence_model": "all"},
        decay_logit=0.03,
        probability_of_learning_without_prerequisites=0.05,
        probability_of_learning_with_prerequisites=0.85,
        practice_increment_logit=0.16,
        initial_skill_level_after_learning=0.25,
    ),
    Skill(
        name="subtraction_within_20",
        code="CCSS.MATH.1.OA.A.2",
        description="Subtract within 20 using strategies",
        prerequisites={"parent_names": ["addition_within_20"], "dependence_model": "all"},
        decay_logit=0.03,
        probability_of_learning_without_prerequisites=0.03,
        probability_of_learning_with_prerequisites=0.82,
        practice_increment_logit=0.15,
        initial_skill_level_after_learning=0.2,
    ),

    # Multi-digit Operations (2-3)
    Skill(
        name="addition_with_regrouping",
        code="CCSS.MATH.2.NBT.B.6",
        description="Add within 100 using strategies based on place value",
        prerequisites={"parent_names": ["addition_within_20", "place_value_tens"], "dependence_model": "all"},
        decay_logit=0.04,
        probability_of_learning_without_prerequisites=0.02,
        probability_of_learning_with_prerequisites=0.78,
        practice_increment_logit=0.14,
        initial_skill_level_after_learning=0.2,
    ),
    Skill(
        name="subtraction_with_regrouping",
        code="CCSS.MATH.2.NBT.B.7",
        description="Subtract within 100 using strategies based on place value",
        prerequisites={"parent_names": ["subtraction_within_20", "addition_with_regrouping"], "dependence_model": "all"},
        decay_logit=0.04,
        probability_of_learning_without_prerequisites=0.01,
        probability_of_learning_with_prerequisites=0.75,
        practice_increment_logit=0.13,
        initial_skill_level_after_learning=0.15,
    ),

    # Multiplication Foundation (2-3)
    Skill(
        name="multiplication_concept",
        code="CCSS.MATH.2.OA.C.4",
        description="Understand multiplication as repeated addition",
        prerequisites={"parent_names": ["addition_within_20"], "dependence_model": "all"},
        decay_logit=0.05,
        probability_of_learning_without_prerequisites=0.1,
        probability_of_learning_with_prerequisites=0.8,
        practice_increment_logit=0.12,
        initial_skill_level_after_learning=0.25,
    ),
    Skill(
        name="multiplication_facts",
        code="CCSS.MATH.3.OA.C.7",
        description="Multiply and divide within 100 using strategies",
        prerequisites={"parent_names": ["multiplication_concept", "addition_with_regrouping"], "dependence_model": "all"},
        decay_logit=0.06,
        probability_of_learning_without_prerequisites=0.01,
        probability_of_learning_with_prerequisites=0.7,
        practice_increment_logit=0.11,
        initial_skill_level_after_learning=0.1,
    ),

    # Division and Fractions (3-4)
    Skill(
        name="division_concept",
        code="CCSS.MATH.3.OA.A.2",
        description="Understand division as sharing and grouping",
        prerequisites={"parent_names": ["multiplication_facts"], "dependence_model": "all"},
        decay_logit=0.06,
        probability_of_learning_without_prerequisites=0.02,
        probability_of_learning_with_prerequisites=0.65,
        practice_increment_logit=0.1,
        initial_skill_level_after_learning=0.15,
    ),
    Skill(
        name="fraction_concept",
        code="CCSS.MATH.3.NF.A.1",
        description="Understand fractions as parts of a whole",
        prerequisites={"parent_names": ["division_concept"], "dependence_model": "all"},
        decay_logit=0.07,
        probability_of_learning_without_prerequisites=0.05,
        probability_of_learning_with_prerequisites=0.6,
        practice_increment_logit=0.09,
        initial_skill_level_after_learning=0.1,
    ),

    # Advanced Operations (4-5)
    Skill(
        name="multi_digit_multiplication",
        code="CCSS.MATH.4.NBT.B.5",
        description="Multiply a whole number of up to four digits by a one-digit number",
        prerequisites={"parent_names": ["multiplication_facts", "addition_with_regrouping"], "dependence_model": "all"},
        decay_logit=0.08,
        probability_of_learning_without_prerequisites=0.01,
        probability_of_learning_with_prerequisites=0.55,
        practice_increment_logit=0.08,
        initial_skill_level_after_learning=0.05,
    ),
    Skill(
        name="decimal_operations",
        code="CCSS.MATH.5.NBT.B.7",
        description="Add, subtract, multiply, and divide decimals to hundredths",
        prerequisites={"parent_names": ["fraction_concept", "multi_digit_multiplication"], "dependence_model": "all"},
        decay_logit=0.09,
        probability_of_learning_without_prerequisites=0.005,
        probability_of_learning_with_prerequisites=0.5,
        practice_increment_logit=0.07,
        initial_skill_level_after_learning=0.05,
    ),
]

# Create skill space
skill_space = SkillSpace(skills=skills)

# Generate 100 students with random skill levels
students = create_random_students(
    skill_space=skill_space,
    n_students=100,
    practice_count=[10, 50]  # Random practice sessions between 10-50
)

# Visualize skill mastery
plot_skill_mastery(skill_space=skill_space, students=students, filename="skill_mastery.png")

# Create comprehensive item pool
activity_provider = ActivityProvider()
activity_provider.register_skills(skill_space)

item_pool = activity_provider.construct_item_pool(
    name="comprehensive_math_item_pool",
    skills=skill_space.skills,
    n_items_per_skill=25,
    difficulty_logit_range=(-2.5, 2.5),
    guess_range=(0.05, 0.25),
    slip_range=(0.01, 0.15),
    discrimination_range=(0.8, 1.5),
)

# Create tiered assessments
foundation_test = activity_provider.generate_fixed_form_assessment(
    n_items=15,
    item_pool=item_pool,
    skills=[skill_space.get_skill("number_recognition"),
            skill_space.get_skill("counting_sequence"),
            skill_space.get_skill("place_value_ones"),
            skill_space.get_skill("place_value_tens")]
)

intermediate_test = activity_provider.generate_fixed_form_assessment(
    n_items=20,
    item_pool=item_pool,
    skills=[skill_space.get_skill("addition_within_20"),
            skill_space.get_skill("subtraction_within_20"),
            skill_space.get_skill("addition_with_regrouping"),
            skill_space.get_skill("subtraction_with_regrouping")]
)

advanced_test = activity_provider.generate_fixed_form_assessment(
    n_items=25,
    item_pool=item_pool,
    skills=[skill_space.get_skill("multiplication_concept"),
            skill_space.get_skill("multiplication_facts"),
            skill_space.get_skill("division_concept"),
            skill_space.get_skill("fraction_concept"),
            skill_space.get_skill("multi_digit_multiplication"),
            skill_space.get_skill("decimal_operations")]
)

# Simulate learning progression over time
for student in students:
    foundation_result = activity_provider.administer_fixed_form_assessment(student, foundation_test)
    intermediate_result = activity_provider.administer_fixed_form_assessment(student, intermediate_test)
    advanced_result = activity_provider.administer_fixed_form_assessment(student, advanced_test)

# Save results
save_student_daily_skill_states_to_csv(students=students, filename="students_daily_skill_states.csv")

# Save with train/validation/test split for machine learning
save_student_events_to_csv(
    students=students,
    filename="student_events.csv",
    train_val_test_split=(0.7, 0.15, 0.15),  # 70% train, 15% validation, 15% test
    observation_rate=0.9  # 90% of events are observed
)
```

## Learning Dynamics with Lessons and Practice

Here's an example showing how to simulate learning lessons and practice over time:

```python
from studentsimulator.student import Student

# Create a single student for detailed tracking
student = Student(name="learning_student", skill_space=skill_space)

# Get skill objects for learning and practice
basic_math = skill_space.get_skill("number_recognition")
advanced_math = skill_space.get_skill("addition_within_20")
complex_math = skill_space.get_skill("multiplication_concept")

# Week 1: Learn basic math
print("Week 1: Learning basic math")
student.learn(basic_math)

# Practice basic math for 5 days
for day in range(1, 6):
    print(f"Day {day}: Practicing basic math")
    for session in range(3):  # 3 practice sessions per day
        student.practice(basic_math)
    student.wait(days=1)  # Advance time and apply forgetting

# Week 2: Learn advanced math
print("Week 2: Learning advanced math")
student.learn(advanced_math)

# Practice both skills for 5 days
for day in range(6, 11):
    print(f"Day {day}: Practicing both skills")
    for session in range(2):  # 2 practice sessions per day
        student.practice(basic_math)
        student.practice(advanced_math)
    student.wait(days=1)

# Week 3: Learn complex math
print("Week 3: Learning complex math")
student.learn(complex_math)

# Practice all skills for 5 days
for day in range(11, 16):
    print(f"Day {day}: Practicing all skills")
    for session in range(2):
        student.practice(basic_math)
        student.practice(advanced_math)
        student.practice(complex_math)
    student.wait(days=1)

# Let skills decay for 30 days
print("Allowing skills to decay for 30 days")
student.wait(days=30)

# Plot the learning trajectory
plot_skill_trajectory(
    student,
    filename="learning_trajectory.png"
)
```

## Key Features of This Simulation

### 1. **Educational Research-Based Design**
- Skills follow Common Core State Standards progression
- Prerequisites reflect cognitive development research
- Learning probabilities decrease with skill complexity

### 2. **Realistic Learning Parameters**
- **Decay rates** increase with skill complexity (0.01 → 0.09)
- **Learning probabilities** decrease without prerequisites (0.8 → 0.005)
- **Practice increments** decrease with complexity (0.15 → 0.07)
- **Initial skill levels** decrease with complexity (0.3 → 0.05)

### 3. **Tiered Assessment Strategy**
- **Foundation test**: Basic number sense and place value
- **Intermediate test**: Basic operations with regrouping
- **Advanced test**: Multiplication, division, fractions, decimals

### 4. **Visualization**
- `plot_skill_mastery()` shows skill dependencies and student counts
- `plot_skill_trajectory()` shows individual student learning over time
- Node size = number of students with the skill
- Edge width = number of students with both connected skills

## Analysis Opportunities

The generated data enables analysis of:
- **Learning progression**: How students advance through skill levels
- **Prerequisite effectiveness**: Impact of prerequisites on learning
- **Assessment validity**: How well tests measure intended skills
- **Student heterogeneity**: Distribution of skill mastery across population
- **Machine learning workflows**: Train/validation/test splits for model development

## Scaling Considerations

- **100 students**: Good for statistical analysis, manageable computation
- **12 skills**: Complex enough to show learning patterns, not overwhelming
- **3 assessment points**: Captures progression without excessive data
- **25 items per skill**: Sufficient for reliable measurement

## Machine Learning Integration

The CSV export supports machine learning workflows with optional train/validation/test splits:

```python
# Standard export (no split)
save_student_events_to_csv(students=students, filename="all_data.csv")

# Export with train/validation/test split
save_student_events_to_csv(
    students=students,
    filename="ml_ready_data.csv",
    train_val_test_split=(0.7, 0.15, 0.15),  # 70% train, 15% validation, 15% test
    observation_rate=0.95  # 95% of events are observed
)

# Export with train/test only
save_student_events_to_csv(
    students=students,
    filename="train_test_data.csv",
    train_val_test_split=(0.8, 0.0, 0.2),  # 80% train, 20% test
    observation_rate=0.8  # 80% of events are observed
)

# Export with missing data simulation
save_student_events_to_csv(
    students=students,
    filename="missing_data.csv",
    observation_rate=0.7  # 70% of events are observed (simulates dropout)
)
```

**CSV Output Options:**

**Standard Output:**
- `student_id, day, event_type, skill_id, item_id, score, prob_correct`

**With Train/Validation/Test Split:**
- `student_id, day, event_type, skill_id, item_id, score, prob_correct, train_val_test`
- `train_val_test` values: 0=train, 1=validation, 2=test
- All events for a student get the same split value

**With Observation Rate:**
- `student_id, day, event_type, skill_id, item_id, score, prob_correct, observed`
- `observed` values: 1=observed, 0=missing/unobserved
- Randomly distributed across all events

**With Both Features:**
- `student_id, day, event_type, skill_id, item_id, score, prob_correct, train_val_test, observed`
- Combines both train/val/test splits and missing data simulation
