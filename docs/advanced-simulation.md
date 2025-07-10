# Advanced Simulation Guide

This guide shows how to create large-scale simulations with complex skill hierarchies, based on educational research and standards.

## Large-Scale Math Simulation

Here's how to create a comprehensive simulation with 14 math skills following Common Core State Standards:

```python
from studentsimulator.student import save_student_activity_to_csv, save_student_profile_to_csv
from studentsimulator.activity_provider import ActivityProvider
from studentsimulator.general import Skill, SkillSpace
from studentsimulator.factory import create_random_students

# Define comprehensive skill hierarchy
skills = [
    # Foundation Level (K-1)
    {
        "name": "number_recognition",
        "code": "CCSS.MATH.K.CC.A.3",
        "description": "Recognize and write numerals 0-20",
        "decay_logit": 0.01,
        "probability_of_learning_without_prerequisites": 0.8,
        "probability_of_learning_with_prerequisites": 0.95,
        "practice_increment_logit": 0.15,
        "initial_skill_level": 0.3,
    },
    {
        "name": "counting_sequence",
        "code": "CCSS.MATH.K.CC.A.1",
        "description": "Count to 100 by ones and tens",
        "prerequisites": {"parent_names": ["number_recognition"]},
        "decay_logit": 0.01,
        "probability_of_learning_without_prerequisites": 0.2,
        "probability_of_learning_with_prerequisites": 0.9,
        "practice_increment_logit": 0.12,
        "initial_skill_level": 0.4,
    },

    # Place Value (1-2)
    {
        "name": "place_value_ones",
        "code": "CCSS.MATH.1.NBT.A.1",
        "description": "Understand place value of ones",
        "prerequisites": {"parent_names": ["counting_sequence"]},
        "decay_logit": 0.02,
        "probability_of_learning_without_prerequisites": 0.1,
        "probability_of_learning_with_prerequisites": 0.85,
        "practice_increment_logit": 0.14,
        "initial_skill_level": 0.35,
    },
    {
        "name": "place_value_tens",
        "code": "CCSS.MATH.1.NBT.A.2",
        "description": "Understand place value of tens",
        "prerequisites": {"parent_names": ["place_value_ones"]},
        "decay_logit": 0.02,
        "probability_of_learning_without_prerequisites": 0.05,
        "probability_of_learning_with_prerequisites": 0.8,
        "practice_increment_logit": 0.13,
        "initial_skill_level": 0.3,
    },

    # Basic Operations (1-2)
    {
        "name": "addition_within_20",
        "code": "CCSS.MATH.1.OA.A.1",
        "description": "Add within 20 using strategies",
        "prerequisites": {"parent_names": ["counting_sequence", "place_value_ones"]},
        "decay_logit": 0.03,
        "probability_of_learning_without_prerequisites": 0.05,
        "probability_of_learning_with_prerequisites": 0.75,
        "practice_increment_logit": 0.16,
        "initial_skill_level": 0.25,
    },
    {
        "name": "subtraction_within_20",
        "code": "CCSS.MATH.1.OA.A.2",
        "description": "Subtract within 20 using strategies",
        "prerequisites": {"parent_names": ["addition_within_20"]},
        "decay_logit": 0.03,
        "probability_of_learning_without_prerequisites": 0.03,
        "probability_of_learning_with_prerequisites": 0.7,
        "practice_increment_logit": 0.15,
        "initial_skill_level": 0.2,
    },

    # Multi-digit Operations (2-3)
    {
        "name": "addition_with_regrouping",
        "code": "CCSS.MATH.2.NBT.B.6",
        "description": "Add within 100 using strategies based on place value",
        "prerequisites": {"parent_names": ["addition_within_20", "place_value_tens"]},
        "decay_logit": 0.04,
        "probability_of_learning_without_prerequisites": 0.02,
        "probability_of_learning_with_prerequisites": 0.65,
        "practice_increment_logit": 0.14,
        "initial_skill_level": 0.2,
    },
    {
        "name": "subtraction_with_regrouping",
        "code": "CCSS.MATH.2.NBT.B.7",
        "description": "Subtract within 100 using strategies based on place value",
        "prerequisites": {"parent_names": ["subtraction_within_20", "addition_with_regrouping"]},
        "decay_logit": 0.04,
        "probability_of_learning_without_prerequisites": 0.01,
        "probability_of_learning_with_prerequisites": 0.6,
        "practice_increment_logit": 0.13,
        "initial_skill_level": 0.15,
    },

    # Multiplication Foundation (2-3)
    {
        "name": "multiplication_concept",
        "code": "CCSS.MATH.2.OA.C.4",
        "description": "Understand multiplication as repeated addition",
        "prerequisites": {"parent_names": ["addition_within_20"]},
        "decay_logit": 0.05,
        "probability_of_learning_without_prerequisites": 0.1,
        "probability_of_learning_with_prerequisites": 0.7,
        "practice_increment_logit": 0.12,
        "initial_skill_level": 0.25,
    },
    {
        "name": "multiplication_facts",
        "code": "CCSS.MATH.3.OA.C.7",
        "description": "Multiply and divide within 100 using strategies",
        "prerequisites": {"parent_names": ["multiplication_concept", "addition_with_regrouping"]},
        "decay_logit": 0.06,
        "probability_of_learning_without_prerequisites": 0.01,
        "probability_of_learning_with_prerequisites": 0.55,
        "practice_increment_logit": 0.11,
        "initial_skill_level": 0.1,
    },

    # Division and Fractions (3-4)
    {
        "name": "division_concept",
        "code": "CCSS.MATH.3.OA.A.2",
        "description": "Understand division as sharing and grouping",
        "prerequisites": {"parent_names": ["multiplication_facts"]},
        "decay_logit": 0.06,
        "probability_of_learning_without_prerequisites": 0.02,
        "probability_of_learning_with_prerequisites": 0.5,
        "practice_increment_logit": 0.1,
        "initial_skill_level": 0.15,
    },
    {
        "name": "fraction_concept",
        "code": "CCSS.MATH.3.NF.A.1",
        "description": "Understand fractions as parts of a whole",
        "prerequisites": {"parent_names": ["division_concept"]},
        "decay_logit": 0.07,
        "probability_of_learning_without_prerequisites": 0.05,
        "probability_of_learning_with_prerequisites": 0.45,
        "practice_increment_logit": 0.09,
        "initial_skill_level": 0.1,
    },

    # Advanced Operations (4-5)
    {
        "name": "multi_digit_multiplication",
        "code": "CCSS.MATH.4.NBT.B.5",
        "description": "Multiply a whole number of up to four digits by a one-digit number",
        "prerequisites": {"parent_names": ["multiplication_facts", "addition_with_regrouping"]},
        "decay_logit": 0.08,
        "probability_of_learning_without_prerequisites": 0.01,
        "probability_of_learning_with_prerequisites": 0.4,
        "practice_increment_logit": 0.08,
        "initial_skill_level": 0.05,
    },
    {
        "name": "decimal_operations",
        "code": "CCSS.MATH.5.NBT.B.7",
        "description": "Add, subtract, multiply, and divide decimals to hundredths",
        "prerequisites": {"parent_names": ["fraction_concept", "multi_digit_multiplication"]},
        "decay_logit": 0.09,
        "probability_of_learning_without_prerequisites": 0.005,
        "probability_of_learning_with_prerequisites": 0.35,
        "practice_increment_logit": 0.07,
        "initial_skill_level": 0.05,
    },
]

# Create skill space
skill_objects = [Skill(**skill_dict) for skill_dict in skills]
skill_space = SkillSpace(skills=skill_objects)

# Generate 10,000 students with random skill levels
students = create_random_students(skill_space=skill_space, n_students=10000)

# Visualize skill mastery
skill_space.plot_skill_mastery(students=students, filename="skill_mastery.png")

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
    foundation_result = student.take_test(foundation_test, timestamp=1)
    intermediate_result = student.take_test(intermediate_test, timestamp=2)
    advanced_result = student.take_test(advanced_test, timestamp=3)

# Save results
save_student_profile_to_csv(students=students, filename="students.csv")
save_student_activity_to_csv(students=students, filename="student_activity.csv")
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

### 3. **Tiered Assessment Strategy**
- **Foundation test**: Basic number sense and place value
- **Intermediate test**: Basic operations with regrouping
- **Advanced test**: Multiplication, division, fractions, decimals

### 4. **Visualization**
- `plot_skill_mastery()` shows skill dependencies and student counts
- Node size = number of students with the skill
- Edge width = number of students with both connected skills

## Analysis Opportunities

The generated data enables analysis of:
- **Learning progression**: How students advance through skill levels
- **Prerequisite effectiveness**: Impact of prerequisites on learning
- **Assessment validity**: How well tests measure intended skills
- **Student heterogeneity**: Distribution of skill mastery across population

## Scaling Considerations

- **10,000 students**: Good for statistical analysis, manageable computation
- **14 skills**: Complex enough to show learning patterns, not overwhelming
- **3 assessment points**: Captures progression without excessive data
- **25 items per skill**: Sufficient for reliable measurement
