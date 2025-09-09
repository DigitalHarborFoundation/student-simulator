"""
Test file to verify the advanced-simulation vignette code examples work correctly.
"""

import os
import random

import pandas as pd
import pytest

from studentsimulator.activity_provider import ActivityProvider
from studentsimulator.analytics import plot_skill_mastery, plot_skill_trajectory
from studentsimulator.factory import create_random_students
from studentsimulator.io import (
    save_student_daily_skill_states_to_csv,
    save_student_events_to_csv,
)
from studentsimulator.skill import Skill, SkillSpace
from studentsimulator.student import Student


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set a fixed random seed for all tests in this class."""
    random.seed(42)
    yield


class TestAdvancedSimulationVignette:
    """Test the advanced-simulation vignette examples."""

    def test_complex_skill_structure(self):
        """Test the complex skill structure example from advanced-simulation."""

        # Define comprehensive skill hierarchy (simplified for testing)
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
                prerequisites={
                    "parent_names": ["number_recognition"],
                    "dependence_model": "all",
                },
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
                prerequisites={
                    "parent_names": ["counting_sequence"],
                    "dependence_model": "all",
                },
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
                prerequisites={
                    "parent_names": ["place_value_ones"],
                    "dependence_model": "all",
                },
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
                prerequisites={
                    "parent_names": ["counting_sequence", "place_value_ones"],
                    "dependence_model": "all",
                },
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
                prerequisites={
                    "parent_names": ["addition_within_20"],
                    "dependence_model": "all",
                },
                decay_logit=0.03,
                probability_of_learning_without_prerequisites=0.03,
                probability_of_learning_with_prerequisites=0.82,
                practice_increment_logit=0.15,
                initial_skill_level_after_learning=0.2,
            ),
        ]

        # Create skill space
        skill_space = SkillSpace(skills=skills)

        # Generate 10 students with random skill levels (smaller number for testing)
        students = create_random_students(
            skill_space=skill_space,
            n_students=10,
            practice_count=[10, 50],  # Random practice sessions between 10-50
        )

        # Visualize skill mastery
        plot_skill_mastery(
            skill_space=skill_space,
            students=students,
            filename="test_skill_mastery.png",
        )

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
            skills=[
                skill_space.get_skill("number_recognition"),
                skill_space.get_skill("counting_sequence"),
                skill_space.get_skill("place_value_ones"),
                skill_space.get_skill("place_value_tens"),
            ],
        )

        intermediate_test = activity_provider.generate_fixed_form_assessment(
            n_items=20,
            item_pool=item_pool,
            skills=[
                skill_space.get_skill("addition_within_20"),
                skill_space.get_skill("subtraction_within_20"),
            ],
        )

        # Simulate learning progression over time
        for student in students:
            activity_provider.administer_fixed_form_assessment(student, foundation_test)
            activity_provider.administer_fixed_form_assessment(
                student, intermediate_test
            )

        # Save results
        save_student_daily_skill_states_to_csv(
            students=students, filename="test_students_daily_skill_states.csv"
        )

        # Save with train/validation/test split for machine learning
        save_student_events_to_csv(
            students=students,
            filename="test_student_events.csv",
            train_val_test_split=(
                0.7,
                0.15,
                0.15,
            ),  # 70% train, 15% validation, 15% test
            observation_rate=0.9,  # 90% of events are observed
        )

        # Verify files were created
        assert os.path.exists("test_students_daily_skill_states.csv")
        assert os.path.exists("test_student_events.csv")
        assert os.path.exists("test_skill_mastery.png")

        # Verify CSV has expected columns
        df = pd.read_csv("test_student_events.csv")
        expected_columns = [
            "student_id",
            "day",
            "event_type",
            "skill_id",
            "item_id",
            "score",
            "prob_correct",
            "train_val_test",
        ]
        for col in expected_columns:
            assert col in df.columns

        # Clean up
        os.remove("test_students_daily_skill_states.csv")
        os.remove("test_student_events.csv")
        os.remove("test_skill_mastery.png")

    def test_learning_dynamics(self):
        """Test the learning dynamics example from advanced-simulation."""

        # Create a simplified skill space for testing
        skills = [
            Skill(
                name="number_recognition",
                description="Recognize and write numerals 0-20",
                decay_logit=0.01,
                initial_skill_level_after_learning=0.3,
            ),
            Skill(
                name="addition_within_20",
                description="Add within 20 using strategies",
                prerequisites={
                    "parent_names": ["number_recognition"],
                    "dependence_model": "all",
                },
                decay_logit=0.03,
                initial_skill_level_after_learning=0.25,
            ),
            Skill(
                name="multiplication_concept",
                description="Understand multiplication as repeated addition",
                prerequisites={
                    "parent_names": ["addition_within_20"],
                    "dependence_model": "all",
                },
                decay_logit=0.05,
                initial_skill_level_after_learning=0.25,
            ),
        ]

        skill_space = SkillSpace(skills=skills)

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
        plot_skill_trajectory(student, filename="test_learning_trajectory.png")

        # Verify file was created
        assert os.path.exists("test_learning_trajectory.png")

        # Verify student has learned skills
        assert student.skills["number_recognition"].learned
        assert student.skills["addition_within_20"].learned
        assert student.skills["multiplication_concept"].learned

        # Verify skill levels are reasonable
        assert student.skills["number_recognition"].skill_level > 0.01
        assert student.skills["addition_within_20"].skill_level > 0.01
        assert student.skills["multiplication_concept"].skill_level > 0.01

        # Clean up
        os.remove("test_learning_trajectory.png")

    def test_machine_learning_integration(self):
        """Test the machine learning integration examples from advanced-simulation."""

        # Create a simple skill space
        skills = [
            Skill(
                name="test_skill",
                description="Test skill",
                initial_skill_level_after_learning=0.3,
            )
        ]
        skill_space = SkillSpace(skills=skills)

        # Create students
        students = create_random_students(skill_space=skill_space, n_students=5)

        # Create activity provider and test
        activity_provider = ActivityProvider()
        activity_provider.register_skills(skill_space)

        item_pool = activity_provider.construct_item_pool(
            name="test_pool",
            skills=skill_space.skills,
            n_items_per_skill=10,
            difficulty_logit_range=(-2, 2),
            guess_range=(0.1, 0.3),
            slip_range=(0.01, 0.2),
            discrimination_range=(1.0, 1.0),
        )

        test = activity_provider.generate_fixed_form_assessment(
            n_items=10, item_pool=item_pool, skills=skill_space
        )

        # Take tests
        for student in students:
            activity_provider.administer_fixed_form_assessment(student, test)

        # Test standard export (no split)
        save_student_events_to_csv(students=students, filename="test_all_data.csv")
        assert os.path.exists("test_all_data.csv")

        # Test export with train/validation/test split
        save_student_events_to_csv(
            students=students,
            filename="test_ml_ready_data.csv",
            train_val_test_split=(
                0.7,
                0.15,
                0.15,
            ),  # 70% train, 15% validation, 15% test
            observation_rate=0.95,  # 95% of events are observed
        )
        assert os.path.exists("test_ml_ready_data.csv")

        # Test export with train/test only
        save_student_events_to_csv(
            students=students,
            filename="test_train_test_data.csv",
            train_val_test_split=(0.8, 0.0, 0.2),  # 80% train, 20% test
            observation_rate=0.8,  # 80% of events are observed
        )
        assert os.path.exists("test_train_test_data.csv")

        # Test export with missing data simulation
        save_student_events_to_csv(
            students=students,
            filename="test_missing_data.csv",
            observation_rate=0.7,  # 70% of events are observed (simulates dropout)
        )
        assert os.path.exists("test_missing_data.csv")

        # Verify CSV column structures
        df_standard = pd.read_csv("test_all_data.csv")
        expected_standard_columns = [
            "student_id",
            "day",
            "event_type",
            "skill_id",
            "item_id",
            "score",
            "prob_correct",
        ]
        for col in expected_standard_columns:
            assert col in df_standard.columns

        df_ml = pd.read_csv("test_ml_ready_data.csv")
        expected_ml_columns = [
            "student_id",
            "day",
            "event_type",
            "skill_id",
            "item_id",
            "score",
            "prob_correct",
            "train_val_test",
        ]
        for col in expected_ml_columns:
            assert col in df_ml.columns

        # Clean up
        os.remove("test_all_data.csv")
        os.remove("test_ml_ready_data.csv")
        os.remove("test_train_test_data.csv")
        os.remove("test_missing_data.csv")

    def test_skill_parameters_advanced(self):
        """Test that advanced skill parameters work as described in the vignette."""

        skill = Skill(
            name="advanced_skill",
            code="CCSS.MATH.5.NBT.B.7",
            description="Advanced mathematical operations",
            prerequisites={"parent_names": ["basic_skill"], "dependence_model": "all"},
            decay_logit=0.09,
            probability_of_learning_without_prerequisites=0.005,
            probability_of_learning_with_prerequisites=0.5,
            practice_increment_logit=0.07,
            initial_skill_level_after_learning=0.05,
        )

        assert skill.name == "advanced_skill"
        assert skill.code == "CCSS.MATH.5.NBT.B.7"
        assert skill.prerequisites.parent_names == ["basic_skill"]
        assert skill.prerequisites.dependence_model == "all"
        assert skill.decay_logit == 0.09
        assert skill.probability_of_learning_without_prerequisites == 0.005
        assert skill.probability_of_learning_with_prerequisites == 0.5
        assert skill.practice_increment_logit == 0.07
        assert skill.initial_skill_level_after_learning == 0.05
