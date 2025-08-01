"""
Test file to verify the getting-started vignette code examples work correctly.
"""

import os

import pandas as pd

from studentsimulator.activity_provider import ActivityProvider
from studentsimulator.factory import create_random_students
from studentsimulator.io import (
    save_student_daily_skill_states_to_csv,
    save_student_events_to_csv,
)
from studentsimulator.skill import Skill, SkillSpace
from studentsimulator.student import Student


class TestGettingStartedVignette:
    """Test the getting-started vignette examples."""

    def test_manual_student_creation(self):
        """Test the manual student creation example from getting-started."""

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
        student2 = Student(skill_space=skill_space).set_skill_values(
            {"number_recognition": 0.5, "place_value_ones": 0.3}
        )

        ## Setting skills with more advanced knowledge
        student3 = Student(skill_space=skill_space).set_skill_values(
            {
                "number_recognition": 0.8,
                "place_value_ones": 0.6,
                "place_value_tens": 0.4,
                "addition_no_carry": 0.9,
            }
        )

        ## Randomly initialize skills while respecting prerequisites
        student4 = Student(skill_space=skill_space).randomly_initialize_skills(
            practice_count=[1, 9]
        )

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
            n_items=20, item_pool=item_pool, skills=skill_space
        )

        # Students take the test
        for student in [student1, student2, student3, student4]:
            activity_provider.administer_fixed_form_assessment(student, test)

        # Save results
        save_student_daily_skill_states_to_csv(
            students=[student1, student2, student3, student4],
            filename="test_students_daily_skill_states.csv",
        )
        save_student_events_to_csv(
            students=[student1, student2, student3, student4],
            filename="test_student_events.csv",
        )

        # Verify files were created
        assert os.path.exists("test_students_daily_skill_states.csv")
        assert os.path.exists("test_student_events.csv")

        # Verify student skill states
        assert (
            student1.skills["number_recognition"].skill_level >= 0.01
        )  # Minimum level
        assert student2.skills["number_recognition"].skill_level == 0.5
        assert student3.skills["addition_no_carry"].skill_level == 0.9

        # Clean up
        os.remove("test_students_daily_skill_states.csv")
        os.remove("test_student_events.csv")

    def test_group_student_creation(self):
        """Test the group student creation example from getting-started."""

        # Define skills
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
        ]

        skill_space = SkillSpace(skills=skills)

        # Create 10 students with random skill levels (smaller number for testing)
        students = create_random_students(
            skill_space=skill_space,
            n_students=10,
            practice_count=[5, 20],  # Random practice sessions between 5-20
        )

        # Create activity provider
        activity_provider = ActivityProvider()
        activity_provider.register_skills(skill_space)

        item_pool = activity_provider.construct_item_pool(
            name="basic_math_pool",
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

        # Take assessments
        for student in students:
            activity_provider.administer_fixed_form_assessment(student, test)

        # Save with train/validation/test split for machine learning
        save_student_events_to_csv(
            students=students,
            filename="test_student_events_with_split.csv",
            train_val_test_split=(
                0.7,
                0.15,
                0.15,
            ),  # 70% train, 15% validation, 15% test
            observation_rate=0.9,  # 90% of events are observed
        )

        # Verify file was created
        assert os.path.exists("test_student_events_with_split.csv")

        # Verify CSV has expected columns
        df = pd.read_csv("test_student_events_with_split.csv")
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

        # Verify train/val/test split values
        assert set(df["train_val_test"].unique()).issubset({0, 1, 2})

        # Clean up
        os.remove("test_student_events_with_split.csv")

    def test_skill_parameters(self):
        """Test that skill parameters work as described in the vignette."""

        skill = Skill(
            name="test_skill",
            description="Test skill",
            prerequisites={"parent_names": ["prereq_skill"]},
            probability_of_learning_without_prerequisites=0.1,
            probability_of_learning_with_prerequisites=0.9,
            decay_logit=0.02,
            initial_skill_level_after_learning=0.3,
            practice_increment_logit=0.1,
        )

        assert skill.name == "test_skill"
        assert skill.prerequisites.parent_names == ["prereq_skill"]
        assert skill.probability_of_learning_without_prerequisites == 0.1
        assert skill.probability_of_learning_with_prerequisites == 0.9
        assert skill.decay_logit == 0.02
        assert skill.initial_skill_level_after_learning == 0.3
        assert skill.practice_increment_logit == 0.1

    def test_assessment_parameters(self):
        """Test that assessment parameters work as described in the vignette."""

        skills = [
            Skill(
                name="test_skill",
                description="Test skill",
                initial_skill_level_after_learning=0.3,
            )
        ]
        skill_space = SkillSpace(skills=skills)

        activity_provider = ActivityProvider()
        activity_provider.register_skills(skill_space)

        item_pool = activity_provider.construct_item_pool(
            name="test_pool",
            skills=skill_space.skills,
            n_items_per_skill=20,
            difficulty_logit_range=(-2, 2),
            guess_range=(0.1, 0.3),
            slip_range=(0.01, 0.2),
            discrimination_range=(1.0, 1.0),
        )

        assert len(item_pool.items) == 20
        assert item_pool.name == "test_pool"
