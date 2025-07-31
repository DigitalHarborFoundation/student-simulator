import csv
import tempfile
from pathlib import Path

import pytest

from studentsimulator.activity_provider import ActivityProvider
from studentsimulator.general import Skill, SkillSpace
from studentsimulator.student import (
    OBSERVED,
    TEST_SPLIT,
    TRAIN_SPLIT,
    UNOBSERVED,
    VAL_SPLIT,
    Student,
    save_student_activity_to_csv,
)


@pytest.fixture
def sample_skill_space():
    """Create a simple skill space for testing."""
    skills = [
        Skill(name="skill1", prerequisites={"parent_names": []}),
        Skill(name="skill2", prerequisites={"parent_names": ["skill1"]}),
        Skill(name="skill3", prerequisites={"parent_names": ["skill2"]}),
    ]
    return SkillSpace(skills=skills)


@pytest.fixture
def sample_students(sample_skill_space):
    """Create a list of students for testing."""
    students = []
    for i in range(100):  # Create 100 students for good split testing
        student = Student(name=f"student_{i}", skill_space=sample_skill_space)
        student.randomly_initialize_skills(practice_count=[1, 3])
        students.append(student)
    return students


@pytest.fixture
def activity_provider(sample_skill_space):
    """Create an activity provider with assessments."""
    provider = ActivityProvider()
    provider.register_skills(sample_skill_space)

    # Create item pool
    item_pool = provider.construct_item_pool(
        name="test_pool",
        skills=sample_skill_space.skills,
        n_items_per_skill=5,
        difficulty_logit_range=(-1, 1),
        guess_range=(0.1, 0.2),
        slip_range=(0.01, 0.1),
    )

    # Create assessment
    assessment = provider.generate_fixed_form_assessment(
        n_items=10, item_pool=item_pool, skills=sample_skill_space
    )

    return provider, assessment


def test_train_val_test_split_basic(sample_students, activity_provider):
    """Test basic train/validation/test split functionality."""
    provider, assessment = activity_provider

    # Have students take the assessment
    for student in sample_students:
        provider.administer_fixed_form_assessment(
            student_or_students=student, test=assessment
        )

    # Test with 70/15/15 split
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        save_student_activity_to_csv(
            students=sample_students,
            filename=tmp_path,
            train_val_test_split=(0.7, 0.15, 0.15),
        )

        # Read the CSV and check the split
        with open(tmp_path, "r") as f:
            reader = csv.DictReader(f)
            # Check that train_val_test column exists
            assert reader.fieldnames is not None
            assert "train_val_test" in reader.fieldnames

            # Count students in each split
            student_splits = {}
            for row in reader:
                student_id = int(row["studentid"])
                split = int(row["train_val_test"])
                if student_id not in student_splits:
                    student_splits[student_id] = split

        # Count splits
        train_count = sum(
            1 for split in student_splits.values() if split == TRAIN_SPLIT
        )
        val_count = sum(1 for split in student_splits.values() if split == VAL_SPLIT)
        test_count = sum(1 for split in student_splits.values() if split == TEST_SPLIT)

        # Check proportions (allow some tolerance for rounding)
        total_students = len(student_splits)
        assert (
            abs(train_count / total_students - 0.7) < 0.05
        ), f"Train proportion {train_count/total_students:.3f} not close to 0.7"
        assert (
            abs(val_count / total_students - 0.15) < 0.05
        ), f"Val proportion {val_count/total_students:.3f} not close to 0.15"
        assert (
            abs(test_count / total_students - 0.15) < 0.05
        ), f"Test proportion {test_count/total_students:.3f} not close to 0.15"

        # Check that all students have the same split value across all their events
        with open(tmp_path, "r") as f:
            reader = csv.DictReader(f)
            for student_id, split in student_splits.items():
                student_rows = [
                    row for row in reader if int(row["studentid"]) == student_id
                ]
                for row in student_rows:
                    assert (
                        int(row["train_val_test"]) == split
                    ), f"Student {student_id} has inconsistent split values"

    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_train_val_test_split_no_split(sample_students, activity_provider):
    """Test that CSV works correctly without split (backward compatibility)."""
    provider, assessment = activity_provider

    # Have students take the assessment
    for student in sample_students:
        provider.administer_fixed_form_assessment(
            student_or_students=student, test=assessment
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        save_student_activity_to_csv(students=sample_students, filename=tmp_path)

        # Read the CSV and check that train_val_test column is NOT present
        with open(tmp_path, "r") as f:
            reader = csv.DictReader(f)
            # Check that train_val_test column does NOT exist
            assert reader.fieldnames is not None
            assert "train_val_test" not in reader.fieldnames
            assert "observed" not in reader.fieldnames

            # Check that we have the expected columns
            expected_columns = ["studentid", "timeid", "itemid", "response", "groupid"]
            for col in expected_columns:
                assert col in reader.fieldnames, f"Expected column {col} not found"

    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_train_val_test_split_validation():
    """Test that invalid split percentages raise an error."""
    skill_space = SkillSpace(skills=[Skill(name="skill1")])
    student = Student(name="test", skill_space=skill_space)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        # Test that percentages must sum to 1.0
        with pytest.raises(ValueError, match="percentages must sum to 1.0"):
            save_student_activity_to_csv(
                students=[student],
                filename=tmp_path,
                train_val_test_split=(0.5, 0.3, 0.3),  # Sums to 1.1
            )

        with pytest.raises(ValueError, match="percentages must sum to 1.0"):
            save_student_activity_to_csv(
                students=[student],
                filename=tmp_path,
                train_val_test_split=(0.5, 0.3, 0.1),  # Sums to 0.9
            )

    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_train_val_test_split_edge_cases(sample_students, activity_provider):
    """Test edge cases for train/validation/test split."""
    provider, assessment = activity_provider

    # Have students take the assessment
    for student in sample_students:
        provider.administer_fixed_form_assessment(
            student_or_students=student, test=assessment
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        # Test 100% train (0% val, 0% test)
        save_student_activity_to_csv(
            students=sample_students,
            filename=tmp_path,
            train_val_test_split=(1.0, 0.0, 0.0),
        )

        with open(tmp_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # All should be train (0)
        student_splits = set()
        for row in rows:
            # student_id = int(row["studentid"])  # Removed unused variable
            split = int(row["train_val_test"])
            student_splits.add(split)

        assert student_splits == {
            TRAIN_SPLIT
        }, f"All students should be in train set, got {student_splits}"

    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_train_val_test_split_reproducibility(sample_students, activity_provider):
    """Test that the split is reproducible with the same random seed."""
    provider, assessment = activity_provider

    # Have students take the assessment
    for student in sample_students:
        provider.administer_fixed_form_assessment(
            student_or_students=student, test=assessment
        )

    # Set a fixed random seed for reproducibility
    import random

    random.seed(42)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp_file:
        tmp_path1 = tmp_file.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp_file:
        tmp_path2 = tmp_file.name

    try:
        # Generate two splits with the same seed
        random.seed(42)
        save_student_activity_to_csv(
            students=sample_students,
            filename=tmp_path1,
            train_val_test_split=(0.7, 0.15, 0.15),
        )

        random.seed(42)
        save_student_activity_to_csv(
            students=sample_students,
            filename=tmp_path2,
            train_val_test_split=(0.7, 0.15, 0.15),
        )

        # Read both files and compare
        with open(tmp_path1, "r") as f1, open(tmp_path2, "r") as f2:
            reader1 = csv.DictReader(f1)
            reader2 = csv.DictReader(f2)
            rows1 = list(reader1)
            rows2 = list(reader2)

        # Check that the splits are identical
        for row1, row2 in zip(rows1, rows2):
            assert (
                row1["train_val_test"] == row2["train_val_test"]
            ), "Splits should be identical with same seed"

    finally:
        Path(tmp_path1).unlink(missing_ok=True)
        Path(tmp_path2).unlink(missing_ok=True)


def test_observation_rate_basic(sample_students, activity_provider):
    """Test basic observation rate functionality."""
    provider, assessment = activity_provider

    # Have students take the assessment
    for student in sample_students:
        provider.administer_fixed_form_assessment(
            student_or_students=student, test=assessment
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        # Test with 80% observation rate
        save_student_activity_to_csv(
            students=sample_students, filename=tmp_path, observation_rate=0.8
        )

        # Read the CSV and check the observation rate
        with open(tmp_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Check that observed column exists
        assert reader.fieldnames is not None
        assert "observed" in reader.fieldnames

        # Count observed vs unobserved events
        observed_count = sum(1 for row in rows if int(row["observed"]) == OBSERVED)
        unobserved_count = sum(1 for row in rows if int(row["observed"]) == UNOBSERVED)
        total_count = len(rows)

        # Check that we have both observed and unobserved events
        assert observed_count > 0, "Should have some observed events"
        assert unobserved_count > 0, "Should have some unobserved events"

        # Check that the proportion is roughly correct (allow some tolerance)
        actual_rate = observed_count / total_count
        assert (
            abs(actual_rate - 0.8) < 0.1
        ), f"Observation rate {actual_rate:.3f} not close to 0.8"

    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_observation_rate_validation():
    """Test that invalid observation rates raise an error."""
    skill_space = SkillSpace(skills=[Skill(name="skill1")])
    student = Student(name="test", skill_space=skill_space)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        # Test that observation rate must be between 0 and 1
        with pytest.raises(
            ValueError, match="observation_rate must be between 0.0 and 1.0"
        ):
            save_student_activity_to_csv(
                students=[student], filename=tmp_path, observation_rate=1.5  # Too high
            )

        with pytest.raises(
            ValueError, match="observation_rate must be between 0.0 and 1.0"
        ):
            save_student_activity_to_csv(
                students=[student], filename=tmp_path, observation_rate=-0.1  # Too low
            )

    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_observation_rate_with_train_val_test_split(sample_students, activity_provider):
    """Test that observation rate works with train/validation/test split."""
    provider, assessment = activity_provider

    # Have students take the assessment
    for student in sample_students:
        provider.administer_fixed_form_assessment(
            student_or_students=student, test=assessment
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        save_student_activity_to_csv(
            students=sample_students,
            filename=tmp_path,
            train_val_test_split=(0.7, 0.15, 0.15),
            observation_rate=0.9,
        )

        # Read the CSV and check both features work together
        with open(tmp_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Check that both columns exist
        assert reader.fieldnames is not None
        assert "train_val_test" in reader.fieldnames
        assert "observed" in reader.fieldnames

        # Check that all values are valid
        for row in rows:
            train_val_test = int(row["train_val_test"])
            observed = int(row["observed"])
            assert train_val_test in [
                0,
                1,
                2,
            ], f"Invalid train_val_test value: {train_val_test}"
            assert observed in [0, 1], f"Invalid observed value: {observed}"

    finally:
        Path(tmp_path).unlink(missing_ok=True)
