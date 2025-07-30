"""Test to validate that student skill levels correlate with response accuracy.

This test ensures that the simulation produces realistic behavior where:
1. Students with higher skill levels tend to answer questions correctly more often
2. Students with lower skill levels tend to answer questions incorrectly more often
3. The correlation between skill level and response accuracy is statistically significant
"""

import statistics
from typing import List, Tuple

import pytest

from studentsimulator.activity_provider import ActivityProvider
from studentsimulator.general import Skill, SkillSpace
from studentsimulator.student import Student


@pytest.fixture
def simple_skill_space() -> SkillSpace:
    """Create a simple skill space with one skill for testing."""
    skill = Skill(
        name="test_skill",
        description="A test skill for validation",
        probability_of_learning_without_prerequisites=0.0,  # Must be learned explicitly
        probability_of_learning_with_prerequisites=1.0,
        practice_increment_logit=0.1,
        initial_skill_level_after_learning=0.5,
    )
    return SkillSpace(skills=[skill])


@pytest.fixture
def activity_provider(simple_skill_space: SkillSpace) -> ActivityProvider:
    """Create an activity provider with the test skill space."""
    provider = ActivityProvider()
    provider.register_skills(simple_skill_space)
    return provider


def create_students_with_skill_levels(
    skill_space: SkillSpace, skill_levels: List[float]
) -> List[Student]:
    """Create students with specific skill levels."""
    students = []
    for i, skill_level in enumerate(skill_levels):
        student = Student(name=f"student_{i}", skill_space=skill_space)
        # Set the skill level and mark as learned
        student.skill_state["test_skill"].skill_level = skill_level
        student.skill_state["test_skill"].learned = True
        students.append(student)
    return students


def calculate_skill_accuracy_correlation(
    students: List[Student], responses_per_student: int = 50
) -> Tuple[float, List[float], List[float]]:
    """
    Calculate correlation between skill levels and response accuracy.

    Returns:
        Tuple of (correlation, skill_levels, accuracies)
    """
    # Create a simple item pool and assessment
    provider = ActivityProvider()
    provider.register_skills(students[0].skill_space)

    item_pool = provider.construct_item_pool(
        name="test_pool",
        skills=students[0].skill_space.skills,
        n_items_per_skill=responses_per_student,
        difficulty_logit_range=(-1.0, 1.0),  # Moderate difficulty range
        guess_range=(0.1, 0.2),
        slip_range=(0.05, 0.1),
        discrimination_range=(0.8, 1.2),
    )

    assessment = provider.generate_fixed_form_assessment(
        n_items=responses_per_student,
        item_pool=item_pool,
        skills=students[0].skill_space.skills,
    )

    # Collect skill levels and accuracies
    skill_levels = []
    accuracies = []

    for student in students:
        # Get student's skill level
        skill_level = student.skill_state["test_skill"].skill_level
        skill_levels.append(skill_level)

        # Have student take the assessment multiple times to get stable accuracy
        total_correct = 0
        total_responses = 0

        for _ in range(5):  # Take assessment 5 times for stable measurement
            test_results = provider.administer_fixed_form_assessment(
                student_or_students=student, test=assessment
            )
            test_result = test_results[0]  # Get the first (and only) result
            total_correct += sum(
                1 for event in test_result.behavioral_events if event.score == 1
            )
            total_responses += len(test_result.behavioral_events)

        accuracy = total_correct / total_responses if total_responses > 0 else 0.0
        accuracies.append(accuracy)

    # Calculate correlation
    if len(skill_levels) > 1:
        correlation = statistics.correlation(skill_levels, accuracies)
    else:
        correlation = 0.0

    return correlation, skill_levels, accuracies


def test_skill_level_correlates_with_accuracy(simple_skill_space: SkillSpace):
    """Test that higher skill levels correlate with higher accuracy."""
    # Create students with varying skill levels
    skill_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    students = create_students_with_skill_levels(simple_skill_space, skill_levels)

    # Calculate correlation
    correlation, skill_levels, accuracies = calculate_skill_accuracy_correlation(
        students
    )

    # Assertions
    assert correlation > 0.5, f"Expected positive correlation, got {correlation:.3f}"
    assert correlation <= 1.0, f"Correlation should be <= 1.0, got {correlation:.3f}"

    # Check that higher skill levels generally have higher accuracy
    # (allowing for some noise due to randomness)
    for i in range(len(skill_levels) - 1):
        if skill_levels[i] < skill_levels[i + 1]:
            # Higher skill level should generally have higher accuracy
            # Allow some tolerance for randomness
            assert (
                accuracies[i] <= accuracies[i + 1] + 0.2
            ), f"Student {i} (skill={skill_levels[i]:.2f}) should not have much higher accuracy than student {i+1} (skill={skill_levels[i+1]:.2f})"


def test_low_skill_students_perform_poorly(simple_skill_space: SkillSpace):
    """Test that students with very low skill levels perform poorly."""
    # Create students with very low skill levels
    low_skill_levels = [0.0, 0.05, 0.1, 0.15, 0.2]
    students = create_students_with_skill_levels(simple_skill_space, low_skill_levels)

    # Calculate accuracies
    _, _, accuracies = calculate_skill_accuracy_correlation(students)

    # Students with very low skill levels should have low accuracy
    # (mostly guessing or making mistakes)
    for accuracy in accuracies:
        assert (
            accuracy < 0.4
        ), f"Low skill student should have accuracy < 0.4, got {accuracy:.3f}"


def test_high_skill_students_perform_well(simple_skill_space: SkillSpace):
    """Test that students with very high skill levels perform well."""
    # Create students with very high skill levels
    high_skill_levels = [0.8, 0.85, 0.9, 0.95, 1.0]
    students = create_students_with_skill_levels(simple_skill_space, high_skill_levels)

    # Calculate accuracies
    _, _, accuracies = calculate_skill_accuracy_correlation(students)

    # Students with very high skill levels should have high accuracy
    for accuracy in accuracies:
        assert (
            accuracy > 0.6
        ), f"High skill student should have accuracy > 0.6, got {accuracy:.3f}"


def test_skill_response_monotonicity(simple_skill_space: SkillSpace):
    """Test that response accuracy increases monotonically with skill level."""
    # Create students with evenly spaced skill levels
    skill_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    students = create_students_with_skill_levels(simple_skill_space, skill_levels)

    # Calculate correlation and data
    correlation, skill_levels, accuracies = calculate_skill_accuracy_correlation(
        students, responses_per_student=100
    )

    # Check monotonicity: each higher skill level should have higher or equal accuracy
    # (allowing for some noise)
    violations = 0
    for i in range(len(skill_levels) - 1):
        if (
            accuracies[i] > accuracies[i + 1] + 0.15
        ):  # Allow some tolerance for randomness
            violations += 1

    # Should have very few violations (at most 1 out of 4 comparisons)
    assert (
        violations <= 1
    ), f"Too many monotonicity violations: {violations} out of {len(skill_levels)-1}"


def test_probability_calculation_accuracy(simple_skill_space: SkillSpace):
    """Test that the theoretical probabilities match observed accuracy."""
    # Create a student with a known skill level
    student = create_students_with_skill_levels(simple_skill_space, [0.7])[0]

    # Create a simple item
    provider = ActivityProvider()
    provider.register_skills(simple_skill_space)

    item_pool = provider.construct_item_pool(
        name="test_pool",
        skills=simple_skill_space.skills,
        n_items_per_skill=1,
        difficulty_logit_range=(0.0, 0.0),  # Fixed difficulty
        guess_range=(0.1, 0.1),  # Fixed guess
        slip_range=(0.05, 0.05),  # Fixed slip
        discrimination_range=(1.0, 1.0),  # Fixed discrimination
    )

    item = item_pool.items[0]

    # Calculate theoretical probability
    theoretical_prob = student.get_prob_correct(item)

    # Simulate many responses to get empirical probability
    n_trials = 1000
    correct_count = 0

    for _ in range(n_trials):
        response = student._respond_to_item(item)
        if response.score == 1:
            correct_count += 1

    empirical_prob = correct_count / n_trials

    # The empirical probability should be close to the theoretical probability
    # (within 0.05 due to sampling error)
    assert (
        abs(empirical_prob - theoretical_prob) < 0.05
    ), f"Theoretical prob {theoretical_prob:.3f} should be close to empirical prob {empirical_prob:.3f}"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
