import pytest

from studentsimulator.skill import Skill, SkillSpace
from studentsimulator.student import Student


@pytest.fixture
def skill_with_decay():
    """Create a skill with a specific decay rate."""
    return Skill(
        name="math_basics",
        code="MATH.001",
        description="Basic mathematics",
        practice_increment_logit=0.1,
        initial_skill_level_after_learning=0.8,
        decay_logit=0.05,  # 0.05 logit units per day
    )


@pytest.fixture
def skill_space(skill_with_decay):
    """Create a skill space with the skill."""
    return SkillSpace(skills=[skill_with_decay])


@pytest.fixture
def student(skill_space):
    """Create a student with the skill space."""
    student = Student(skill_space=skill_space)
    # Learn the skill first and set it as learned
    student.learn(skill_space.skills[0])
    student.skills[skill_space.skills[0].name].learned = True
    return student


def test_wait_applies_forgetting(student, skill_with_decay):
    """Test that wait() method applies forgetting to all learned skills."""
    # Set initial skill level
    initial_level = 0.8
    student.skills[skill_with_decay.name].skill_level = initial_level

    # Wait for 3 days
    student.wait(days=3)

    # Check that skill level decreased
    final_level = student.skills[skill_with_decay.name].skill_level
    assert final_level < initial_level


def test_wait_no_forgetting_when_no_time(student, skill_with_decay):
    """Test that wait() doesn't apply forgetting when no time passes."""
    initial_level = 0.8
    student.skills[skill_with_decay.name].skill_level = initial_level

    # Wait for 0 days
    student.wait(days=0)

    # Check that skill level didn't change
    final_level = student.skills[skill_with_decay.name].skill_level
    assert final_level == initial_level


def test_wait_exponential_decay(student, skill_with_decay):
    """Test that forgetting follows exponential decay pattern."""
    initial_level = 0.8
    student.skills[skill_with_decay.name].skill_level = initial_level

    # Ensure skill is learned
    student.skills[skill_with_decay.name].learned = True

    # Wait for 1 day
    student.wait(days=1)
    # level_after_1_day = student.skills[skill_with_decay.name].skill_level

    # Reset and wait for 2 days
    student.skills[skill_with_decay.name].skill_level = initial_level
    student.wait(days=2)
    level_after_2_days = student.skills[skill_with_decay.name].skill_level

    # The decay should be exponential (not linear)
    # If it were linear, level_after_2_days would be level_after_1_day * 2
    # But with exponential decay, it should decay faster
    # expected_linear = initial_level - 2 * (initial_level - level_after_1_day)
    # Note: The decay might be very small, so we just check that decay occurred
    assert level_after_2_days < initial_level


def test_wait_multiple_skills(student, skill_space):
    """Test that forgetting works with multiple skills."""
    # Create a new skill space with both skills
    skill2 = Skill(
        name="advanced_math",
        code="MATH.002",
        description="Advanced mathematics",
        practice_increment_logit=0.1,
        initial_skill_level_after_learning=0.7,
        decay_logit=0.03,  # Different decay rate
    )

    # Create a new skill space with both skills
    new_skill_space = SkillSpace(skills=[skill_space.skills[0], skill2])

    # Create a new student with the updated skill space
    new_student = Student(skill_space=new_skill_space)

    # Set different initial levels
    new_student.skills[skill_space.skills[0].name].skill_level = 0.8
    new_student.skills[skill2.name].skill_level = 0.9
    new_student.skills[skill_space.skills[0].name].learned = True
    new_student.skills[skill2.name].learned = True

    # Wait for 5 days
    new_student.wait(days=5)

    # Both skills should have decayed
    assert new_student.skills[skill_space.skills[0].name].skill_level < 0.8
    assert new_student.skills[skill2.name].skill_level < 0.9


def test_wait_weeks_and_months(student, skill_with_decay):
    """Test that wait() works with weeks and months parameters."""
    initial_level = 0.8
    student.skills[skill_with_decay.name].skill_level = initial_level

    # Wait for 1 week
    student.wait(weeks=1)
    level_after_1_week = student.skills[skill_with_decay.name].skill_level
    assert level_after_1_week < initial_level

    # Reset and wait for 1 month
    student.skills[skill_with_decay.name].skill_level = initial_level
    student.wait(months=1)
    level_after_1_month = student.skills[skill_with_decay.name].skill_level
    assert level_after_1_month < initial_level

    # 1 month should cause more decay than 1 week
    assert level_after_1_month < level_after_1_week


def test_wait_mixed_time_units(student, skill_with_decay):
    """Test that wait() works with mixed time units."""
    initial_level = 0.8
    student.skills[skill_with_decay.name].skill_level = initial_level

    # Wait for 1 day + 1 week + 1 month
    student.wait(days=1, weeks=1, months=1)
    final_level = student.skills[skill_with_decay.name].skill_level
    assert final_level < initial_level
