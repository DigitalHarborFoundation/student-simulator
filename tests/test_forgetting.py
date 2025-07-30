import pytest

from studentsimulator.general import Skill, SkillSpace
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
    student.skill_state[skill_space.skills[0].name].learned = True
    return student


def test_forget_learned_skill(student, skill_with_decay):
    """Test that forgetting reduces skill level for learned skills."""
    # Set initial skill level
    initial_level = 0.8
    student.skill_state[skill_with_decay.name].skill_level = initial_level

    # Apply forgetting for 5 days
    student.forget(skill_with_decay, time_days=5)

    # Check that skill level decreased
    final_level = student.skill_state[skill_with_decay.name].skill_level
    assert final_level < initial_level
    assert final_level > 0.01  # Should not go below minimum


def test_forget_unlearned_skill(student, skill_with_decay):
    """Test that unlearned skills don't decay."""
    # Set skill as unlearned
    student.skill_state[skill_with_decay.name].learned = False
    initial_level = 0.5
    student.skill_state[skill_with_decay.name].skill_level = initial_level

    # Apply forgetting for 10 days
    student.forget(skill_with_decay, time_days=10)

    # Check that skill level didn't change
    final_level = student.skill_state[skill_with_decay.name].skill_level
    assert final_level == initial_level


def test_forget_custom_rate(student, skill_with_decay):
    """Test that custom forgetting rate works."""
    initial_level = 0.8
    student.skill_state[skill_with_decay.name].skill_level = initial_level

    # Ensure skill is learned
    student.skill_state[skill_with_decay.name].learned = True

    # Use custom rate (higher than skill's default)
    custom_rate = 0.1  # Higher decay rate
    student.forget(skill_with_decay, time_days=5, rate=custom_rate)

    # Should decay more than with default rate
    final_level = student.skill_state[skill_with_decay.name].skill_level
    assert final_level < initial_level


def test_forget_no_time_passed(student, skill_with_decay):
    """Test that no forgetting occurs when no time passes."""
    initial_level = 0.8
    student.skill_state[skill_with_decay.name].skill_level = initial_level

    # Apply forgetting for 0 days
    student.forget(skill_with_decay, time_days=0)

    # Check that skill level didn't change
    final_level = student.skill_state[skill_with_decay.name].skill_level
    assert final_level == initial_level


def test_forget_minimum_level(student, skill_with_decay):
    """Test that skill level doesn't go below minimum threshold."""
    # Set very low initial level
    initial_level = 0.02
    student.skill_state[skill_with_decay.name].skill_level = initial_level

    # Apply forgetting for many days
    student.forget(skill_with_decay, time_days=100)

    # Check that skill level doesn't go below 0.01
    final_level = student.skill_state[skill_with_decay.name].skill_level
    assert final_level >= 0.01


def test_wait_applies_forgetting(student, skill_with_decay):
    """Test that wait() method applies forgetting to all learned skills."""
    # Set initial skill level
    initial_level = 0.8
    student.skill_state[skill_with_decay.name].skill_level = initial_level

    # Wait for 3 days
    student.wait(days=3)

    # Check that skill level decreased
    final_level = student.skill_state[skill_with_decay.name].skill_level
    assert final_level < initial_level

    # Check that days_since_initialization increased
    assert student.days_since_initialization == 3


def test_wait_no_forgetting_when_no_time(student, skill_with_decay):
    """Test that wait() doesn't apply forgetting when no time passes."""
    initial_level = 0.8
    student.skill_state[skill_with_decay.name].skill_level = initial_level

    # Wait for 0 days
    student.wait(days=0)

    # Check that skill level didn't change
    final_level = student.skill_state[skill_with_decay.name].skill_level
    assert final_level == initial_level


def test_forget_exponential_decay(student, skill_with_decay):
    """Test that forgetting follows exponential decay pattern."""
    initial_level = 0.8
    student.skill_state[skill_with_decay.name].skill_level = initial_level

    # Ensure skill is learned
    student.skill_state[skill_with_decay.name].learned = True

    # Apply forgetting for different time periods
    student.forget(skill_with_decay, time_days=1)
    level_after_1_day = student.skill_state[skill_with_decay.name].skill_level

    # Reset and apply forgetting for 2 days
    student.skill_state[skill_with_decay.name].skill_level = initial_level
    student.forget(skill_with_decay, time_days=2)
    level_after_2_days = student.skill_state[skill_with_decay.name].skill_level

    # The decay should be exponential (not linear)
    # If it were linear, level_after_2_days would be level_after_1_day * 2
    # But with exponential decay, it should decay faster
    expected_linear = initial_level - 2 * (initial_level - level_after_1_day)
    assert level_after_2_days < expected_linear  # Exponential decay is faster initially


def test_forget_multiple_skills(student, skill_space):
    """Test that forgetting works with multiple skills."""
    # Add another skill
    skill2 = Skill(
        name="advanced_math",
        code="MATH.002",
        description="Advanced mathematics",
        practice_increment_logit=0.1,
        initial_skill_level_after_learning=0.7,
        decay_logit=0.03,  # Different decay rate
    )
    skill_space.skills.append(skill2)

    # Update student's skill space and initialize the new skill
    student.skill_space = skill_space
    from studentsimulator.student import SkillState

    student.skill_state[skill2.name] = SkillState(
        skill_name=skill2.name, skill_level=0.9, learned=True
    )

    # Set different initial levels
    student.skill_state[skill_space.skills[0].name].skill_level = 0.8
    student.skill_state[skill_space.skills[1].name].skill_level = 0.9

    # Wait for 5 days
    student.wait(days=5)

    # Both skills should have decayed
    assert student.skill_state[skill_space.skills[0].name].skill_level < 0.8
    assert student.skill_state[skill_space.skills[1].name].skill_level < 0.9
