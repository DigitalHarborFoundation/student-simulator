import pytest

from studentsimulator.analytics import plot_skill_trajectory
from studentsimulator.item import Item
from studentsimulator.skill import Skill, SkillSpace
from studentsimulator.student import Student


@pytest.fixture
def simple_skill():
    """Create a simple skill for testing."""
    return Skill(
        name="math_basics",
        code="MATH.001",
        description="Basic mathematics",
        practice_increment_logit=0.1,
        decay_logit=0.05,
        initial_skill_level_after_learning=0.3,
    )


@pytest.fixture
def student_with_history(simple_skill):
    """Create a student with some learning history."""
    skill_space = SkillSpace(skills=[simple_skill])
    student = Student(name="Test Student", skill_space=skill_space)

    # Learn the skill
    student.learn(simple_skill)

    # Practice with some items
    item1 = Item(skill=simple_skill, difficulty_logit=0.0)
    item2 = Item(skill=simple_skill, difficulty_logit=0.5)

    student.practice(simple_skill, item=item1)
    student.practice(simple_skill, item=item2)

    # Wait some time
    student.wait(days=2)

    # More practice
    student.practice(simple_skill, item=item1)

    return student


def test_daily_history_basic_functionality(simple_skill):
    """Test that end-of-day skill states are created and updated correctly."""
    skill_space = SkillSpace(skills=[simple_skill])
    student = Student(name="Test Student", skill_space=skill_space)

    # Should have initial snapshot at day 0
    trajectory = student.skills.end_of_day_skill_states.get_skill_trajectories(
        "math_basics"
    )
    assert len(trajectory) == 1
    assert trajectory["math_basics"][0] == (0, 0.01)  # Default minimum skill level

    # Learn skill (happens on same day, should update existing snapshot)
    student.learn(simple_skill)

    # Should still have 1 snapshot, but with updated skill level
    trajectory = student.skills.end_of_day_skill_states.get_skill_trajectories(
        "math_basics"
    )
    assert len(trajectory) == 1
    assert trajectory["math_basics"][0][0] == 0  # Same day
    assert (
        trajectory["math_basics"][0][1] == 0.3
    )  # Updated to initial_skill_level_after_learning


def test_daily_history_with_practice(simple_skill):
    """Test that practice updates end-of-day skill states."""
    skill_space = SkillSpace(skills=[simple_skill])
    student = Student(name="Test Student", skill_space=skill_space)

    # Learn and practice (both happen on same day)
    student.learn(simple_skill)
    initial_level = student.skills["math_basics"].skill_level

    item = Item(skill=simple_skill, difficulty_logit=0.0)
    student.practice(simple_skill, item=item)

    # Should still have 1 snapshot (same day), but skill level should be higher after practice
    trajectory = student.skills.end_of_day_skill_states.get_skill_trajectories(
        "math_basics"
    )
    assert len(trajectory) == 1

    # Final level should be higher than initial level due to practice
    final_level = trajectory["math_basics"][0][1]
    assert final_level > initial_level


def test_daily_history_with_waiting(simple_skill):
    """Test that waiting updates end-of-day skill states with forgetting."""
    skill_space = SkillSpace(skills=[simple_skill])
    student = Student(name="Test Student", skill_space=skill_space)

    # Learn skill
    student.learn(simple_skill)
    initial_level = student.skills["math_basics"].skill_level

    # Wait 3 days
    student.wait(days=3)

    # Should have snapshots for each day
    trajectory = student.skills.end_of_day_skill_states.get_skill_trajectories(
        "math_basics"
    )
    assert len(trajectory) == 1
    assert len(trajectory["math_basics"]) >= 4  # Initial + learning + 3 days of waiting

    # Final level should be lower due to forgetting
    final_level = trajectory["math_basics"][-1][1]
    assert final_level < initial_level


def test_plot_skill_trajectory_single_skill(student_with_history):
    """Test plotting trajectory for a single skill."""
    # Should not raise an exception
    plot_skill_trajectory(student_with_history, skill_name="math_basics")


def test_plot_skill_trajectory_all_skills(student_with_history):
    """Test plotting trajectories for all skills."""
    # Should not raise an exception
    plot_skill_trajectory(student_with_history)


def test_plot_skill_trajectory_faceted(student_with_history):
    """Test plotting trajectories with faceted subplots."""
    # Should not raise an exception
    plot_skill_trajectory(student_with_history, faceted=True)


def test_get_all_skill_trajectories(student_with_history):
    """Test getting trajectories for all skills."""
    all_trajectories = (
        student_with_history.skills.end_of_day_skill_states.get_skill_trajectories()
    )

    assert isinstance(all_trajectories, dict)
    assert "math_basics" in all_trajectories

    trajectory = all_trajectories["math_basics"]
    assert isinstance(trajectory, list)
    assert len(trajectory) > 1

    # Each point should be (day, level)
    for day, level in trajectory:
        assert isinstance(day, int)
        assert isinstance(level, float)
        assert 0.0 <= level <= 1.0


def test_daily_history_deterministic():
    """Test that end-of-day skill states are deterministic."""
    skill = Skill(
        name="test_skill",
        practice_increment_logit=0.1,
        decay_logit=0.05,
        initial_skill_level_after_learning=0.5,
    )
    skill_space = SkillSpace(skills=[skill])

    # Create two identical students
    student1 = Student(name="Student1", skill_space=skill_space)
    student2 = Student(name="Student2", skill_space=skill_space)

    # Set the same random seed for both
    import random

    random.seed(42)
    student1.learn(skill)
    student1.wait(days=2)

    random.seed(42)
    student2.learn(skill)
    student2.wait(days=2)

    # Trajectories should be identical
    traj1 = student1.skills.end_of_day_skill_states.get_skill_trajectories("test_skill")
    traj2 = student2.skills.end_of_day_skill_states.get_skill_trajectories("test_skill")

    assert len(traj1) == len(traj2)
    for (day1, level1), (day2, level2) in zip(traj1["test_skill"], traj2["test_skill"]):
        assert day1 == day2
        assert abs(level1 - level2) < 1e-10  # Should be exactly equal
