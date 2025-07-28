"""Tests for skill transfer effects during practice."""

import sys

import pytest

from studentsimulator.general import Skill, SkillSpace
from studentsimulator.student import Student

print("PYTHONPATH:", sys.path)
print("STUDENT MODULE:", sys.modules.get("studentsimulator"))


@pytest.fixture
def hierarchical_skill_space():
    """Create a skill space with a clear hierarchy for testing transfer effects."""
    skills = [
        Skill(
            name="basic_math",
            description="Basic mathematical operations",
            practice_increment_logit=0.1,
        ),
        Skill(
            name="addition",
            description="Addition operations",
            prerequisites={"parent_names": ["basic_math"], "dependence_model": "all"},
            practice_increment_logit=0.15,
        ),
        Skill(
            name="multiplication",
            description="Multiplication operations",
            prerequisites={"parent_names": ["addition"], "dependence_model": "all"},
            practice_increment_logit=0.2,
        ),
    ]
    return SkillSpace(skills=skills)


def test_skill_transfer_basic(hierarchical_skill_space):
    """Test that practicing a skill also benefits its parent skills."""
    student = Student(name="test_student", skill_space=hierarchical_skill_space)

    # Initialize all skills as learned with moderate levels
    student.set_skill_values(
        {
            "basic_math": 0.5,
            "addition": 0.6,
            "multiplication": 0.7,
        }
    )

    # Record initial skill levels
    initial_basic_math = student.skill_state["basic_math"].skill_level
    initial_addition = student.skill_state["addition"].skill_level
    initial_multiplication = student.skill_state["multiplication"].skill_level

    # Practice multiplication (should also benefit addition and basic_math)
    student.practice(hierarchical_skill_space.get_skill("multiplication"))

    # Check that all skills improved
    assert student.skill_state["multiplication"].skill_level > initial_multiplication
    assert student.skill_state["addition"].skill_level > initial_addition
    assert student.skill_state["basic_math"].skill_level > initial_basic_math

    # Check that the transfer effect is smaller than the direct effect
    multiplication_gain = (
        student.skill_state["multiplication"].skill_level - initial_multiplication
    )
    addition_gain = student.skill_state["addition"].skill_level - initial_addition
    basic_math_gain = student.skill_state["basic_math"].skill_level - initial_basic_math

    # The direct effect should be larger than transfer effects
    assert multiplication_gain > addition_gain
    assert multiplication_gain > basic_math_gain


def test_skill_transfer_no_parents(hierarchical_skill_space):
    """Test that practicing a skill with no parents only affects that skill."""
    student = Student(name="test_student", skill_space=hierarchical_skill_space)

    # Initialize skills
    student.set_skill_values(
        {
            "basic_math": 0.5,
            "addition": 0.6,
            "multiplication": 0.7,
        }
    )

    # Record initial skill levels
    initial_basic_math = student.skill_state["basic_math"].skill_level
    initial_addition = student.skill_state["addition"].skill_level
    initial_multiplication = student.skill_state["multiplication"].skill_level

    # Practice basic_math (has no parents)
    student.practice(hierarchical_skill_space.get_skill("basic_math"))

    # Check that only basic_math improved
    assert student.skill_state["basic_math"].skill_level > initial_basic_math
    assert student.skill_state["addition"].skill_level == initial_addition
    assert student.skill_state["multiplication"].skill_level == initial_multiplication


def test_skill_transfer_unlearned_parents(hierarchical_skill_space):
    """Test that transfer doesn't occur to unlearned parent skills."""
    student = Student(name="test_student", skill_space=hierarchical_skill_space)

    # Initialize skills, but don't learn basic_math
    student.set_skill_values(
        {
            "basic_math": 0.0,  # Not learned
            "addition": 0.6,
            "multiplication": 0.7,
        }
    )
    student.skill_state["basic_math"].learned = False
    student.skill_state["addition"].learned = True
    student.skill_state["multiplication"].learned = True

    # Record initial skill levels
    initial_basic_math = student.skill_state["basic_math"].skill_level
    initial_addition = student.skill_state["addition"].skill_level
    initial_multiplication = student.skill_state["multiplication"].skill_level

    # Practice multiplication
    student.practice(hierarchical_skill_space.get_skill("multiplication"))

    # Check that basic_math didn't improve (not learned)
    assert student.skill_state["basic_math"].skill_level == initial_basic_math
    # But addition should still improve
    assert student.skill_state["addition"].skill_level > initial_addition
    assert student.skill_state["multiplication"].skill_level > initial_multiplication


def test_transfer_factor_effect():
    """Test that the transfer factor (0.3) produces the expected effect."""
    # Create a simple skill space
    skills = [
        Skill(name="parent", practice_increment_logit=0.1),
        Skill(
            name="child",
            prerequisites={"parent_names": ["parent"], "dependence_model": "all"},
            practice_increment_logit=0.2,
        ),
    ]
    skill_space = SkillSpace(skills=skills)

    student = Student(name="test_student", skill_space=skill_space)
    student.set_skill_values({"parent": 0.5, "child": 0.6})

    # Practice the child skill
    student.practice(skill_space.get_skill("child"))

    # Calculate the expected transfer effect
    # Transfer increment = 0.2 * 0.3 = 0.06
    # Parent should increase by approximately this amount
    parent_gain = student.skill_state["parent"].skill_level - 0.5
    child_gain = student.skill_state["child"].skill_level - 0.6

    # The child gain should be larger than parent gain (direct vs transfer effect)
    assert child_gain > parent_gain
    # Parent gain should be positive but smaller
    assert parent_gain > 0
    assert parent_gain < child_gain


def test_recursive_ancestor_update(hierarchical_skill_space):
    """Test that practicing a skill updates all ancestors recursively with diminishing effects."""
    student = Student(name="test_student", skill_space=hierarchical_skill_space)

    # Initialize all skills as learned with moderate levels
    student.set_skill_values(
        {
            "basic_math": 0.5,
            "addition": 0.6,
            "multiplication": 0.7,
        }
    )

    # Record initial skill levels
    initial_basic_math = student.skill_state["basic_math"].skill_level
    initial_addition = student.skill_state["addition"].skill_level
    initial_multiplication = student.skill_state["multiplication"].skill_level

    # Practice multiplication (should benefit multiplication, addition, and basic_math)
    student.practice(hierarchical_skill_space.get_skill("multiplication"))

    # Check that all skills improved
    assert student.skill_state["multiplication"].skill_level > initial_multiplication
    assert student.skill_state["addition"].skill_level > initial_addition
    assert student.skill_state["basic_math"].skill_level > initial_basic_math

    # Check that the effects diminish with distance from the practiced skill
    multiplication_gain = (
        student.skill_state["multiplication"].skill_level - initial_multiplication
    )
    addition_gain = student.skill_state["addition"].skill_level - initial_addition
    basic_math_gain = student.skill_state["basic_math"].skill_level - initial_basic_math

    # The direct effect should be largest, then immediate parent, then grandparent
    assert multiplication_gain > addition_gain
    assert addition_gain > basic_math_gain

    # Verify the diminishing effect follows the expected pattern
    # multiplication: direct effect (full increment)
    # addition: 0.3 * increment (first level transfer)
    # basic_math: 0.3^2 * increment = 0.09 * increment (second level transfer)
    # So basic_math gain should be much smaller than addition gain
    assert basic_math_gain < addition_gain * 0.5  # Should be much smaller


def test_recursive_transfer_with_larger_hierarchy():
    """Test recursive transfer with a deeper skill hierarchy."""
    # Create a deeper skill hierarchy
    skills = [
        Skill(name="foundation", practice_increment_logit=0.1),
        Skill(
            name="level1",
            prerequisites={"parent_names": ["foundation"], "dependence_model": "all"},
            practice_increment_logit=0.15,
        ),
        Skill(
            name="level2",
            prerequisites={"parent_names": ["level1"], "dependence_model": "all"},
            practice_increment_logit=0.2,
        ),
        Skill(
            name="level3",
            prerequisites={"parent_names": ["level2"], "dependence_model": "all"},
            practice_increment_logit=0.25,
        ),
    ]
    skill_space = SkillSpace(skills=skills)

    student = Student(name="test_student", skill_space=skill_space)
    student.set_skill_values(
        {
            "foundation": 0.5,
            "level1": 0.6,
            "level2": 0.7,
            "level3": 0.8,
        }
    )

    # Record initial levels
    initial_levels = {
        name: student.skill_state[name].skill_level
        for name in ["foundation", "level1", "level2", "level3"]
    }

    # Practice the highest level skill
    student.practice(skill_space.get_skill("level3"))

    # Check that all skills improved
    for name in ["foundation", "level1", "level2", "level3"]:
        assert student.skill_state[name].skill_level > initial_levels[name]

    # Check that effects diminish with distance
    gains = {
        name: student.skill_state[name].skill_level - initial_levels[name]
        for name in ["foundation", "level1", "level2", "level3"]
    }

    # Effects should diminish: level3 > level2 > level1 > foundation
    assert gains["level3"] > gains["level2"]
    assert gains["level2"] > gains["level1"]
    assert gains["level1"] > gains["foundation"]

    # The foundation should have the smallest gain (0.3^3 = 0.027 of the original increment)
    assert gains["foundation"] < gains["level1"] * 0.5
