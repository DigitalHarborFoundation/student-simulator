import pytest
from pydantic import ValidationError

from studentsimulator.activity_provider import Item
from studentsimulator.general import (
    Misconception,
    Model,
    PrerequisiteStructure,
    Skill,
    SkillSpace,
)


@pytest.mark.parametrize(
    "cls, kwargs",
    [
        (Model, {}),
        (Misconception, {"parent_skill": Skill(name="test"), "description": "test"}),
        (Item, {"skill": Skill(name="test")}),
    ],
)
def test_auto_increment_id(cls, kwargs):
    """Test that each instance of the class gets a unique auto-incremented ID."""
    obj1 = cls(**kwargs)
    obj2 = cls(**kwargs)
    obj3 = cls(**kwargs)
    assert obj1.id == 0, f"Expected id '0', got {obj1.id} for {cls.__name__}"
    assert obj2.id == 1, f"Expected id '1', got {obj2.id} for {cls.__name__}"
    assert obj3.id == 2, f"Expected id '2', got {obj3.id} for {cls.__name__}"


def test_item_single_parameters():
    """Test Item creation with single parameter values."""
    skill = Skill(name="test_skill")
    item = Item(
        skill=skill,
        difficulty_logit=1.5,
        guess=0.25,
        slip=0.15,
        discrimination=1.2,
        practice_effectiveness_logit=0.3,
    )

    assert item.difficulty_logit == 1.5
    assert item.guess == 0.25
    assert item.slip == 0.15
    assert item.discrimination == 1.2
    assert item.practice_effectiveness_logit == 0.3


def test_item_range_parameters():
    """Test Item creation with range parameters."""
    skill = Skill(name="test_skill")
    item = Item(
        skill=skill,
        difficulty_logit_range=(-2, 2),
        guess_range=(0.1, 0.3),
        slip_range=(0.05, 0.2),
        discrimination_range=(0.8, 1.5),
        practice_effectiveness_logit_range=(0.1, 0.5),
    )

    # Check that ranges are set and values are generated
    assert item.difficulty_logit_range == (-2, 2)
    assert item.guess_range == (0.1, 0.3)
    assert item.slip_range == (0.05, 0.2)
    assert item.discrimination_range == (0.8, 1.5)
    assert item.practice_effectiveness_logit_range == (0.1, 0.5)

    # Check that actual values are generated within ranges
    assert -2 <= item.difficulty_logit <= 2
    assert 0.1 <= item.guess <= 0.3
    assert 0.05 <= item.slip <= 0.2
    assert 0.8 <= item.discrimination <= 1.5
    assert 0.1 <= item.practice_effectiveness_logit <= 0.5


def test_item_mixed_parameters():
    """Test Item creation with mixed single and range parameters."""
    skill = Skill(name="test_skill")
    item = Item(
        skill=skill,
        difficulty_logit=0.0,  # single value
        guess_range=(0.1, 0.3),  # range
        slip=0.1,  # single value
        discrimination_range=(1.0, 2.0),  # range
        practice_effectiveness_logit=0.2,  # single value
    )

    assert item.difficulty_logit == 0.0
    assert item.guess_range == (0.1, 0.3)
    assert item.slip == 0.1
    assert item.discrimination_range == (1.0, 2.0)
    assert item.practice_effectiveness_logit == 0.2


def test_item_range_validation():
    """Test Item range validation."""
    skill = Skill(name="test_skill")

    # Test invalid range lengths
    # These tests check that invalid range lengths and bounds are caught by Pydantic validation.
    # They should raise a ValidationError due to the constraints in the Item model.

    # Invalid range length: only one value instead of two
    with pytest.raises(Exception) as excinfo:
        Item(skill=skill, difficulty_logit_range=(1,))
    assert "ValidationError" in str(
        type(excinfo.value)
    ) or "_pydantic_core.ValidationError" in str(type(excinfo.value))

    # Invalid range length: three values instead of two
    with pytest.raises(Exception) as excinfo:
        Item(skill=skill, guess_range=(0.1, 0.2, 0.3))
    assert "ValidationError" in str(
        type(excinfo.value)
    ) or "_pydantic_core.ValidationError" in str(type(excinfo.value))

    # Invalid range bounds: out of allowed range for difficulty
    with pytest.raises(Exception) as excinfo:
        Item(skill=skill, difficulty_logit_range=(5, 6))
    assert "ValidationError" in str(
        type(excinfo.value)
    ) or "_pydantic_core.ValidationError" in str(type(excinfo.value))

    # Invalid range bounds: out of allowed range for guess
    with pytest.raises(Exception) as excinfo:
        Item(skill=skill, guess_range=(0.6, 0.7))
    assert "ValidationError" in str(
        type(excinfo.value)
    ) or "_pydantic_core.ValidationError" in str(type(excinfo.value))

    # Invalid range bounds: out of allowed range for discrimination
    with pytest.raises(Exception) as excinfo:
        Item(skill=skill, discrimination_range=(4, 5))
    assert "ValidationError" in str(
        type(excinfo.value)
    ) or "_pydantic_core.ValidationError" in str(type(excinfo.value))


def test_item_default_values():
    """Test Item default parameter values."""
    skill = Skill(name="test_skill")
    item = Item(skill=skill)

    assert item.difficulty_logit == 0.0
    assert item.guess == 0.2
    assert item.slip == 0.1
    assert item.discrimination == 1.0
    assert item.practice_effectiveness_logit == 0.0


def test_skill_space_creation():
    """Test SkillSpace creation and validation."""
    skills = [Skill(name="skill1"), Skill(name="skill2"), Skill(name="skill3")]

    skill_space = SkillSpace(skills=skills)
    assert len(skill_space.skills) == 3
    assert skill_space.skills[0].name == "skill1"


def test_skill_space_duplicate_validation():
    """Test SkillSpace duplicate name validation."""
    skills = [
        Skill(name="skill1"),
        Skill(name="skill1"),  # duplicate name
    ]

    with pytest.raises(ValidationError, match="Skill names must be unique"):
        SkillSpace(skills=skills)


def test_skill_space_duplicate_code_validation():
    """Test SkillSpace duplicate code validation."""
    skills = [
        Skill(name="skill1", code="CODE1"),
        Skill(name="skill2", code="CODE1"),  # duplicate code
    ]

    with pytest.raises(ValidationError, match="Skill codes must be unique"):
        SkillSpace(skills=skills)


def test_prerequisite_structure():
    """Test PrerequisiteStructure creation and validation."""
    prereq = PrerequisiteStructure(
        parent_names=["skill1", "skill2"], dependence_model="all"
    )

    assert prereq.parent_names == ["skill1", "skill2"]
    assert prereq.dependence_model == "all"


def test_skill_with_prerequisites():
    """Test Skill creation with prerequisites."""
    prereq = PrerequisiteStructure(
        parent_names=["parent_skill"], dependence_model="all"
    )

    skill = Skill(
        name="child_skill",
        prerequisites=prereq,
        probability_of_learning_with_prerequisites=0.9,
        probability_of_learning_without_prerequisites=0.1,
    )

    assert skill.prerequisites.parent_names == ["parent_skill"]
    assert skill.prerequisites.dependence_model == "all"
    assert skill.probability_of_learning_with_prerequisites == 0.9
    assert skill.probability_of_learning_without_prerequisites == 0.1


def test_skill_validation():
    """Test Skill validation rules."""
    # Test that learning with prerequisites must be greater than without
    with pytest.raises(
        ValidationError,
        match="probability of learning with prerequisites must be greater",
    ):
        Skill(
            name="test_skill",
            probability_of_learning_with_prerequisites=0.1,
            probability_of_learning_without_prerequisites=0.9,
        )


def test_item_options_generation():
    """Test that Item automatically generates options if not provided."""
    skill = Skill(name="test_skill")
    item = Item(skill=skill)

    assert len(item.options) == 4
    assert all(option.label in ["A", "B", "C", "D"] for option in item.options)
    assert sum(1 for option in item.options if option.is_target) == 1
