import pytest

from studentsimulator.activity_provider import Item
from studentsimulator.general import Misconception, Model, Skill


@pytest.mark.parametrize(
    "cls, kwargs",
    [
        (Model, {}),
        (Misconception, {"parent_skill": Skill()}),
        (Item, {"skill": Skill()}),
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
