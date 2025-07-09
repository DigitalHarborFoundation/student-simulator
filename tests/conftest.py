import pytest

from studentsimulator.activity_provider import Item
from studentsimulator.general import Misconception, Model, Skill, SkillSpace


@pytest.fixture(autouse=True)
def reset_model_counters():
    # Reset the _counter for all relevant Model subclasses
    for cls in [Model, Skill, SkillSpace, Misconception, Item]:
        if hasattr(cls, "_counter"):
            cls._counter = 0
