import pytest

from studentsimulator.activity_provider import ActivityProvider
from studentsimulator.item import Item, ItemPool
from studentsimulator.skill import Skill, SkillSpace
from studentsimulator.student import Student


@pytest.fixture
def simple_skill_space():
    """Create a simple skill space with one skill."""
    skill = Skill(
        name="math_basics",
        code="MATH.001",
        description="Basic mathematics",
        practice_increment_logit=0.1,
        initial_skill_level_after_learning=0.5,
    )
    return SkillSpace(skills=[skill])


@pytest.fixture
def activity_provider(simple_skill_space):
    """Create an activity provider with the simple skill space."""
    provider = ActivityProvider()
    provider.register_skills(simple_skill_space)
    return provider


@pytest.fixture
def student(simple_skill_space):
    """Create a student with the simple skill space."""
    student = Student(skill_space=simple_skill_space)
    # Learn the skill first so practice can be effective
    student.learn(simple_skill_space.skills[0])
    return student


@pytest.fixture
def test_item(simple_skill_space):
    """Create a test item for the skill."""
    return Item(
        skill=simple_skill_space.skills[0],
        difficulty_logit=0.0,
        discrimination=1.0,
        guess=0.1,
        slip=0.05,
    )


def test_practice_with_item_appends_to_history(student, test_item):
    """Test that practice with a specific item appends the item and score to history."""
    # Get initial history length
    initial_history_length = len(student.skills.get_individual_events())

    # Practice with the specific item
    student.practice(skill=test_item.skill, item=test_item)

    # Check that history length increased
    assert len(student.skills.get_individual_events()) == initial_history_length + 1

    # Get the last event
    last_event = student.skills.get_individual_events()[-1]

    # Verify it's an ItemResponseEvent
    assert hasattr(last_event, "item")
    assert hasattr(last_event, "score")
    assert hasattr(last_event, "feedback_given")

    # Verify the item is the one we practiced with
    assert last_event.item == test_item

    # Verify score is recorded (should be 0 or 1)
    assert last_event.score in [0, 1]

    # Verify feedback was given (practice provides feedback)
    assert last_event.feedback_given is True


def test_practice_without_item_does_not_append_item(student):
    """Test that practice without an item doesn't append an item to history."""
    # Get initial history length
    initial_history_length = len(student.skills.get_individual_events())

    # Practice without a specific item
    student.practice(skill=student.skill_space.skills[0])

    # Check that history length increased
    assert len(student.skills.get_individual_events()) == initial_history_length + 1

    # Get the last event
    last_event = student.skills.get_individual_events()[-1]

    # Verify it's an ItemResponseEvent
    assert hasattr(last_event, "item")
    assert hasattr(last_event, "score")

    # Verify no item was recorded
    assert last_event.item is None

    # Verify score is still recorded (should be None for practice without item)
    assert last_event.score is None


def test_practice_with_item_records_prob_correct(student, test_item):
    """Test that practice with an item records the probability of correct response."""
    # Practice with the specific item
    student.practice(skill=test_item.skill, item=test_item)

    # Get the last event
    last_event = student.skills.get_individual_events()[-1]

    # Verify prob_correct is recorded
    assert hasattr(last_event, "prob_correct")
    assert last_event.prob_correct is not None
    assert 0.0 <= last_event.prob_correct <= 1.0


def test_multiple_practice_sessions_with_items(student, test_item):
    """Test that multiple practice sessions with items are all recorded."""
    # Get initial event count (includes learning event from fixture)
    initial_count = len(student.skills.get_individual_events())

    # Practice multiple times
    for i in range(3):
        student.practice(skill=test_item.skill, item=test_item)

    # Check that we have 3 new events added to the initial count
    assert len(student.skills.get_individual_events()) == initial_count + 3

    # Check that all practice events have the item
    practice_events = student.skills.get_individual_events()[-3:]  # Last 3 events
    for event in practice_events:
        assert event.item == test_item
        assert event.score in [0, 1]
        assert event.feedback_given is True


def test_practice_with_different_items(student, simple_skill_space):
    """Test that practice with different items records different items."""
    # Get initial event count (includes learning event from fixture)
    initial_count = len(student.skills.get_individual_events())

    # Create two different items
    item1 = Item(
        skill=simple_skill_space.skills[0],
        difficulty_logit=-0.5,
        discrimination=1.0,
        guess=0.1,
        slip=0.05,
    )
    item2 = Item(
        skill=simple_skill_space.skills[0],
        difficulty_logit=0.5,
        discrimination=1.0,
        guess=0.1,
        slip=0.05,
    )

    # Practice with both items
    student.practice(skill=simple_skill_space.skills[0], item=item1)
    student.practice(skill=simple_skill_space.skills[0], item=item2)

    # Check that we have 2 new events added to the initial count
    assert len(student.skills.get_individual_events()) == initial_count + 2

    # Check that the items are different
    events = student.skills.get_individual_events()[-2:]  # Last 2 events
    assert events[0].item == item1
    assert events[1].item == item2
    assert events[0].item != events[1].item


def test_practice_with_pool_by_name(student, simple_skill_space):
    """Test that we can practice using an item pool specified by name."""
    skill = simple_skill_space.skills[0]

    # Create activity provider and register skill
    provider = ActivityProvider()
    provider.register_skills(simple_skill_space)

    # Create item pool and register it
    pool = ItemPool(
        name="test_pool",
        items=[
            Item(
                skill=skill,
                difficulty_logit=0.0,
                discrimination=1.0,
                guess=0.2,
                slip=0.1,
            )
            for _ in range(3)
        ],
    )
    provider.item_pools["test_pool"] = pool

    # Learn the skill first
    student.learn(skill)

    # Practice using pool name
    provider.administer_practice(student, skill, n_items=2, item_pool="test_pool")

    # Should have 2 practice events in history
    practice_events = [
        ev
        for ev in student.skills.get_individual_events()
        if hasattr(ev, "feedback_given") and ev.feedback_given
    ]
    assert len(practice_events) == 2

    # Events should have items from our pool
    for ev in practice_events:
        assert ev.item in pool
