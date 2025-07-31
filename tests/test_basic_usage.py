"""Tests covering the basic usage pattern shown in `.dev/temp.py`.

The goal is to execute the same high-level workflow end-to-end and assert
that it produces sensible artefacts (objects populated, csv files written,
etc.).  We purposely avoid asserting on any **exact** stochastic outcomes
because the simulator relies on randomness. Instead, we validate structural
properties (counts, ranges, presence of data, …) that must always hold.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import pytest

from studentsimulator.activity_provider import ActivityProvider
from studentsimulator.io import (
    save_student_daily_skill_states_to_csv,
    save_student_events_to_csv,
)
from studentsimulator.skill import Skill, SkillSpace
from studentsimulator.student import Student

# ---------------------------------------------------------------------------
# Helper fixtures / builders
# ---------------------------------------------------------------------------


@pytest.fixture()
def example_skill_space() -> SkillSpace:
    """Create the four example skills presented in `.dev/temp.py`."""

    raw_skills = [
        {
            "name": "number_recognition",
            "code": "CCSS.MATH.K.CC.A.3",
            "description": "Recognize and write numerals 0-20",
            "decay_logit": 0.01,  # natural temporal decay
            "initial_skill_level_after_learning": 0.3,
        },
        {
            "name": "place_value_ones",
            "code": "CCSS.MATH.1.NBT.A.1",
            "description": "Understand that the two digits of a two-digit number represent amounts",
            "prerequisites": {
                "parent_names": ["number_recognition"],
                "dependence_model": "all",  # all or any
            },
            "probability_of_learning_without_prerequisites": 0.1,  # probability of learning this skill without prerequisites
            "decay_logit": 0.02,
            "initial_skill_level_after_learning": 0.25,
        },
        {
            "name": "addition_no_carry",
            "code": "CCSS.MATH.1.OA.A.1",
            "description": "Add within 20 without regrouping",
            "prerequisites": {
                "parent_names": ["place_value_tens"],
                "dependence_model": "all",  # all or any
            },
            "decay_logit": 0.03,
            "probability_of_learning_without_prerequisites": 0.01,  # probability of learning this skill without prerequisites
            "initial_skill_level_after_learning": 0.2,
        },
        {
            "name": "place_value_tens",
            "code": "CCSS.MATH.1.NBT.A.2",
            "description": "Understand place value of tens",
            "prerequisites": {
                "parent_names": ["place_value_ones"],
                "dependence_model": "all",  # all or any
            },
            "decay_logit": 0.02,
            "probability_of_learning_without_prerequisites": 0.1,  # probability of learning this skill without prerequisites
            "initial_skill_level_after_learning": 0.15,
        },
    ]

    skills = [Skill(**spec) for spec in raw_skills]
    return SkillSpace(skills=skills)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_end_to_end_basic_usage(example_skill_space: SkillSpace, tmp_path: Path):
    """Replicate the workflow from `.dev/temp.py` and validate outcomes."""

    # Make results deterministic where randomness is used by the simulator.
    random.seed(1234)

    # ------------------------------------------------------------------
    # Students with varying prior knowledge
    # ------------------------------------------------------------------
    s1 = Student(name="bob", skill_space=example_skill_space)

    s2 = Student(skill_space=example_skill_space).set_skill_values(
        {"number_recognition": 0.5, "place_value_ones": 0.3}
    )

    s3 = Student(skill_space=example_skill_space).set_skill_values(
        {
            "number_recognition": 0.8,
            "place_value_ones": 0.6,
            "place_value_tens": 0.4,
            "addition_no_carry": 0.9,
        }
    )

    s4 = Student(skill_space=example_skill_space).randomly_initialize_skills(
        practice_count=[1, 9]
    )  # random

    # Ensure each student has a skill_state entry for every registered skill.
    for student in (s1, s2, s3, s4):
        assert student.skill_state  # not None / empty
        assert set(student.skill_state.keys()) == {
            sk.name for sk in example_skill_space.skills
        }

    # ------------------------------------------------------------------
    # Activity provider, item pool, and assessment
    # ------------------------------------------------------------------
    provider = ActivityProvider()
    provider.register_skills(example_skill_space)

    item_pool = provider.construct_item_pool(
        name="basic_arithmetic_item_pool",
        skills=example_skill_space.skills,
        n_items_per_skill=20,
        difficulty_logit_range=(-2, 2),
        guess_range=(0.1, 0.3),
        slip_range=(0.01, 0.2),
        discrimination_range=(1.0, 1.0),
    )

    # The pool should contain exactly 4 * 20 items.
    assert len(item_pool.items) == 4 * 20

    assessment = provider.generate_fixed_form_assessment(
        n_items=20, item_pool=item_pool, skills=example_skill_space
    )

    assert len(assessment.items) == 20

    # ------------------------------------------------------------------
    # Students take the assessment
    # ------------------------------------------------------------------
    for student in (s1, s2, s3, s4):
        results = provider.administer_fixed_form_assessment(
            student_or_students=student, test=assessment
        )
        # The results are now a list, so we need to get the first (and only) result
        test_result = results[0]

        # A BehaviorEventCollection is appended to history and returned.
        # The test_result is a BehaviorEventCollection, but get_individual_events() flattens it
        # So we check that the test_result's behavioral_events are in the individual events
        individual_events = student.skills.get_individual_events()
        test_events = test_result.behavioral_events
        # Check that all test events are in the individual events
        for test_event in test_events:
            assert (
                test_event in individual_events
            ), f"Test event {test_event.id} not found in individual events"

        # One response per item.
        assert len(test_result.behavioral_events) == len(assessment.items)

        # Percent correct is calculated and lies in [0, 100].
        assert 0.0 <= test_result.percent_correct <= 100.0

    # ------------------------------------------------------------------
    # Persist outcomes to CSV and verify contents
    # ------------------------------------------------------------------
    # Create outputs directory if it doesn't exist
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    daily_states_csv = outputs_dir / "students_daily_skill_states.csv"
    events_csv = outputs_dir / "student_events.csv"

    save_student_daily_skill_states_to_csv([s1, s2, s3], filename=str(daily_states_csv))
    save_student_events_to_csv([s1, s2, s3], filename=str(events_csv))

    # The files should now exist and contain more than just the header.
    for csv_path in (daily_states_csv, events_csv):
        assert csv_path.exists(), f"{csv_path} was not created."

        with csv_path.open() as f:
            rows = list(csv.reader(f))

        # Always at least header + one data row.
        assert len(rows) > 1, f"{csv_path} is empty other than header."

    # The events file should have exactly 3 * 20 response rows (+1 header).
    with events_csv.open() as f:
        rows = list(csv.reader(f))
    expected_rows = 1 + 3 * len(assessment.items)
    assert len(rows) == expected_rows, "Unexpected number of event rows written."
