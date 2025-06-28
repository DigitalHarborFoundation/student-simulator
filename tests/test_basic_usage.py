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
from studentsimulator.general import Skill
from studentsimulator.student import (
    Student,
    save_student_activity_to_csv,
    save_student_profile_to_csv,
)

# ---------------------------------------------------------------------------
# Helper fixtures / builders
# ---------------------------------------------------------------------------


@pytest.fixture()
def example_skills() -> list[Skill]:
    """Create the four example skills presented in `.dev/temp.py`."""

    raw_skills = [
        {
            "name": "number_recognition",
            "code": "CCSS.MATH.K.CC.A.3",
            "description": "Recognize and write numerals 0-20",
            "parents": [],
            "decay": 0.01,
        },
        {
            "name": "place_value_ones",
            "code": "CCSS.MATH.1.NBT.A.1",
            "description": "Understand that the two digits of a two-digit number represent amounts",
            "parents": ["number_recognition"],
            "decay": 0.02,
        },
        {
            "name": "place_value_tens",
            "code": "CCSS.MATH.1.NBT.A.2",
            "description": "Understand place value of tens",
            "parents": ["place_value_ones"],
            "decay": 0.02,
        },
        {
            "name": "addition_no_carry",
            "code": "CCSS.MATH.1.OA.A.1",
            "description": "Add within 20 without regrouping",
            "parents": ["number_recognition", "place_value_ones"],
            "decay": 0.03,
        },
    ]

    return [Skill(**spec) for spec in raw_skills]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_end_to_end_basic_usage(example_skills: list[Skill], tmp_path: Path):
    """Replicate the workflow from `.dev/temp.py` and validate outcomes."""

    # Make results deterministic where randomness is used by the simulator.
    random.seed(1234)

    # ------------------------------------------------------------------
    # Students with varying prior knowledge
    # ------------------------------------------------------------------
    s1 = Student(name="bob", skills=example_skills)

    s2 = Student(skills=example_skills).set_skill_values(
        {"number_recognition": 0.5, "place_value_ones": 0.3}
    )

    s3 = Student(skills=example_skills).set_skill_values(
        {
            "number_recognition": 0.8,
            "place_value_ones": 0.6,
            "place_value_tens": 0.4,
            "addition_no_carry": 0.9,
        }
    )

    # Ensure each student has a skill_state entry for every registered skill.
    for student in (s1, s2, s3):
        assert student.skill_state  # not None / empty
        assert set(student.skill_state.keys()) == {sk.name for sk in example_skills}

    # ------------------------------------------------------------------
    # Activity provider, item pool, and assessment
    # ------------------------------------------------------------------
    provider = ActivityProvider()
    provider.register_skills(example_skills)

    item_pool = provider.construct_item_pool(
        name="basic_arithmetic_item_pool",
        skills=example_skills,
        n_items_per_skill=20,
        difficulty=(-2, 2),
        guess=(0.1, 0.3),
        slip=(0.01, 0.2),
        descrimination=1.0,  # note: typo kept to match original code path
    )

    # The pool should contain exactly 4 * 20 items.
    assert len(item_pool.items) == 4 * 20

    assessment = provider.generate_fixed_form_assessment(
        n_items=20, item_pool=item_pool
    )

    assert len(assessment.items) == 20

    # ------------------------------------------------------------------
    # Students take the assessment
    # ------------------------------------------------------------------
    for student in (s1, s2, s3):
        results = student.take_test(assessment, timestamp=1)

        # A BehaviorEventCollection is appended to history and returned.
        assert results is student.history[-1]

        # One response per item.
        assert len(results.behavioral_events) == len(assessment.items)

        # Percent correct is calculated and lies in [0, 100].
        assert 0.0 <= results.percent_correct <= 100.0

    # ------------------------------------------------------------------
    # Persist outcomes to CSV and verify contents
    # ------------------------------------------------------------------
    profile_csv = tmp_path / "students.csv"
    activity_csv = tmp_path / "student_activity.csv"

    save_student_profile_to_csv([s1, s2, s3], filename=str(profile_csv))
    save_student_activity_to_csv([s1, s2, s3], filename=str(activity_csv))

    # The files should now exist and contain more than just the header.
    for csv_path in (profile_csv, activity_csv):
        assert csv_path.exists(), f"{csv_path} was not created."

        with csv_path.open() as f:
            rows = list(csv.reader(f))

        # Always at least header + one data row.
        assert len(rows) > 1, f"{csv_path} is empty other than header."

    # The activity file should have exactly 3 * 20 response rows (+1 header).
    with activity_csv.open() as f:
        rows = list(csv.reader(f))
    expected_rows = 1 + 3 * len(assessment.items)
    assert len(rows) == expected_rows, "Unexpected number of activity rows written."
