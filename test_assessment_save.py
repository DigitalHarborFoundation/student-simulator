#!/usr/bin/env python3
"""Test to verify fixed form assessment responses are saved"""

import csv

from studentsimulator.activity_provider import ActivityProvider
from studentsimulator.factory import create_random_students
from studentsimulator.io import save_student_events_to_csv
from studentsimulator.skill import Skill, SkillSpace

# Create simple setup
skills = [
    Skill(name="skill1", code="S1", description="Skill 1"),
    Skill(name="skill2", code="S2", description="Skill 2"),
]
skill_space = SkillSpace(skills=skills)

# Create student
students = create_random_students(
    skill_space=skill_space, n_students=1, practice_count=[1, 2]
)
student = students[0]

# Create activity provider
activity_provider = ActivityProvider()
activity_provider.register_skills(skill_space)

# Create item pool
item_pool = activity_provider.construct_item_pool(
    name="test_pool",
    skills=skills,
    n_items_per_skill=10,
)

print(f"Created item pool with {len(item_pool.items)} items")

# Create and administer fixed form assessment
assessment = activity_provider.generate_fixed_form_assessment(
    n_items=5, item_pool=item_pool, skills=skills  # Small assessment
)

print(f"Assessment has {len(assessment.items)} items")
print("Student events before assessment:", len(student.skills.get_individual_events()))

# Administer the assessment
activity_provider.administer_fixed_form_assessment(student, assessment)

print("Student events after assessment:", len(student.skills.get_individual_events()))

# Check events
all_events = student.skills.get_individual_events()
response_events = [e for e in all_events if type(e).__name__ == "ItemResponseEvent"]
print(f"ItemResponseEvent count: {len(response_events)}")

# Now test the save function
print("\n=== Testing save_student_events_to_csv ===")

# Save without activity provider name filter
save_student_events_to_csv(
    students=[student],
    filename="test_all_events.csv",
    observation_rate=1.0,  # Save all events
)

# Save with activity provider name filter
save_student_events_to_csv(
    students=[student],
    filename="test_filtered_events.csv",
    activity_provider_name=activity_provider.name,
    observation_rate=1.0,  # Save all events
)


def count_csv_rows(filename):
    with open(filename, "r") as f:
        reader = csv.reader(f)
        return sum(1 for row in reader) - 1  # Subtract header


all_count = count_csv_rows("test_all_events.csv")
filtered_count = count_csv_rows("test_filtered_events.csv")

print(f"Events in CSV without filter: {all_count}")
print(f"Events in CSV with filter: {filtered_count}")
print(f"Expected ItemResponseEvent count: {len(response_events)}")
