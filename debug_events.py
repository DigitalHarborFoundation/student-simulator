#!/usr/bin/env python3
"""Debug script to check what events are being created and their activity provider names"""

from studentsimulator.activity_provider import ActivityProvider
from studentsimulator.factory import create_random_students
from studentsimulator.skill import Skill, SkillSpace

# Create simple setup
skill = Skill(name="test_skill", code="TEST", description="Test skill")
skill_space = SkillSpace(skills=[skill])

# Create student
students = create_random_students(
    skill_space=skill_space, n_students=1, practice_count=[1, 2]
)
student = students[0]

# Create activity provider
activity_provider = ActivityProvider()
activity_provider.register_skills(skill_space)

print(f"ActivityProvider name: '{activity_provider.name}'")

# Create item pool
item_pool = activity_provider.construct_item_pool(
    name="test_pool",
    skills=[skill],
    n_items_per_skill=5,
)

print(
    f"Sample item activity_provider_name: '{item_pool.items[0].activity_provider_name}'"
)

# Create fixed form assessment
assessment = activity_provider.generate_fixed_form_assessment(
    n_items=3, item_pool=item_pool, skills=[skill]
)

print(f"Assessment items count: {len(assessment.items)}")

# Administer assessment
print("\n=== Before assessment ===")
print(f"Student events count: {len(student.skills.get_individual_events())}")

activity_provider.administer_fixed_form_assessment(student, assessment)

print("\n=== After assessment ===")
all_events = student.skills.get_individual_events()
print(f"Student events count: {len(all_events)}")

# Check activity provider names in events
for i, event in enumerate(all_events):
    if hasattr(event, "activity_provider_name"):
        print(
            f"Event {i}: {type(event).__name__} - activity_provider_name: '{event.activity_provider_name}'"
        )
    else:
        print(
            f"Event {i}: {type(event).__name__} - no activity_provider_name attribute"
        )

# Test the filtering logic from save_student_events_to_csv
print("\n=== Testing filter logic ===")
print(f"Filter by activity_provider_name='{activity_provider.name}'")

filtered_count = 0
for event in all_events:
    if (
        hasattr(event, "activity_provider_name")
        and event.activity_provider_name == activity_provider.name
    ):
        filtered_count += 1

print(f"Events that would pass filter: {filtered_count}")
