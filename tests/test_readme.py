"""Test that README code examples work correctly."""

import os
import tempfile


def test_readme_quick_start_example():
    """Test the Quick Start code example from README.md."""
    from studentsimulator.activity_provider import ActivityProvider
    from studentsimulator.factory import create_random_students
    from studentsimulator.io import (
        save_student_daily_skill_states_to_csv,
        save_student_events_to_csv,
    )
    from studentsimulator.skill import Skill, SkillSpace
    from studentsimulator.student import Student

    # Define skills with prerequisites and learning parameters
    skills = [
        Skill(
            name="counting",
            prerequisites={"parent_names": []},
            practice_increment_logit=0.1,
            initial_skill_level_after_learning=0.5,
        ),
        Skill(
            name="addition",
            prerequisites={"parent_names": ["counting"]},
            practice_increment_logit=0.12,
            initial_skill_level_after_learning=0.4,
        ),
        Skill(
            name="multiplication",
            prerequisites={"parent_names": ["addition"]},
            practice_increment_logit=0.15,
            initial_skill_level_after_learning=0.3,
        ),
    ]

    skill_space = SkillSpace(skills=skills)

    # Create students
    student = Student(name="alice", skill_space=skill_space)
    student.randomly_initialize_skills(practice_count=[3, 10])

    # Or create multiple random students at once
    students = create_random_students(
        skill_space=skill_space, n_students=5, practice_count=[1, 20]
    )

    # Create activity provider and assessment
    provider = ActivityProvider()
    provider.register_skills(skill_space)
    item_pool = provider.construct_item_pool(
        name="math_pool",
        skills=skill_space.skills,
        n_items_per_skill=20,
        difficulty_logit_range=(-2, 2),
        guess_range=(0.1, 0.3),
        slip_range=(0.01, 0.2),
        discrimination_range=(1.0, 2.0),
    )

    # Administer lesson and practice
    provider.administer_lesson(student=student, skill=skill_space.skills[0])
    provider.administer_practice(
        student, skill=skill_space.skills[0], n_items=5, item_pool=item_pool
    )

    # Create and administer assessment
    assessment = provider.generate_fixed_form_assessment(
        n_items=10, item_pool=item_pool, skills=skill_space.skills
    )
    results = provider.administer_fixed_form_assessment(student, assessment)
    print(f"{student.name}: {results[0].percent_correct:.1f}% correct")

    # Save results in temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        save_student_daily_skill_states_to_csv(
            [student], os.path.join(tmpdir, "students_daily_skill_states.csv")
        )
        save_student_events_to_csv(
            [student],
            os.path.join(tmpdir, "student_events.csv"),
            activity_provider_name=provider.name,
            train_val_test_split=(0.8, 0.0, 0.2),
            observation_rate=0.05,
        )

        # Verify files were created
        assert os.path.exists(os.path.join(tmpdir, "students_daily_skill_states.csv"))
        assert os.path.exists(os.path.join(tmpdir, "student_events.csv"))

    # Basic assertions to ensure the code ran correctly
    assert student.name == "alice"
    assert len(skill_space.skills) == 3
    assert len(students) == 5
    assert assessment is not None
    assert results is not None
    assert hasattr(results[0], "percent_correct")


def test_readme_skill_transfer_example():
    """Test the Skill Transfer code example from README.md."""
    from studentsimulator.skill import Skill, SkillSpace
    from studentsimulator.student import Student

    # Create skills with prerequisites
    skills = [
        Skill(name="counting"),
        Skill(name="addition", prerequisites={"parent_names": ["counting"]}),
        Skill(name="multiplication", prerequisites={"parent_names": ["addition"]}),
    ]
    skill_space = SkillSpace(skills=skills)

    # Create and initialize student
    student = Student(name="test_student", skill_space=skill_space)
    student.randomly_initialize_skills(practice_count=[5, 10])

    # Get initial skill levels
    initial_mult = student.skills["multiplication"].skill_level

    # Practice multiplication (benefits multiplication, addition, and counting)
    student.practice(skill_space.get_skill("multiplication"))

    # Check that all skills improved with diminishing effects
    print(f"Multiplication: {student.skills['multiplication'].skill_level}")
    print(f"Addition: {student.skills['addition'].skill_level}")
    print(f"Counting: {student.skills['counting'].skill_level}")

    # Verify skill transfer occurred (if skills were learned)
    if student.skills["multiplication"].learned:
        assert student.skills["multiplication"].skill_level >= initial_mult
        # Parent skills may have improved due to transfer
        # but this depends on whether they were learned


def test_readme_dual_history_example():
    """Test the Dual-History System code example from README.md."""
    from studentsimulator.analytics import plot_skill_trajectory
    from studentsimulator.skill import Skill, SkillSpace
    from studentsimulator.student import Student

    # Create a simple skill space
    skills = [Skill(name="math"), Skill(name="reading")]
    skill_space = SkillSpace(skills=skills)

    # Create and initialize student
    student = Student(name="test_student", skill_space=skill_space)
    student.randomly_initialize_skills(practice_count=[3, 8])

    # Plot skill trajectories over time (in temp directory)
    with tempfile.TemporaryDirectory() as tmpdir:
        plot_file = os.path.join(tmpdir, "learning_trajectory.png")
        plot_skill_trajectory(student, filename=plot_file)
        assert os.path.exists(plot_file)

    # Simulate time passing with forgetting
    initial_levels = {
        s.name: student.skills[s.name].skill_level for s in skill_space.skills
    }
    student.wait(days=7)  # Skills decay over time

    # Verify wait created an event and potentially caused forgetting
    assert student.days_since_initialization >= 7

    # Check that skills may have decayed (if they were learned)
    for skill in skill_space.skills:
        if student.skills[skill.name].learned:
            # Forgetting should occur for learned skills
            assert (
                student.skills[skill.name].skill_level
                <= initial_levels[skill.name] + 0.01
            )  # Small tolerance for rounding


if __name__ == "__main__":
    test_readme_quick_start_example()
    test_readme_skill_transfer_example()
    test_readme_dual_history_example()
    print("All README examples work correctly!")
