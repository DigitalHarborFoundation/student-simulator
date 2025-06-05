"""
Test the new two-tier event architecture:
1. Intervention Events - things that happen TO the student
2. Behavioral Events - things the student DOES (with potential feedback)
"""

from simlearn.config import (
    BehavioralEvent,
    BehaviorRepresentation,
    Feedback,
    InterventionEvent,
    Student,
)


def test_intervention_event_structure():
    """Test that intervention events properly represent things done TO students."""

    # Create different types of interventions
    lesson_event = InterventionEvent(
        student_id="alice",
        timestamp=100,
        intervention_type="lesson",
        target_skill="fractions",
        intervention_data={
            "lesson_type": "video",
            "duration_minutes": 15,
            "content_id": "fractions_intro_v2",
        },
        success_prob=0.85,
    )

    hint_event = InterventionEvent(
        student_id="alice",
        timestamp=200,
        intervention_type="hint",
        target_skill="fractions",
        intervention_data={
            "hint_text": "Remember: the denominator tells you how many parts the whole is divided into",
            "hint_level": 2,
        },
        success_prob=0.95,
    )

    # Verify structure
    assert lesson_event.intervention_type == "lesson"
    assert lesson_event.target_skill == "fractions"
    assert lesson_event.intervention_data["lesson_type"] == "video"

    assert hint_event.intervention_type == "hint"
    assert hint_event.intervention_data["hint_level"] == 2


def test_behavioral_event_structure():
    """Test that behavioral events properly represent student actions with feedback."""

    # Multiple choice item response
    mc_behavior = BehaviorRepresentation(
        behavior_type="item_selection", item_id="fraction_q1", selected_option="C"
    )

    mc_feedback = Feedback(
        feedback_type="binary",
        correct=True,
        feedback_text="Correct! 3/4 is indeed larger than 1/2.",
    )

    mc_event = BehavioralEvent(
        student_id="alice",
        timestamp=150,
        behavior=mc_behavior,
        feedback=mc_feedback,
        context={"attempt": 1, "time_spent_seconds": 45},
    )

    # Text response with rubric scoring
    text_behavior = BehaviorRepresentation(
        behavior_type="text_response",
        item_id="explain_fractions",
        text_response="A fraction represents a part of a whole. The top number shows how many parts you have, and the bottom number shows how many parts the whole is divided into.",
    )

    text_feedback = Feedback(
        feedback_type="rubric",
        score=8.5,
        max_score=10.0,
        rubric_scores={
            "conceptual_understanding": 9.0,
            "clarity": 8.0,
            "examples": 7.0,
        },
        feedback_text="Good explanation! Consider adding an example to strengthen your response.",
    )

    text_event = BehavioralEvent(
        student_id="alice",
        timestamp=300,
        behavior=text_behavior,
        feedback=text_feedback,
    )

    # Interactive behavior (no immediate feedback)
    interaction_behavior = BehaviorRepresentation(
        behavior_type="interaction",
        interaction_data={
            "activity_type": "fraction_builder",
            "pieces_selected": [1, 1, 1, 0],  # selected 3 out of 4 pieces
            "final_fraction": "3/4",
            "num_attempts": 2,
        },
    )

    interaction_event = BehavioralEvent(
        student_id="alice",
        timestamp=250,
        behavior=interaction_behavior,
        feedback=None,  # No immediate feedback for exploratory activity
        context={"session_id": "explore_fractions_001"},
    )

    # Verify structures
    assert mc_event.behavior.behavior_type == "item_selection"
    assert mc_event.behavior.selected_option == "C"
    assert mc_event.feedback.correct

    assert text_event.behavior.text_response.startswith("A fraction represents")
    assert text_event.feedback.rubric_scores["conceptual_understanding"] == 9.0

    assert interaction_event.behavior.interaction_data["final_fraction"] == "3/4"
    assert interaction_event.feedback is None


def test_student_event_recording():
    """Test that students properly record both types of events in their learning history."""

    student = Student(id="alice", name="Alice Johnson")

    # Create and record an intervention event
    intervention = InterventionEvent(
        student_id="alice",
        timestamp=100,
        intervention_type="lesson",
        target_skill="fractions",
        intervention_data={"lesson_id": "frac_101"},
    )

    student.record_intervention_event(intervention)

    # Create and record a behavioral event
    behavior = BehaviorRepresentation(
        behavior_type="item_selection", item_id="q1", selected_option="B"
    )

    feedback = Feedback(
        feedback_type="binary", correct=False, feedback_text="Incorrect. Try again!"
    )

    behavioral_event = BehavioralEvent(
        student_id="alice", timestamp=150, behavior=behavior, feedback=feedback
    )

    student.record_behavioral_event(behavioral_event)

    # Verify learning history
    assert len(student.learning_history) == 2

    # Check intervention event recording
    intervention_record = student.learning_history[0]
    assert intervention_record.event_type == "intervention_event"
    assert intervention_record.context["intervention_type"] == "lesson"
    assert intervention_record.context["target_skill"] == "fractions"

    # Check behavioral event recording
    behavioral_record = student.learning_history[1]
    assert behavioral_record.event_type == "behavioral_event"
    assert behavioral_record.context["behavior"]["behavior_type"] == "item_selection"
    assert not behavioral_record.context["feedback"]["correct"]


def test_realistic_learning_sequence():
    """Test a realistic sequence mixing interventions and behaviors."""

    student = Student(id="bob", name="Bob Smith")

    # Sequence: Lesson → Practice → Hint → More Practice → Assessment

    # 1. Student receives a lesson (intervention)
    lesson = InterventionEvent(
        student_id="bob",
        timestamp=0,
        intervention_type="lesson",
        target_skill="decimal_place_value",
        intervention_data={
            "lesson_type": "interactive_demo",
            "concepts_covered": ["tenths", "hundredths", "place_value"],
        },
    )
    student.record_intervention_event(lesson)

    # 2. Student attempts practice problem (behavior)
    practice_behavior = BehaviorRepresentation(
        behavior_type="item_selection",
        item_id="decimal_practice_1",
        selected_option="A",
    )

    practice_feedback = Feedback(
        feedback_type="binary",
        correct=False,
        feedback_text="Not quite. Remember that the first digit after the decimal point represents tenths.",
    )

    practice_event = BehavioralEvent(
        student_id="bob",
        timestamp=10,
        behavior=practice_behavior,
        feedback=practice_feedback,
    )
    student.record_behavioral_event(practice_event)

    # 3. System provides targeted hint (intervention)
    hint = InterventionEvent(
        student_id="bob",
        timestamp=11,
        intervention_type="hint",
        target_skill="decimal_place_value",
        intervention_data={
            "hint_text": "In 0.23, the 2 is in the tenths place and the 3 is in the hundredths place.",
            "trigger": "incorrect_answer",
        },
    )
    student.record_intervention_event(hint)

    # 4. Student tries again (behavior)
    retry_behavior = BehaviorRepresentation(
        behavior_type="item_selection",
        item_id="decimal_practice_1",
        selected_option="C",
    )

    retry_feedback = Feedback(
        feedback_type="binary",
        correct=True,
        feedback_text="Excellent! You correctly identified the tenths place.",
    )

    retry_event = BehavioralEvent(
        student_id="bob", timestamp=15, behavior=retry_behavior, feedback=retry_feedback
    )
    student.record_behavioral_event(retry_event)

    # 5. Student writes explanation (behavior)
    explanation_behavior = BehaviorRepresentation(
        behavior_type="text_response",
        item_id="explain_decimals",
        text_response="In a decimal number, the digits after the decimal point represent parts of a whole. The first digit is tenths, the second is hundredths, etc.",
    )

    explanation_feedback = Feedback(
        feedback_type="score",
        score=4.5,
        max_score=5.0,
        feedback_text="Good understanding! Your explanation clearly shows the place value concept.",
    )

    explanation_event = BehavioralEvent(
        student_id="bob",
        timestamp=25,
        behavior=explanation_behavior,
        feedback=explanation_feedback,
    )
    student.record_behavioral_event(explanation_event)

    # Verify the complete learning sequence
    assert len(student.learning_history) == 5

    event_types = [event.event_type for event in student.learning_history]
    expected = [
        "intervention_event",
        "behavioral_event",
        "intervention_event",
        "behavioral_event",
        "behavioral_event",
    ]
    assert event_types == expected

    # Verify progression shows learning
    first_attempt = student.learning_history[1].context["feedback"]["correct"]
    second_attempt = student.learning_history[3].context["feedback"]["correct"]
    assert not first_attempt
    assert second_attempt

    # Verify rich feedback data is preserved
    explanation_score = student.learning_history[4].context["feedback"]["score"]
    assert explanation_score == 4.5


if __name__ == "__main__":
    # Allow running directly to see the new event structure
    test_intervention_event_structure()
    test_behavioral_event_structure()
    test_student_event_recording()
    test_realistic_learning_sequence()
    print("✅ All event architecture tests passed!")
    print("\n🎯 New Event Architecture Successfully Implemented:")
    print("   📚 Intervention Events: Things that happen TO students")
    print("   🎭 Behavioral Events: Things students DO (with feedback)")
    print("   📝 Rich behavior representations and feedback models")
    print("   🔄 Complete learning history tracking")
