"""
Test the hybrid BKT/CDM/IRT learning model.

The model has three proficiency states:
1. No Learning: P(correct) = guess parameter
2. Post-Intervention: P(correct) = baseline proficiency, increases with practice
3. Mastery: P(correct) = 1 - slip parameter

Learning progression:
- Intervention → Probabilistic skill acquisition → Baseline proficiency
- Practice → Gradual improvement → Maximum proficiency (mastery)
- Item-specific practice effectiveness affects learning rate
"""

import numpy as np

from simlearn import Sim
from simlearn.config import InterventionEvent, SkillState, Student


def test_skill_state_probability_calculation():
    """Test that SkillState correctly calculates P(correct) for each proficiency state."""

    # Item parameters
    guess_param = 0.2
    slip_param = 0.1

    # Test 1: No learning state
    no_learning_skill = SkillState(
        skill_id="math", proficiency_state="no_learning", current_proficiency=0.0
    )
    prob = no_learning_skill.get_probability_correct(guess_param, slip_param)
    assert (
        prob == guess_param
    ), f"No learning should give guess parameter {guess_param}, got {prob}"

    # Test 2: Mastery state
    mastery_skill = SkillState(
        skill_id="math", proficiency_state="mastery", current_proficiency=1.0
    )
    prob = mastery_skill.get_probability_correct(guess_param, slip_param)
    expected = 1.0 - slip_param
    assert prob == expected, f"Mastery should give 1-slip={expected}, got {prob}"

    # Test 3: Baseline/intermediate state
    baseline_skill = SkillState(
        skill_id="math",
        proficiency_state="baseline",
        baseline_proficiency=0.6,
        current_proficiency=0.8,  # 80% progress toward mastery
        max_proficiency=1.0,
    )
    prob = baseline_skill.get_probability_correct(guess_param, slip_param)

    # Expected: linear interpolation between baseline (0.6) and mastery (0.9)
    # Progress: 0.8 / 1.0 = 0.8
    # P(correct) = 0.6 + 0.8 * (0.9 - 0.6) = 0.6 + 0.8 * 0.3 = 0.84
    expected = 0.6 + 0.8 * (0.9 - 0.6)
    assert (
        abs(prob - expected) < 0.001
    ), f"Baseline interpolation should give {expected}, got {prob}"


def test_intervention_probabilistic_learning():
    """Test that interventions probabilistically teach skills."""

    student = Student(id="alice")
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility

    # Create intervention with 80% learning probability
    intervention = InterventionEvent(
        student_id="alice",
        timestamp=100,
        intervention_type="lesson",
        target_skill="fractions",
        learning_probability=0.8,
        baseline_proficiency=0.6,
    )

    # Apply intervention multiple times to test probabilistic nature
    learning_outcomes = []
    for i in range(100):
        test_student = Student(id="alice")
        test_rng = np.random.default_rng(i)  # Different seed each time
        result = test_student.apply_intervention(intervention, test_rng)
        learning_outcomes.append(result)

    # Should learn approximately 80% of the time
    learning_rate = sum(learning_outcomes) / len(learning_outcomes)
    assert (
        0.7 < learning_rate < 0.9
    ), f"Learning rate should be around 0.8, got {learning_rate}"

    # Test specific learning outcome
    student = Student(id="alice")
    rng = np.random.default_rng(1)  # Seed that should trigger learning
    success = student.apply_intervention(intervention, rng)

    if success:
        skill_state = student.get_skill_state("fractions")
        assert skill_state.proficiency_state == "baseline"
        assert skill_state.baseline_proficiency == 0.6
        assert skill_state.current_proficiency == 0.6

        # Check learning history
        assert len(student.learning_history) == 1
        event = student.learning_history[0]
        assert event.event_type == "skill_learned"
        assert event.skill_changes["fractions"] == 0.6


def test_practice_effects():
    """Test that practice increases proficiency toward mastery."""

    student = Student(id="bob")

    # First, student needs to learn the skill via intervention
    intervention = InterventionEvent(
        student_id="bob",
        timestamp=10,
        intervention_type="lesson",
        target_skill="decimals",
        learning_probability=1.0,  # Guaranteed learning
        baseline_proficiency=0.5,
    )

    rng = np.random.default_rng(42)
    student.apply_intervention(intervention, rng)

    # Initial state after intervention
    skill_state = student.get_skill_state("decimals")
    assert skill_state.proficiency_state == "baseline"
    assert skill_state.current_proficiency == 0.5

    # Practice the skill multiple times
    practice_effectiveness = 0.2
    for i in range(3):
        student.practice_skill("decimals", practice_effectiveness, timestamp=20 + i)

    # Check proficiency increased
    assert skill_state.current_proficiency > 0.5
    expected_proficiency = min(0.5 + 3 * 0.2, 1.0)  # 0.5 + 0.6 = 1.0 (capped)
    assert abs(skill_state.current_proficiency - expected_proficiency) < 0.001

    # Check mastery achieved
    assert skill_state.proficiency_state == "mastery"
    assert skill_state.practice_count == 3

    # Check learning history recorded practice events
    practice_events = [
        e for e in student.learning_history if e.event_type == "skill_practiced"
    ]
    assert len(practice_events) == 3


def test_no_practice_without_learning():
    """Test that practice has no effect before skill is learned."""

    student = Student(id="charlie")

    # Try to practice skill without learning it first
    initial_state = student.get_skill_state("algebra")
    assert initial_state.proficiency_state == "no_learning"
    assert initial_state.current_proficiency == 0.0

    # Practice should have no effect
    student.practice_skill("algebra", practice_effectiveness=0.3, timestamp=50)

    # State should be unchanged
    final_state = student.get_skill_state("algebra")
    assert final_state.proficiency_state == "no_learning"
    assert final_state.current_proficiency == 0.0
    assert final_state.practice_count == 0  # No practice recorded

    # No learning events should be recorded
    assert len(student.learning_history) == 0


def test_complete_learning_progression():
    """Test complete learning journey: no learning → intervention → practice → mastery."""

    # Set up simulation with hybrid model
    skills = [{"id": "geometry", "type": "continuous"}]
    items = [
        {
            "id": "triangle_area",
            "title": "Calculate triangle area",
            "skills": [{"id": "geometry", "type": "continuous"}],
            "g": 0.25,  # 25% guess rate
            "s": 0.05,  # 5% slip rate
            "practice_effectiveness": 0.3,
        }
    ]
    student = Student(id="diana", name="Diana")

    sim = Sim(seed=123).skills(skills).items(items).population([student.model_dump()])

    # Learning progression events
    events = [
        # Phase 1: No learning - should perform at guess level
        {
            "student_id": "diana",
            "time": 1,
            "item_id": "triangle_area",
            "observed": True,
        },
        {
            "student_id": "diana",
            "time": 2,
            "item_id": "triangle_area",
            "observed": True,
        },
        # Phase 2: Intervention - probabilistic learning
        {
            "student_id": "diana",
            "time": 10,
            "item_id": "triangle_area",
            "observed": True,
            "intervention_type": "skill_boost",
            "context": {"target_skill": "geometry", "boost": 0.7},
        },
        # Phase 3: Practice - should improve toward mastery
        {
            "student_id": "diana",
            "time": 20,
            "item_id": "triangle_area",
            "observed": True,
        },
        {
            "student_id": "diana",
            "time": 21,
            "item_id": "triangle_area",
            "observed": True,
        },
        {
            "student_id": "diana",
            "time": 22,
            "item_id": "triangle_area",
            "observed": True,
        },
        {
            "student_id": "diana",
            "time": 23,
            "item_id": "triangle_area",
            "observed": True,
        },
    ]

    # Run simulation
    response_log, latent_state = sim.run(events)

    # Analyze results
    diana = latent_state["students"]["diana"]
    geometry_state = diana.get_skill_state("geometry")

    # Verify learning progression occurred
    assert geometry_state.proficiency_state in [
        "baseline",
        "mastery",
    ], "Should have learned the skill"
    assert geometry_state.current_proficiency > 0, "Should have some proficiency"

    # Check response patterns
    diana_responses = response_log[response_log["student_id"] == "diana"]

    # Early responses (no learning) should be around guess rate
    early_responses = diana_responses[diana_responses["time"] <= 5]
    if len(early_responses) > 0:
        early_performance = early_responses["response"].mean()
        # Should be around guess rate (0.25), allowing for randomness
        assert (
            0.0 <= early_performance <= 0.6
        ), f"Early performance should be low, got {early_performance}"

    # Later responses should show improvement
    later_responses = diana_responses[diana_responses["time"] >= 20]
    if len(later_responses) > 0:
        later_performance = later_responses["response"].mean()
        # Should be higher than early performance
        if len(early_responses) > 0:
            assert (
                later_performance >= early_performance
            ), "Performance should improve over time"

    # Verify learning history contains expected event types
    event_types = {event.event_type for event in diana.learning_history}
    expected_types = {"skill_learned", "skill_practiced", "behavioral_event"}
    assert expected_types.issubset(
        event_types
    ), f"Missing event types. Got: {event_types}"

    print("\nDiana's Learning Progression:")
    print(f"Final proficiency state: {geometry_state.proficiency_state}")
    print(f"Current proficiency: {geometry_state.current_proficiency:.3f}")
    print(f"Practice count: {geometry_state.practice_count}")
    print(
        f"Early performance: {early_performance:.2f}"
        if len(early_responses) > 0
        else "No early responses"
    )
    print(
        f"Later performance: {later_performance:.2f}"
        if len(later_responses) > 0
        else "No later responses"
    )
    print(f"Learning events: {len(diana.learning_history)}")


def test_item_specific_practice_effectiveness():
    """Test that different items have different practice effectiveness."""

    student = Student(id="eve")

    # Learn skill first
    intervention = InterventionEvent(
        student_id="eve",
        timestamp=10,
        intervention_type="tutorial",
        target_skill="statistics",
        learning_probability=1.0,
        baseline_proficiency=0.4,
    )

    rng = np.random.default_rng(42)
    student.apply_intervention(intervention, rng)

    initial_proficiency = student.get_skill_state("statistics").current_proficiency

    # Practice with high effectiveness item
    student.practice_skill("statistics", practice_effectiveness=0.4, timestamp=20)
    high_effect_proficiency = student.get_skill_state("statistics").current_proficiency

    # Reset to test low effectiveness
    student.get_skill_state("statistics").current_proficiency = initial_proficiency

    # Practice with low effectiveness item
    student.practice_skill("statistics", practice_effectiveness=0.1, timestamp=30)
    low_effect_proficiency = student.get_skill_state("statistics").current_proficiency

    # High effectiveness should produce greater improvement
    high_improvement = high_effect_proficiency - initial_proficiency
    low_improvement = low_effect_proficiency - initial_proficiency

    assert (
        high_improvement > low_improvement
    ), f"High effectiveness ({high_improvement}) should exceed low effectiveness ({low_improvement})"
    assert (
        abs(high_improvement - 0.4) < 0.001
    ), f"High effectiveness should increase by 0.4, got {high_improvement}"
    assert (
        abs(low_improvement - 0.1) < 0.001
    ), f"Low effectiveness should increase by 0.1, got {low_improvement}"


def test_proficiency_caps_at_maximum():
    """Test that proficiency cannot exceed maximum value."""

    student = Student(id="frank")

    # Learn skill with high baseline
    intervention = InterventionEvent(
        student_id="frank",
        timestamp=10,
        intervention_type="intensive_course",
        target_skill="calculus",
        learning_probability=1.0,
        baseline_proficiency=0.9,  # High starting proficiency
    )

    rng = np.random.default_rng(42)
    student.apply_intervention(intervention, rng)

    # Practice extensively
    for i in range(5):
        student.practice_skill("calculus", practice_effectiveness=0.3, timestamp=20 + i)

    skill_state = student.get_skill_state("calculus")

    # Should be capped at maximum
    assert skill_state.current_proficiency <= skill_state.max_proficiency
    assert (
        skill_state.current_proficiency == 1.0
    ), f"Should be capped at 1.0, got {skill_state.current_proficiency}"
    assert skill_state.proficiency_state == "mastery"


if __name__ == "__main__":
    # Allow running directly to see hybrid learning model in action
    test_skill_state_probability_calculation()
    test_intervention_probabilistic_learning()
    test_practice_effects()
    test_no_practice_without_learning()
    test_complete_learning_progression()
    test_item_specific_practice_effectiveness()
    test_proficiency_caps_at_maximum()

    print("✅ All hybrid learning model tests passed!")
    print("\n🧠 Hybrid BKT/CDM/IRT Model Successfully Implemented:")
    print("   📚 Three proficiency states: no_learning → baseline → mastery")
    print("   🎲 Probabilistic skill acquisition from interventions")
    print("   📈 Practice-driven improvement toward mastery")
    print("   🎯 Item-specific practice effectiveness")
    print("   🔒 Proficiency capped at maximum values")
    print("   🧮 P(correct) = guess | interpolated | (1-slip)")
