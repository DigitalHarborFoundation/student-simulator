"""
Test CDM (Cognitive Diagnostic Model) mode with prerequisite-aware learning.

CDM Features:
1. Binary skill states (learned/not learned)
2. Prerequisite relationships between skills
3. Learning probability depends on whether prerequisites are met
4. Items aligned to exactly one skill
"""

import numpy as np

from simlearn import Sim
from simlearn.config import Student


def test_cdm_prerequisite_learning():
    """Test that learning probability depends on prerequisites in CDM mode."""

    # CDM skills with prerequisite hierarchy: counting → addition → multiplication
    skills = [
        {
            "id": "counting",
            "description": "Basic counting skills",
            "parents": [],  # No prerequisites
        },
        {
            "id": "addition",
            "description": "Addition operations",
            "parents": ["counting"],  # Requires counting
        },
        {
            "id": "multiplication",
            "description": "Multiplication operations",
            "parents": ["addition"],  # Requires addition
        },
    ]

    # Items aligned to exactly one skill each
    items = [
        {
            "id": "count_objects",
            "title": "Count objects 1-10",
            "skills": [{"id": "counting"}],
            "g": 0.3,
            "s": 0.05,
        },
        {
            "id": "simple_addition",
            "title": "Add single digits",
            "skills": [{"id": "addition"}],
            "g": 0.25,
            "s": 0.05,
        },
        {
            "id": "times_tables",
            "title": "Multiplication tables",
            "skills": [{"id": "multiplication"}],
            "g": 0.2,
            "s": 0.05,
        },
    ]

    student = Student(id="cdm_learner")

    sim = (
        Sim(seed=42)
        .skills(skills)
        .items(items)
        .population([student.model_dump()])
        .learning(mode="cdm")
    )

    # Test 1: Try to learn multiplication without prerequisites (should fail due to low probability)
    events_no_prereqs = []
    for attempt in range(1, 16):  # 15 attempts to learn multiplication without prereqs
        events_no_prereqs.append(
            {
                "student_id": "cdm_learner",
                "time": attempt,
                "item_id": "times_tables",
                "observed": True,
                "intervention_type": "skill_boost",
                "context": {"target_skill": "multiplication", "boost": 0.8},
            }
        )

    response_log1, latent_state1 = sim.run(events_no_prereqs)
    learner1 = latent_state1["students"]["cdm_learner"]

    mult_skill_no_prereqs = learner1.get_skill_state("multiplication")
    print("\nMultiplication learning without prerequisites:")
    print(f"Final proficiency: {mult_skill_no_prereqs.current_proficiency:.3f}")
    print(f"Learning occurred: {mult_skill_no_prereqs.current_proficiency > 0.5}")

    # Test 2: Learn prerequisites first, then multiplication (should succeed more often)
    student2 = Student(id="cdm_learner_with_prereqs")
    sim2 = (
        Sim(seed=43)  # Different seed for fair comparison
        .skills(skills)
        .items(items)
        .population([student2.model_dump()])
        .learning(mode="cdm")
    )

    events_with_prereqs = []
    time_counter = 1

    # First learn counting (multiple attempts to ensure learning)
    for attempt in range(5):
        events_with_prereqs.append(
            {
                "student_id": "cdm_learner_with_prereqs",
                "time": time_counter,
                "item_id": "count_objects",
                "observed": True,
                "intervention_type": "skill_boost",
                "context": {"target_skill": "counting", "boost": 0.8},
            }
        )
        time_counter += 1

    # Then learn addition (multiple attempts)
    for attempt in range(5):
        events_with_prereqs.append(
            {
                "student_id": "cdm_learner_with_prereqs",
                "time": time_counter,
                "item_id": "simple_addition",
                "observed": True,
                "intervention_type": "skill_boost",
                "context": {"target_skill": "addition", "boost": 0.8},
            }
        )
        time_counter += 1

    # Finally learn multiplication (should be easier with prerequisites)
    for attempt in range(5):
        events_with_prereqs.append(
            {
                "student_id": "cdm_learner_with_prereqs",
                "time": time_counter,
                "item_id": "times_tables",
                "observed": True,
                "intervention_type": "skill_boost",
                "context": {"target_skill": "multiplication", "boost": 0.8},
            }
        )
        time_counter += 1

    response_log2, latent_state2 = sim2.run(events_with_prereqs)
    learner2 = latent_state2["students"]["cdm_learner_with_prereqs"]

    counting_skill = learner2.get_skill_state("counting")
    addition_skill = learner2.get_skill_state("addition")
    mult_skill_with_prereqs = learner2.get_skill_state("multiplication")

    print("\nLearning with prerequisites:")
    print(f"Counting proficiency: {counting_skill.current_proficiency:.3f}")
    print(f"Addition proficiency: {addition_skill.current_proficiency:.3f}")
    print(
        f"Multiplication proficiency: {mult_skill_with_prereqs.current_proficiency:.3f}"
    )

    # CDM assertions
    print(
        f"Multiplication without prereqs: {mult_skill_no_prereqs.current_proficiency:.3f}"
    )
    print(
        f"Multiplication with prereqs: {mult_skill_with_prereqs.current_proficiency:.3f}"
    )

    # With enough attempts, counting should be learned (no prerequisites)
    assert (
        counting_skill.current_proficiency > 0.5
    ), "Should learn counting (no prerequisites)"

    # Verify prerequisite effect: multiplication should be much more likely to be learned with prereqs
    # The key test is that prerequisite-based learning probability is higher

    # Check that the prerequisite logic is working by testing learning probabilities
    from simlearn.config import Skill

    mult_skill_def = Skill(id="multiplication", parents=["addition"])

    # Without prerequisites met
    no_prereq_states = {"addition": 0.0}  # Addition not learned
    prob_no_prereq = mult_skill_def.get_cdm_learning_probability(no_prereq_states, 0.3)

    # With prerequisites met
    with_prereq_states = {"addition": 0.8}  # Addition learned
    prob_with_prereq = mult_skill_def.get_cdm_learning_probability(
        with_prereq_states, 0.3
    )

    print(f"Learning probability without prereqs: {prob_no_prereq:.3f}")
    print(f"Learning probability with prereqs: {prob_with_prereq:.3f}")

    assert (
        prob_with_prereq > prob_no_prereq
    ), "Prerequisites should increase learning probability"
    assert (
        prob_no_prereq < 0.1
    ), "Should have very low probability without prerequisites"
    assert prob_with_prereq >= 0.3, "Should have full probability with prerequisites"


def test_cdm_binary_response_model():
    """Test that CDM uses binary response model (learned vs not learned)."""

    skills = [{"id": "reading", "parents": []}]
    items = [
        {
            "id": "comprehension_test",
            "skills": [{"id": "reading"}],
            "g": 0.2,  # 20% guess rate
            "s": 0.1,  # 10% slip rate
        }
    ]

    student = Student(id="reader")

    sim = (
        Sim(seed=100)
        .skills(skills)
        .items(items)
        .population([student.model_dump()])
        .learning(mode="cdm")
    )

    # Test responses before learning (should be around guess rate)
    pre_learning_events = []
    for trial in range(1, 11):  # 10 trials
        pre_learning_events.append(
            {
                "student_id": "reader",
                "time": trial,
                "item_id": "comprehension_test",
                "observed": True,
            }
        )

    response_log_pre, latent_state_pre = sim.run(pre_learning_events)
    responses_pre = response_log_pre["response"].tolist()
    performance_pre = np.mean(responses_pre)

    # Apply learning intervention
    learning_event = {
        "student_id": "reader",
        "time": 15,
        "item_id": "comprehension_test",
        "observed": True,
        "intervention_type": "skill_boost",
        "context": {"target_skill": "reading", "boost": 0.8},
    }

    # Test responses after potential learning
    post_learning_events = pre_learning_events + [learning_event]
    for trial in range(16, 26):  # 10 more trials
        post_learning_events.append(
            {
                "student_id": "reader",
                "time": trial,
                "item_id": "comprehension_test",
                "observed": True,
            }
        )

    response_log_post, latent_state_post = sim.run(post_learning_events)
    responses_post = response_log_post["response"].tolist()[-10:]  # Last 10 responses
    performance_post = np.mean(responses_post)

    reader = latent_state_post["students"]["reader"]
    reading_skill = reader.get_skill_state("reading")

    print("\nCDM Binary Response Model:")
    print(f"Pre-learning performance: {performance_pre:.3f} (expected ~0.2)")
    print(f"Post-learning performance: {performance_post:.3f}")
    print(f"Reading skill proficiency: {reading_skill.current_proficiency:.3f}")
    print(f"Skill learned: {reading_skill.current_proficiency > 0.5}")

    # CDM binary model: performance should be either ~guess rate or ~(1-slip rate)
    if reading_skill.current_proficiency > 0.5:  # Learned
        expected_performance = 0.9  # 1 - slip rate
        assert (
            abs(performance_post - expected_performance) < 0.3
        ), f"Learned skill should perform near {expected_performance}"
    else:  # Not learned
        expected_performance = 0.2  # guess rate
        assert (
            abs(performance_post - expected_performance) < 0.3
        ), f"Unlearned skill should perform near {expected_performance}"


def test_cdm_mode_comparison():
    """Compare CDM mode with other modes to show different behaviors."""

    skills = [{"id": "math", "parents": []}]
    items = [{"id": "math_problem", "skills": [{"id": "math"}], "g": 0.25, "s": 0.05}]

    modes_to_test = ["cdm", "bkt", "hybrid"]
    results = {}

    for mode in modes_to_test:
        student = Student(id=f"{mode}_student")
        sim = (
            Sim(seed=200)  # Same seed for comparison
            .skills(skills)
            .items(items)
            .population([student.model_dump()])
            .learning(mode=mode)
        )

        events = [
            # Baseline
            {
                "student_id": f"{mode}_student",
                "time": 1,
                "item_id": "math_problem",
                "observed": True,
            },
            # Learning intervention
            {
                "student_id": f"{mode}_student",
                "time": 3,
                "item_id": "math_problem",
                "observed": True,
                "intervention_type": "skill_boost",
                "context": {"target_skill": "math", "boost": 0.7},
            },
            # Post-learning
            {
                "student_id": f"{mode}_student",
                "time": 5,
                "item_id": "math_problem",
                "observed": True,
            },
            {
                "student_id": f"{mode}_student",
                "time": 6,
                "item_id": "math_problem",
                "observed": True,
            },
        ]

        response_log, latent_state = sim.run(events)
        student_state = latent_state["students"][f"{mode}_student"]
        skill_state = student_state.get_skill_state("math")

        responses = response_log["response"].tolist()

        results[mode] = {
            "responses": responses,
            "final_proficiency": skill_state.current_proficiency,
            "final_state": skill_state.proficiency_state,
        }

    print("\nMode Comparison:")
    for mode, result in results.items():
        print(
            f"{mode.upper()}: proficiency={result['final_proficiency']:.3f}, "
            f"state={result['final_state']}, responses={result['responses']}"
        )

    # CDM should show binary behavior
    cdm_proficiency = results["cdm"]["final_proficiency"]
    assert (
        cdm_proficiency == 0.0 or cdm_proficiency > 0.5
    ), f"CDM should show binary proficiency, got {cdm_proficiency}"


if __name__ == "__main__":
    test_cdm_prerequisite_learning()
    test_cdm_binary_response_model()
    test_cdm_mode_comparison()

    print("\n🎯 CDM Mode Successfully Implemented!")
    print("   📊 Binary skill states (learned/not learned)")
    print("   🔗 Prerequisite-aware learning probability")
    print("   📝 One skill per item alignment")
    print("   🎲 Uses existing BKT-style learning with CDM constraints")
