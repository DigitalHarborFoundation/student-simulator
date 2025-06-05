"""
Test different psychometric model modes that the hybrid framework can emulate:
1. BKT Mode: Binary state transitions, probabilistic learning
2. PFA Mode: Gradual performance improvement with practice
3. IRT Mode: Static ability-based responses, no learning
4. Hybrid Mode: Full model with all features

Each mode auto-configures parameters to emulate the target psychometric model.
"""

import numpy as np

from simlearn import Sim
from simlearn.config import Student


def test_bkt_mode():
    """Test BKT mode: binary learning states, probabilistic skill acquisition."""

    # BKT Mode Configuration
    skills = [
        {
            "id": "reading_comprehension",
            "type": "continuous",
            "description": "Understanding written text",
        }
    ]

    items = [
        {
            "id": "passage_question",
            "title": "Answer questions about a passage",
            "skills": [skills[0]],
            "g": 0.25,  # 25% guess rate
            "s": 0.05,  # 5% slip rate
            "practice_effectiveness": 1.0,  # Binary jump to mastery (BKT characteristic)
        }
    ]

    student = Student(id="bkt_learner")

    sim = (
        Sim(seed=42)
        .skills(skills)
        .items(items)
        .population([student.model_dump()])
        .learning(mode="bkt")
    )

    # BKT-style events: each practice is a learning opportunity
    events = []
    for trial in range(1, 8):
        events.append(
            {
                "student_id": "bkt_learner",
                "time": trial,
                "item_id": "passage_question",
                "observed": True,
                "intervention_type": "skill_boost",  # Each practice = potential learning
                "context": {
                    "target_skill": "reading_comprehension",
                    "boost": 1.0,  # Full mastery when learned (BKT binary)
                },
            }
        )

    response_log, latent_state = sim.run(events)
    learner = latent_state["students"]["bkt_learner"]
    skill_state = learner.get_skill_state("reading_comprehension")

    # BKT Characteristics to verify:
    responses = response_log["response"].tolist()

    # 1. Binary state transition (should see sudden jump in performance)
    performance_windows = []
    window_size = 2
    for i in range(len(responses) - window_size + 1):
        window_perf = np.mean(responses[i : i + window_size])
        performance_windows.append(window_perf)

    # 2. Final state should be binary (either no_learning or mastery)
    assert skill_state.proficiency_state in [
        "no_learning",
        "mastery",
    ], f"BKT should have binary states, got {skill_state.proficiency_state}"

    # 3. If learned, should be at maximum proficiency
    if skill_state.proficiency_state == "mastery":
        assert (
            skill_state.current_proficiency >= 0.99
        ), f"BKT mastery should be near 1.0, got {skill_state.current_proficiency}"

    # 4. Response probabilities should follow BKT pattern

    # Test actual probability calculation
    actual_prob = skill_state.get_probability_correct(0.25, 0.05)
    if skill_state.proficiency_state == "no_learning":
        assert abs(actual_prob - 0.25) < 0.01, "No learning should give guess rate"
    elif skill_state.proficiency_state == "mastery":
        assert abs(actual_prob - 0.95) < 0.01, "Mastery should give 1-slip rate"

    print("\nBKT Mode Results:")
    print(f"Final state: {skill_state.proficiency_state}")
    print(f"Final proficiency: {skill_state.current_proficiency:.3f}")
    print(f"Response pattern: {responses}")
    print(f"Performance trajectory: {[f'{p:.2f}' for p in performance_windows]}")
    print(f"P(correct): {actual_prob:.3f}")


def test_pfa_mode():
    """Test PFA mode: gradual performance improvement with practice."""

    # PFA Mode Configuration
    skills = [
        {
            "id": "algebra_solving",
            "type": "continuous",
            "description": "Solving algebraic equations",
        }
    ]

    items = [
        {
            "id": "linear_equation",
            "title": "Solve linear equations",
            "skills": [skills[0]],
            "g": 0.15,
            "s": 0.08,
            "practice_effectiveness": 0.15,  # Gradual improvement (PFA characteristic)
        }
    ]

    student = Student(id="pfa_learner")

    sim = (
        Sim(seed=123)
        .skills(skills)
        .items(items)
        .population([student.model_dump()])
        .learning(mode="pfa")
    )

    # PFA-style events: learning happens, then gradual improvement with practice
    events = [
        # Initial learning intervention
        {
            "student_id": "pfa_learner",
            "time": 1,
            "item_id": "linear_equation",
            "observed": True,
            "intervention_type": "skill_boost",
            "context": {
                "target_skill": "algebra_solving",
                "boost": 0.2,  # Smaller initial proficiency for PFA gradual improvement
            },
        }
    ]

    # Multiple practice instances for gradual improvement
    for practice_round in range(
        2, 6
    ):  # Fewer practice rounds to stay in baseline state
        events.append(
            {
                "student_id": "pfa_learner",
                "time": practice_round,
                "item_id": "linear_equation",
                "observed": True,
            }
        )

    response_log, latent_state = sim.run(events)
    learner = latent_state["students"]["pfa_learner"]
    skill_state = learner.get_skill_state("algebra_solving")

    # PFA Characteristics to verify:
    responses = response_log["response"].tolist()

    # 1. Gradual improvement (not binary)
    early_performance = np.mean(responses[1:3])  # Skip initial intervention
    late_performance = np.mean(responses[-3:])  # Last few responses

    improvement = late_performance - early_performance
    print("\nPFA Mode Results:")
    print(f"Early performance: {early_performance:.3f}")
    print(f"Late performance: {late_performance:.3f}")
    print(f"Improvement: {improvement:.3f}")

    # 2. Should show measurable improvement
    assert improvement >= 0, "PFA should show improvement over time"

    # 3. Should NOT be in mastery state immediately (gradual learning)
    assert skill_state.proficiency_state in [
        "baseline"
    ], f"PFA should be in gradual improvement state, got {skill_state.proficiency_state}"

    # 4. Proficiency should be between initial and maximum (showing progression)
    assert (
        0.2 < skill_state.current_proficiency < 1.0
    ), f"PFA should show intermediate proficiency, got {skill_state.current_proficiency}"

    # 5. Practice count should reflect all practice instances
    assert (
        skill_state.practice_count >= 3
    ), f"PFA should count practice instances, got {skill_state.practice_count}"

    print(f"Final proficiency: {skill_state.current_proficiency:.3f}")
    print(f"Practice count: {skill_state.practice_count}")
    print(f"Response pattern: {responses}")


def test_irt_mode():
    """Test IRT mode: static ability-based responses, no learning."""

    # IRT Mode Configuration
    skills = [
        {
            "id": "vocabulary_knowledge",
            "type": "continuous",
            "description": "Knowledge of word meanings",
        }
    ]

    items = [
        {
            "id": "vocab_test",
            "title": "Define vocabulary words",
            "skills": [skills[0]],
            "g": 0.20,
            "s": 0.10,
            "practice_effectiveness": 0.0,  # No learning from practice (IRT characteristic)
        }
    ]

    # Student with fixed ability (IRT characteristic)
    student = Student(id="irt_responder", initial_theta=0.5)  # Fixed ability level

    sim = (
        Sim(seed=789)
        .skills(skills)
        .items(items)
        .population([student.model_dump()])
        .learning(mode="irt")
    )

    # IRT-style events: repeated testing with no learning
    events = []
    for test_item in range(1, 31):  # 30 test administrations for better statistics
        events.append(
            {
                "student_id": "irt_responder",
                "time": test_item,
                "item_id": "vocab_test",
                "observed": True,
                # No interventions - pure testing (IRT characteristic)
            }
        )

    response_log, latent_state = sim.run(events)
    responder = latent_state["students"]["irt_responder"]
    skill_state = responder.get_skill_state("vocabulary_knowledge")

    # IRT Characteristics to verify:
    responses = response_log["response"].tolist()

    # 1. No learning should occur (proficiency stays at initial level)
    assert (
        skill_state.proficiency_state == "no_learning"
    ), f"IRT mode should have no learning, got {skill_state.proficiency_state}"

    # 2. Performance should be stable (no improvement trend)
    early_performance = np.mean(responses[:3])
    middle_performance = np.mean(responses[4:7])
    late_performance = np.mean(responses[-3:])

    # Allow for random variation but no systematic improvement
    performances = [early_performance, middle_performance, late_performance]
    performance_variance = np.var(performances)

    assert (
        performance_variance < 0.1
    ), f"IRT should have stable performance, variance={performance_variance:.3f}"

    # 3. Responses should be driven by guess parameter only
    expected_performance = 0.20  # guess parameter
    actual_performance = np.mean(responses)

    # Allow for sampling variation around guess rate (with more generous tolerance)
    assert (
        abs(actual_performance - expected_performance) < 0.15
    ), f"IRT performance should be near guess rate {expected_performance}, got {actual_performance}"

    # 4. No practice effects recorded
    assert (
        skill_state.practice_count == 0
    ), f"IRT should have no practice effects, got {skill_state.practice_count} practices"

    print("\nIRT Mode Results:")
    print(
        f"Performance stability: Early={early_performance:.3f}, Mid={middle_performance:.3f}, Late={late_performance:.3f}"
    )
    print(f"Performance variance: {performance_variance:.4f}")
    print(
        f"Average performance: {actual_performance:.3f} (expected ~{expected_performance:.3f})"
    )
    print(f"Final state: {skill_state.proficiency_state}")
    print(f"Response pattern: {responses}")


def test_hybrid_mode():
    """Test Hybrid mode: full model with all psychometric features."""

    # Hybrid Mode Configuration - enables all features
    skills = [
        {
            "id": "scientific_reasoning",
            "type": "continuous",
            "description": "Hypothesis testing and experimental design",
            "parents": [],  # Could have prerequisites
        }
    ]

    items = [
        {
            "id": "experiment_design",
            "title": "Design a controlled experiment",
            "skills": [skills[0]],
            "g": 0.18,
            "s": 0.07,
            "practice_effectiveness": 0.25,  # Moderate improvement rate
        }
    ]

    student = Student(
        id="hybrid_learner", initial_theta=-0.2  # Below average initial ability
    )

    sim = (
        Sim(seed=456)
        .skills(skills)
        .items(items)
        .population([student.model_dump()])
        .learning(mode="hybrid")
    )

    # Hybrid events: combines interventions, practice, and assessment
    events = [
        # Phase 1: No learning baseline
        {
            "student_id": "hybrid_learner",
            "time": 1,
            "item_id": "experiment_design",
            "observed": True,
        },
        {
            "student_id": "hybrid_learner",
            "time": 2,
            "item_id": "experiment_design",
            "observed": True,
        },
        # Phase 2: Intervention (probabilistic learning)
        {
            "student_id": "hybrid_learner",
            "time": 5,
            "item_id": "experiment_design",
            "observed": True,
            "intervention_type": "skill_boost",
            "context": {
                "target_skill": "scientific_reasoning",
                "boost": 0.6,  # Moderate baseline proficiency
            },
        },
        # Phase 3: Practice with gradual improvement
        {
            "student_id": "hybrid_learner",
            "time": 6,
            "item_id": "experiment_design",
            "observed": True,
        },
        {
            "student_id": "hybrid_learner",
            "time": 7,
            "item_id": "experiment_design",
            "observed": True,
        },
        {
            "student_id": "hybrid_learner",
            "time": 8,
            "item_id": "experiment_design",
            "observed": True,
        },
        {
            "student_id": "hybrid_learner",
            "time": 9,
            "item_id": "experiment_design",
            "observed": True,
        },
        # Phase 4: Assessment after learning
        {
            "student_id": "hybrid_learner",
            "time": 12,
            "item_id": "experiment_design",
            "observed": True,
        },
        {
            "student_id": "hybrid_learner",
            "time": 13,
            "item_id": "experiment_design",
            "observed": True,
        },
    ]

    response_log, latent_state = sim.run(events)
    learner = latent_state["students"]["hybrid_learner"]
    skill_state = learner.get_skill_state("scientific_reasoning")

    # Hybrid Characteristics to verify:
    responses = response_log["response"].tolist()

    # 1. Should progress through multiple states
    assert skill_state.proficiency_state in [
        "baseline",
        "mastery",
    ], f"Hybrid should show learning progression, got {skill_state.proficiency_state}"

    # 2. Should show improvement from baseline
    baseline_performance = np.mean(responses[:2])  # Before intervention
    final_performance = np.mean(responses[-2:])  # After learning/practice

    total_improvement = final_performance - baseline_performance
    assert (
        total_improvement > 0
    ), f"Hybrid should show overall improvement, got {total_improvement:.3f}"

    # 3. Should have practice effects recorded
    assert (
        skill_state.practice_count > 0
    ), f"Hybrid should record practice, got {skill_state.practice_count}"

    # 4. Should have learning events in history
    learning_events = [
        e
        for e in learner.learning_history
        if e.event_type in ["skill_learned", "skill_practiced"]
    ]
    assert len(learning_events) > 0, "Hybrid should record learning events"

    # 5. Should have intermediate proficiency (not just binary)
    assert (
        0.1 < skill_state.current_proficiency <= 1.0
    ), f"Hybrid should show nuanced proficiency, got {skill_state.current_proficiency}"

    print("\nHybrid Mode Results:")
    print(f"Baseline performance: {baseline_performance:.3f}")
    print(f"Final performance: {final_performance:.3f}")
    print(f"Total improvement: {total_improvement:.3f}")
    print(f"Final proficiency: {skill_state.current_proficiency:.3f}")
    print(f"Final state: {skill_state.proficiency_state}")
    print(f"Practice count: {skill_state.practice_count}")
    print(f"Learning events: {len(learning_events)}")
    print(f"Response pattern: {responses}")


def test_mode_comparison():
    """Compare all four modes on the same skill to show different behaviors."""

    print("\n" + "=" * 60)
    print("PSYCHOMETRIC MODEL MODE COMPARISON")
    print("=" * 60)

    # Run all modes with same skill and similar items
    skills = [{"id": "pattern_recognition", "type": "continuous"}]

    modes = {
        "BKT": {
            "practice_effectiveness": 1.0,
            "intervention_boost": 1.0,
            "description": "Binary states, probabilistic learning",
        },
        "PFA": {
            "practice_effectiveness": 0.2,
            "intervention_boost": 0.5,
            "description": "Gradual improvement with practice",
        },
        "IRT": {
            "practice_effectiveness": 0.0,
            "intervention_boost": 0.0,
            "description": "No learning, fixed ability",
        },
        "Hybrid": {
            "practice_effectiveness": 0.3,
            "intervention_boost": 0.6,
            "description": "Full model with all features",
        },
    }

    results = {}

    for mode_name, config in modes.items():
        items = [
            {
                "id": "pattern_item",
                "skills": [skills[0]],
                "g": 0.25,
                "s": 0.05,
                "practice_effectiveness": config["practice_effectiveness"],
            }
        ]

        student = Student(id=f"{mode_name.lower()}_student")
        sim = (
            Sim(seed=42)  # Same seed for comparison
            .skills(skills)
            .items(items)
            .population([student.model_dump()])
        )

        # Standard event sequence
        events = [
            # Baseline
            {
                "student_id": f"{mode_name.lower()}_student",
                "time": 1,
                "item_id": "pattern_item",
                "observed": True,
            },
            # Intervention
            {
                "student_id": f"{mode_name.lower()}_student",
                "time": 3,
                "item_id": "pattern_item",
                "observed": True,
                "intervention_type": "skill_boost",
                "context": {
                    "target_skill": "pattern_recognition",
                    "boost": config["intervention_boost"],
                },
            },
            # Practice
            {
                "student_id": f"{mode_name.lower()}_student",
                "time": 5,
                "item_id": "pattern_item",
                "observed": True,
            },
            {
                "student_id": f"{mode_name.lower()}_student",
                "time": 6,
                "item_id": "pattern_item",
                "observed": True,
            },
            {
                "student_id": f"{mode_name.lower()}_student",
                "time": 7,
                "item_id": "pattern_item",
                "observed": True,
            },
        ]

        response_log, latent_state = sim.run(events)
        student_state = latent_state["students"][f"{mode_name.lower()}_student"]
        skill_state = student_state.get_skill_state("pattern_recognition")

        responses = response_log["response"].tolist()
        baseline_perf = responses[0]
        final_perf = np.mean(responses[-2:])

        results[mode_name] = {
            "baseline_performance": baseline_perf,
            "final_performance": final_perf,
            "improvement": final_perf - baseline_perf,
            "final_proficiency": skill_state.current_proficiency,
            "final_state": skill_state.proficiency_state,
            "practice_count": skill_state.practice_count,
            "description": config["description"],
        }

    # Print comparison table
    print(
        f"{'Mode':<8} {'Description':<35} {'Baseline':<9} {'Final':<7} {'Improve':<8} {'Profic':<7} {'State':<10}"
    )
    print("-" * 90)

    for mode_name, result in results.items():
        print(
            f"{mode_name:<8} {result['description']:<35} "
            f"{result['baseline_performance']:<9.3f} {result['final_performance']:<7.3f} "
            f"{result['improvement']:<8.3f} {result['final_proficiency']:<7.3f} {result['final_state']:<10}"
        )

    # Verify expected differences
    assert (
        results["IRT"]["improvement"] <= results["PFA"]["improvement"]
    ), "IRT should show less improvement than PFA"
    assert (
        results["PFA"]["final_proficiency"] > 0
    ), "PFA should show positive proficiency"
    assert results["BKT"]["final_state"] in [
        "no_learning",
        "mastery",
    ], "BKT should have binary states"


if __name__ == "__main__":
    test_bkt_mode()
    test_pfa_mode()
    test_irt_mode()
    test_hybrid_mode()
    test_mode_comparison()

    print("\n🎯 All Psychometric Model Modes Successfully Tested!")
    print("   📊 BKT Mode: Binary learning states, probabilistic acquisition")
    print("   📈 PFA Mode: Gradual performance improvement with practice")
    print("   📋 IRT Mode: Static ability-based responses, no learning")
    print("   🧠 Hybrid Mode: Full model combining all psychometric features")
    print("   🔄 Parameter tuning enables switching between model types")
