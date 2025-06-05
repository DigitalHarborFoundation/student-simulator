from simlearn import Sim
from simlearn.config import Item, Skill, Student


def test_student_learning_journey_with_prerequisites():
    """
    Test a complete learning journey showing:
    - Skills with prerequisite relationships (basic -> intermediate -> advanced)
    - Student progressing through interventions and practice
    - Learning history tracking skill changes over time
    """

    # Define skills with prerequisite chain: basic_math -> fractions -> decimals
    skills = [
        {
            "id": "basic_math",
            "code": "MATH.K.NBT.1",
            "description": "Count and understand numbers 1-20",
            "domain": "mathematics",
            "type": "continuous",
            "parents": [],  # no prerequisites
            "practice_gain": 0.3,
            "decay": 0.01,
        },
        {
            "id": "fractions",
            "code": "MATH.3.NF.1",
            "description": "Understand fractions as parts of a whole",
            "domain": "mathematics",
            "type": "continuous",
            "parents": ["basic_math"],  # requires basic math
            "practice_gain": 0.2,
            "decay": 0.02,
        },
        {
            "id": "decimals",
            "code": "MATH.5.NBT.3",
            "description": "Read and write decimals to thousandths",
            "domain": "mathematics",
            "type": "continuous",
            "parents": ["fractions"],  # requires fractions
            "practice_gain": 0.15,
            "decay": 0.02,
        },
    ]

    # Define items that test different skills
    items = [
        {
            "id": "counting_item",
            "title": "Count objects 1-10",
            "skills": [skills[0]],  # basic_math only
            "g": 0.1,
            "s": 0.05,
            "a": 1.0,
            "b": -0.5,  # easier item
        },
        {
            "id": "fraction_item",
            "title": "Identify 1/2 of a pizza",
            "skills": [skills[0], skills[1]],  # basic_math + fractions
            "g": 0.15,
            "s": 0.1,
            "a": 1.2,
            "b": 0.0,  # medium difficulty
        },
        {
            "id": "decimal_item",
            "title": "Convert 0.5 to fraction",
            "skills": [skills[1], skills[2]],  # fractions + decimals
            "g": 0.2,
            "s": 0.15,
            "a": 1.5,
            "b": 0.8,  # harder item
        },
    ]

    # Create a student starting with low ability
    student = Student(
        id="alice",
        name="Alice Johnson",
        initial_theta=-1.0,  # starts below average
        metadata={"grade": 3, "age": 8},
    )

    # Set up simulation
    sim = Sim(seed=42).skills(skills).items(items).population([student.model_dump()])

    # Learning journey: practice basic math, then interventions for advanced skills
    learning_events = [
        # Week 1: Practice basic counting (should improve basic_math)
        {
            "student_id": "alice",
            "time": 1,
            "item_id": "counting_item",
            "observed": True,
        },
        {
            "student_id": "alice",
            "time": 2,
            "item_id": "counting_item",
            "observed": True,
        },
        {
            "student_id": "alice",
            "time": 3,
            "item_id": "counting_item",
            "observed": True,
        },
        # Week 2: Intervention - explicit fractions instruction
        {
            "student_id": "alice",
            "time": 10,
            "item_id": "fraction_item",
            "observed": True,
            "intervention_type": "skill_boost",
            "context": {"target_skill": "fractions", "boost": 0.5},
        },
        # Week 3: Practice fractions after intervention
        {
            "student_id": "alice",
            "time": 15,
            "item_id": "fraction_item",
            "observed": True,
        },
        {
            "student_id": "alice",
            "time": 16,
            "item_id": "fraction_item",
            "observed": True,
        },
        # Week 4: Try advanced decimals (might fail initially)
        {
            "student_id": "alice",
            "time": 20,
            "item_id": "decimal_item",
            "observed": True,
        },
        # Week 5: Intervention - decimals instruction
        {
            "student_id": "alice",
            "time": 25,
            "item_id": "decimal_item",
            "observed": True,
            "intervention_type": "skill_boost",
            "context": {"target_skill": "decimals", "boost": 0.4},
        },
        # Week 6: More practice on all skills
        {
            "student_id": "alice",
            "time": 30,
            "item_id": "counting_item",
            "observed": True,
        },
        {
            "student_id": "alice",
            "time": 31,
            "item_id": "fraction_item",
            "observed": True,
        },
        {
            "student_id": "alice",
            "time": 32,
            "item_id": "decimal_item",
            "observed": True,
        },
    ]

    # Run the simulation
    response_log, latent_state = sim.run(learning_events)

    # Verify we got responses for all events
    assert len(response_log) == len(learning_events)
    assert "alice" in latent_state["students"]

    alice = latent_state["students"]["alice"]

    # Check that Alice has skill states for all three skills
    assert "basic_math" in alice.current_skills
    assert "fractions" in alice.current_skills
    assert "decimals" in alice.current_skills

    # Verify learning progression: skills should improve over time
    basic_math_level = alice.get_skill_level("basic_math")
    fractions_level = alice.get_skill_level("fractions")
    decimals_level = alice.get_skill_level("decimals")

    # Alice started at -1.0, so any positive skill levels show learning
    assert (
        basic_math_level > alice.initial_theta
    ), f"Basic math should improve from {alice.initial_theta}"
    assert (
        fractions_level > alice.initial_theta
    ), f"Fractions should improve from {alice.initial_theta}"

    # Check that responses show improvement over time
    alice_responses = response_log[response_log["student_id"] == "alice"]

    # Early counting responses vs later counting responses should show improvement
    early_counting = alice_responses[
        (alice_responses["item_id"] == "counting_item") & (alice_responses["time"] <= 5)
    ]["response"].mean()

    later_counting = alice_responses[
        (alice_responses["item_id"] == "counting_item")
        & (alice_responses["time"] >= 30)
    ]["response"].mean()

    # Should see improvement in performance (though with randomness, not guaranteed)
    print("\nAlice's Learning Journey:")
    print(f"Initial ability: {alice.initial_theta}")
    print("Final skill levels:")
    print(f"  Basic Math: {basic_math_level:.3f}")
    print(f"  Fractions: {fractions_level:.3f}")
    print(f"  Decimals: {decimals_level:.3f}")
    print(f"Early counting performance: {early_counting:.2f}")
    print(f"Later counting performance: {later_counting:.2f}")
    print(f"Total learning events recorded: {len(alice.learning_history)}")

    # Verify learning history was recorded
    assert len(alice.learning_history) > 0, "Should have recorded learning events"

    # Check for different types of events
    event_types = {event.event_type for event in alice.learning_history}
    assert (
        "behavioral_event" in event_types
    ), "Should have behavioral events (student responses)"

    # Print detailed learning history
    print("\nDetailed Learning History:")
    for i, event in enumerate(alice.learning_history[:5]):  # Show first 5 events
        # Extract item and response info from context for behavioral events
        item_info = ""
        response_info = ""
        if event.event_type == "behavioral_event" and "behavior" in event.context:
            behavior = event.context["behavior"]
            item_info = f"item: {behavior.get('item_id', 'N/A')}"
            if "feedback" in event.context and event.context["feedback"]:
                feedback = event.context["feedback"]
                response_info = f"correct: {feedback.get('correct', 'N/A')}"
            else:
                response_info = "response: recorded"
        elif event.event_type == "skill_update":
            item_info = "skill update"
            changes = list(event.skill_changes.keys())
            response_info = f"changed: {', '.join(changes)}"

        print(
            f"  {i+1}. Time {event.timestamp}: {event.event_type} ({item_info}, {response_info})"
        )

    if len(alice.learning_history) > 5:
        print(f"  ... and {len(alice.learning_history) - 5} more events")


def test_prerequisite_skill_relationships():
    """
    Test that prerequisite relationships are properly modeled in the config.
    """

    # Create skills with clear prerequisite chain
    basic_skill = Skill(id="basic", description="Foundation skill", parents=[])

    intermediate_skill = Skill(
        id="intermediate", description="Builds on basic", parents=["basic"]
    )

    advanced_skill = Skill(
        id="advanced", description="Requires intermediate", parents=["intermediate"]
    )

    # Verify the prerequisite relationships
    assert basic_skill.parents == []
    assert intermediate_skill.parents == ["basic"]
    assert advanced_skill.parents == ["intermediate"]

    # Create an item that requires multiple skills
    multi_skill_item = Item(
        id="complex_item",
        title="Requires multiple skills",
        skills=[basic_skill, intermediate_skill, advanced_skill],
    )

    # Verify the item references all three skills
    assert len(multi_skill_item.skills) == 3
    skill_ids = {skill.id for skill in multi_skill_item.skills}
    assert skill_ids == {"basic", "intermediate", "advanced"}


if __name__ == "__main__":
    # Allow running this test file directly to see the learning journey
    test_student_learning_journey_with_prerequisites()
    test_prerequisite_skill_relationships()
    print("\nAll tests passed! 🎉")
