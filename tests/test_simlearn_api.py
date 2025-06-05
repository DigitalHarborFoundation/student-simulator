import subprocess

import pytest
import yaml

# These imports will break until the implementation exists.
# This is intentional: TDD! We will make these work.
try:
    from simlearn import Sim, emit_response, generate_latent, transition
except ImportError:
    Sim = generate_latent = transition = emit_response = None


# --- Helpers: Minimal event stream and config ---
def minimal_event_stream():
    return [
        {"student_id": "stu1", "time": 0, "item_id": "item1", "observed": 1},
        {"student_id": "stu1", "time": 1, "item_id": "item2", "observed": 1},
    ]


def minimal_config_dict():
    # Only use required fields for fast test startup.
    return {
        "rng": {"seed": 1},
        "skills": {
            "skills": [
                {
                    "id": "skill1",
                    "type": "continuous",
                    "parents": [],
                    "practice_gain": 0.1,
                }
            ]
        },
        "items": {
            "items": [
                {"id": "item1", "q_vector": {"skill1": 1}},
                {"id": "item2", "q_vector": {"skill1": 1}},
            ]
        },
        "population": {"groups": [{"name": "default", "size": 1}]},
    }


# ================= USER-FACING FLUENT API USAGE ======================
@pytest.mark.skipif(Sim is None, reason="Fluent Sim API not implemented")
def test_fluent_sim_api_minimal():
    sim = (
        Sim(seed=123)
        .skills([{"id": "skill1", "type": "continuous"}])
        .items(
            [
                {"id": "item1", "q_vector": {"skill1": 1}},
                {"id": "item2", "q_vector": {"skill1": 1}},
            ]
        )
        .population([{"name": "default", "size": 1}])
    )
    # Provide a list of event records (as per requirements)
    event_stream = minimal_event_stream()
    log, latent = sim.run(event_stream)
    # Response log should contain required columns
    assert "student_id" in log.columns
    assert "item_id" in log.columns
    assert "response" in log.columns or "score" in log.columns
    # Latent truth log should be a DataFrame or dict with student skill/truth info
    assert latent is not None


# ================= USER-FACING CLI USAGE (End-to-end) =================
@pytest.mark.skipif(Sim is None, reason="Simlearn CLI run not implemented")
def test_cli_sim_minimal(tmp_path):
    # Write minimal YAML config
    cfg = minimal_config_dict()
    cfg_path = tmp_path / "test_sim.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)
    # Minimal events file (CSV or YAML, to be defined by simlearn)
    event_path = tmp_path / "events.yaml"
    events = minimal_event_stream()
    with open(event_path, "w") as f:
        yaml.safe_dump(events, f)
    # Run CLI: `simulate --config ... --events ...`
    # This will NOT pass until the CLI is extended.
    cmd = [
        "python",
        "-m",
        "simlearn.cli",
        "--config",
        str(cfg_path),
        "--events",
        str(event_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    # Should emit outputs/logs containing student/item
    assert "student_id" in result.stdout


# ================= LOW-LEVEL HOOK API USAGE ==========================
@pytest.mark.skipif(generate_latent is None, reason="Hook API not implemented")
def test_hooks_apparent_contract():
    cfg = minimal_config_dict()
    # Generate initial latent state for a student from config
    state = generate_latent(config=cfg)
    # Run a single transition step
    event = {"student_id": "stu1", "time": 1, "item_id": "item1", "observed": 1}
    new_state = transition(state, event)
    # Get deterministic response for event/state/item
    item_metadata = cfg["items"]["items"][0]  # item1
    response = emit_response(new_state, item_metadata)
    # Response should always have a 'score' or dichotomous output
    assert response in (0, 1)


# ================= OUTPUTS: FULL CONFIG, LATENTS, REPRO CHECK ========
@pytest.mark.skipif(Sim is None, reason="Sim API not implemented")
def test_sim_output_includes_config_and_seed():
    sim = Sim(seed=999)
    sim.skills([{"id": "skill1", "type": "continuous"}])
    sim.items([{"id": "item1", "q_vector": {"skill1": 1}}])
    sim.population([{"name": "default", "size": 1}])
    log, latent, config_dump = sim.run(minimal_event_stream(), return_config=True)
    # Check config/seed included in output
    assert isinstance(config_dump, dict)
    assert config_dump["rng"]["seed"] == 999


# ============ QUICK ADAPTIVE SCHEDULING ===============
@pytest.mark.skipif(Sim is None, reason="Adaptive scheduling not implemented")
def test_sim_adaptive_scheduling_api_shape():
    # This test expresses an API for item selection helpers
    sim = Sim(seed=1)
    sim.skills([{"id": "skill1", "type": "continuous"}])
    sim.items(
        [
            {"id": "item1", "q_vector": {"skill1": 1}},
            {"id": "item2", "q_vector": {"skill1": 1}},
        ]
    )
    sim.population([{"name": "default", "size": 1}])
    sim.helpers({"scheduling": {"mode": "adaptive", "method": "max_info"}})
    # Should be able to request next item for student based on current state
    next_item = sim.next_item_for("stu1", current_state={})
    assert next_item in {"item1", "item2"}
