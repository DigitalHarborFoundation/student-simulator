from simlearn.config import RNGConfig, SimulationConfig, SkillsConfig


def test_default_rng_seed() -> None:
    cfg = SimulationConfig()
    assert isinstance(cfg.rng, RNGConfig)
    assert cfg.rng.seed == 42


def test_default_skills_empty() -> None:
    cfg = SimulationConfig()
    assert isinstance(cfg.skills, SkillsConfig)
    assert cfg.skills.skills == []
