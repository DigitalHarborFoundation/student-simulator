from pathlib import Path

import typer
import yaml

from simlearn.config import SimulationConfig

app = typer.Typer(help="Simlearn: Student Response Simulation Framework")


@app.command()
def run(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Path to YAML config file.",
    ),
    events: Path = typer.Option(
        ...,
        "--events",
        "-e",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Path to events file (YAML).",
    ),
    override: list[str] = typer.Option(
        None,
        "--override",
        "-o",
        help="Override config values, e.g., learning.global_kappa=0.4",
    ),
) -> None:
    """
    Run a simulation based on the given config file and events.
    """
    # Load config
    with open(config, "r") as f:
        config_data = yaml.safe_load(f)
    cfg = SimulationConfig.model_validate(config_data)

    # Apply overrides
    if override:
        for item in override:
            key, _, value = item.partition("=")
            # Simple override - in a real implementation this would need nested key support
            if hasattr(cfg, key):
                setattr(cfg, key, yaml.safe_load(value))

    # Load events
    with open(events, "r") as f:
        events_data = yaml.safe_load(f)

    # Run simulation using the API
    from .api import Sim

    sim = Sim(seed=cfg.rng.seed)
    sim.config = cfg

    result = sim.run(events_data)
    log, latent = result[0], result[1]

    # Output results
    typer.echo(f"Simulation completed. Generated {len(log)} responses.")
    typer.echo("Response log:")
    for _, row in log.iterrows():
        typer.echo(
            f"  student_id: {row['student_id']}, item_id: {row['item_id']}, response: {row['response']}"
        )

    typer.echo(
        f"\nLatent state contains data for {len(latent.get('skills', {}))} students."
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
