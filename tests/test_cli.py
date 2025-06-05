from typer.testing import CliRunner

from simlearn.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    # With single command, typer shows command description instead of app description
    assert "Run a simulation based on the given config file and events" in result.stdout
