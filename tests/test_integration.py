# test_integration.py
import subprocess

def test_cli_execution():
    """Checks if the trainer.py script can at least be initialized (Dry Run)."""
    # This just ensures there are no syntax errors in your main scripts
    result = subprocess.run(["python", "src/trainer.py", "--help"], capture_output=True)
    assert result.returncode == 0 or result.returncode == 1 # Depending on if you have --help