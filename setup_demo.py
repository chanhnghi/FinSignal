"""Install the demo dependencies into a local virtual environment."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
folder = ROOT / ".venv"
python = folder / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
if not folder.exists():
    subprocess.run([sys.executable, "-m", "venv", str(folder)], check=True)
subprocess.run([
    str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt"),
], check=True)
print("Setup complete. Run run_demo.py to open the demo.")
