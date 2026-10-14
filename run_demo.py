"""Launch the demo using the project's virtual environment when available."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
python = ROOT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
if not python.is_file():
    python = Path(sys.executable)
print("Open http://127.0.0.1:8501. Stop with Ctrl+C.", flush=True)
try:
    raise SystemExit(subprocess.call([
        str(python), "-m", "streamlit", "run", str(ROOT / "app.py"),
        "--server.address", "127.0.0.1", "--server.port", "8501",
    ], cwd=ROOT))
except KeyboardInterrupt:
    pass
