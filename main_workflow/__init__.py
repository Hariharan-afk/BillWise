from pathlib import Path
from dotenv import load_dotenv

_repo_root = Path(__file__).resolve().parents[1]
_env_path = _repo_root / ".env"

load_dotenv(_env_path, override=False)

__all__ = []
