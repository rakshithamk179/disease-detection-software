"""Configuration loader for the tomato disease classifier."""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def find_project_root(start_dir: Optional[Path] = None) -> Path:
    """Locate the project root directory by searching upward for config/config.yaml or pyproject.toml."""
    current = (start_dir or Path.cwd()).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "config" / "config.yaml").is_file() or (parent / "pyproject.toml").is_file():
            return parent
    # Fallback to current file's grandparent directory (src/ -> project_root)
    return Path(__file__).resolve().parent.parent


def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load configuration YAML and resolve project paths to absolute paths."""
    project_root = find_project_root()
    if config_path is None:
        target_path = project_root / "config" / "config.yaml"
    else:
        target_path = Path(config_path)
        if not target_path.is_absolute():
            target_path = project_root / target_path

    if not target_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # Attach resolved absolute paths
    paths = config.get("paths", {})
    resolved_paths: Dict[str, Path] = {}
    for key, val in paths.items():
        p = Path(val)
        if not p.is_absolute():
            resolved_paths[key] = (project_root / p).resolve()
        else:
            resolved_paths[key] = p.resolve()

    config["resolved_paths"] = resolved_paths
    config["project_root"] = project_root
    return config
