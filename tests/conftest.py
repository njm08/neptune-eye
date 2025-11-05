"""
Pytest configuration and shared fixtures for Neptune Eye tests.
"""
import pytest
from pathlib import Path
import sys

@pytest.fixture(scope="session", autouse=True)
def setup_python_path():
    """Add the app source directory to Python path for test imports."""
    project_root = Path(__file__).parent.parent
    app_src = project_root / "app" / "src"
    if str(app_src) not in sys.path:
        sys.path.insert(0, str(app_src))
@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent

@pytest.fixture
def app_path(project_root: Path) -> Path:
    """Return the project root directory."""
    return project_root / "app" / "src"

@pytest.fixture
def test_video_path(project_root: Path) -> Path:
    """Return path to test video file."""
    return project_root / "res" / "movies" / "boat_4.MOV"


@pytest.fixture
def neptune_eye_script(project_root: Path) -> Path:
    """Return path to the main Neptune Eye script."""
    return project_root / "app" / "src" / "neptune_eye" / "neptune_eye.py"


@pytest.fixture
def test_config_headless(tmp_path, test_video_path):
    """Create a test configuration file for headless mode."""
    config_content = f"""# Test Configuration for Neptune Eye
general:
  confidence: 0.5
  headless: true
  source: "MOVIE"
  camera_index: 0
  movie_path: "{test_video_path}"

expert:
  model_size: "YOLO11N"
  fp16: false
  iou_threshold: 0.45
  image_size: 640
  override_model_path: null
  override_device: "CPU"
"""
    config_file = tmp_path / "neptune_eye_config.yaml"
    config_file.write_text(config_content)
    return config_file
