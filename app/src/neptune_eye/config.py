"""
Configuration module for Neptune Eye

This module handles loading and validating configuration from YAML files.
"""
from logging import config
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

import yaml

from object_detection.yolo_object_detection import YoloModelSize, InferenceDevice
from utilites import find_project_root
import torch

# Default movie path (from root) if none is provided
DEFAULT_MOVIE_PATH = "res/movies/boat_4.MOV"

class InputSource(Enum):
    """Choose the input source for video processing."""
    CAMERA = "CAMERA"
    MOVIE = "MOVIE"


@dataclass
class GeneralConfig:
    """General configuration parameters."""
    confidence: float
    headless: bool
    source: InputSource
    camera_index: int
    movie_path: Optional[str]

@dataclass
class ExpertConfig:
    """Expert configuration parameters (advanced settings)."""
    model_size: YoloModelSize
    fp16: bool
    iou_threshold: float
    image_size: int
    override_model_path: Optional[str]
    override_device: Optional[InferenceDevice]
    resolved_model_path: Optional[str] = None  # Will be populated during config loading

@dataclass
class NeptuneEyeConfig:
    """Complete Neptune Eye configuration."""
    general: GeneralConfig
    expert: ExpertConfig


def _map_model_size(size_str: str) -> YoloModelSize:
    """Map string model size to YoloModelSize enum."""
    size_mapping = {
        "YOLO11N": YoloModelSize.YOLO11N,
        "YOLO11S": YoloModelSize.YOLO11S,
        "YOLO11M": YoloModelSize.YOLO11M,
    }
    if size_str not in size_mapping:
        raise ValueError(f"Invalid model size: {size_str}. Must be one of {list(size_mapping.keys())}")
    return size_mapping[size_str]


def _map_device(device_str: Optional[str]) -> Optional[InferenceDevice]:
    """Map string device to InferenceDevice enum."""
    if device_str is None:
        return None
    
    device_mapping = {
        "NVIDIA_GPU": InferenceDevice.NVIDIA_GPU,
        "M1_GPU": InferenceDevice.M1_GPU,
        "CPU": InferenceDevice.CPU,
    }
    if device_str not in device_mapping:
        raise ValueError(f"Invalid device: {device_str}. Must be one of {list(device_mapping.keys())} or null")
    return device_mapping[device_str]


def _map_input_source(source_str: str) -> InputSource:
    """Map string input source to InputSource enum."""
    source_mapping = {
        "CAMERA": InputSource.CAMERA,
        "MOVIE": InputSource.MOVIE,
    }
    if source_str not in source_mapping:
        raise ValueError(f"Invalid input source: {source_str}. Must be one of {list(source_mapping.keys())}")
    return source_mapping[source_str]


def _detect_best_device() -> InferenceDevice:
    """Detect the best available device for inference."""
    if torch.backends.mps.is_available():
        return InferenceDevice.M1_GPU
    elif torch.cuda.is_available():
        return InferenceDevice.NVIDIA_GPU
    else:
        return InferenceDevice.CPU


def _get_nvidia_gpu_model_path(model_size: YoloModelSize, fp16: bool) -> str:
    """Get model path for NVIDIA GPU inference."""
    precision_suffix = "16fp" if fp16 else "32fp"
    
    model_paths = {
        YoloModelSize.YOLO11N: f"engine/yolo11n_{precision_suffix}.engine",
        YoloModelSize.YOLO11S: f"engine/yolo11s_{precision_suffix}.engine",
        YoloModelSize.YOLO11M: f"engine/neptunes_{precision_suffix}.engine",  # Only neptunes is available for YOLO11M
    }

    if model_size not in model_paths:
        raise ValueError(f"Unsupported model size for NVIDIA GPU: {model_size}")
    
    return model_paths[model_size]


def _get_pytorch_model_path(model_size: YoloModelSize) -> str:
    """Get model path for PyTorch inference (M1 GPU or CPU)."""
    model_paths = {
        YoloModelSize.YOLO11N: "pytorch/yolo11n.pt",
        YoloModelSize.YOLO11S: "pytorch/yolo11s.pt",
        YoloModelSize.YOLO11M: "pytorch/yolo11m.pt",
    }

    if model_size not in model_paths:
        raise ValueError(f"Unsupported model size for PyTorch: {model_size}")
    
    return model_paths[model_size]

def _resolve_movie_path(general_config: GeneralConfig) -> str:
    """
    Resolve the absolute path to the movie file based on configuration.
    Supports both absolute paths and relative paths (relative to project root).
    
    Args:
        general_config: General configuration
        
    Returns:
        str: Absolute path to the movie file.
    """

    if general_config.source == InputSource.MOVIE:
        # If there is no movie path provided, use default sample movie
        if general_config.movie_path is None:
            movie_path = Path(find_project_root() / DEFAULT_MOVIE_PATH).resolve()
        else:
            provided_path = Path(general_config.movie_path)
            if provided_path.is_absolute():
                # Use absolute path as-is
                movie_path = provided_path.resolve()
            else:
                # Relative path - resolve from project root
                project_root = find_project_root()
                movie_path = (project_root / provided_path).resolve()
    
    return str(movie_path)

def _resolve_model_path(expert_config: ExpertConfig) -> str:
    """
    Resolve the absolute path to the model file based on configuration.
    
    Args:
        expert_config: Expert configuration containing size, device preferences, etc.
        
    Returns:
        str: Absolute path to the model file.
        
    Raises:
        ValueError: If model configuration is invalid or model file doesn't exist.
    """
    # If user specified a custom model path, use it directly
    if expert_config.override_model_path:
        custom_path = Path(expert_config.override_model_path)
        if custom_path.is_absolute():
            if not custom_path.exists():
                raise ValueError(f"Custom model file does not exist: {custom_path}")
            return str(custom_path)
        else:
            # Relative path - resolve from project root
            project_root = find_project_root()
            absolute_path = (project_root / custom_path).resolve()
            if not absolute_path.exists():
                raise ValueError(f"Custom model file does not exist: {absolute_path}")
            return str(absolute_path)
    
    # Determine device to use
    device = expert_config.override_device or _detect_best_device()
    
    # Get relative model path based on device and model size
    if device == InferenceDevice.NVIDIA_GPU:
        relative_path = _get_nvidia_gpu_model_path(expert_config.model_size, expert_config.fp16)
    else:
        relative_path = _get_pytorch_model_path(expert_config.model_size)
    
    # Convert to absolute path
    project_root = find_project_root()
    models_dir = project_root / "models"
    absolute_path = (models_dir / relative_path).resolve()
    
    if not absolute_path.exists():
        raise ValueError(f"Model file does not exist: {absolute_path}. "
                        f"Expected model for {expert_config.model_size.name} on {device.name}")
    
    return str(absolute_path)


def _create_default_config_content() -> str:
    """Create default configuration file content."""
    content = """# Neptune Eye Configuration
# This file contains all configuration parameters for the Neptune Eye object detection system

# General Configuration
general:
  confidence: 0.5                    # Confidence threshold for detections (0.0 - 1.0)
  headless: true                     # True to run without showing images (headless mode), False to display images
  source: "MOVIE"                    # Options: "CAMERA", "MOVIE"
  camera_index: 0                    # Camera index (0 for default/built-in, 1+ for external cameras)
  movie_path: null                   # Path to movie file. Can be relative to root or absolute. If null is set, a sample video will be used.

# Expert Configuration (Advanced Settings)
expert:
  model_size: "YOLO11S"              # Options: YOLO11N, YOLO11S, YOLO11M
  fp16: false                        # True to use FP16 precision for better performance
  iou_threshold: 0.45                # IoU threshold for NMS (Non-Maximum Suppression)
  image_size: 640                    # Input image size for YOLO model
  override_model_path: null          # Path to model file. Can be relative to root or absolute. If null, the best model will be used.
  override_device: null              # Options: null (Device is detected automatically), "NVIDIA_GPU", "M1_GPU", "CPU"
"""
    return content


def _create_default_config_file(config_path: Path) -> None:
    """Create a default configuration file at the specified path."""
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(_create_default_config_content())
        print(f"Created default configuration file: {config_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to create default configuration file {config_path}: {e}")


def load_config(config_path: Optional[Path] = None) -> NeptuneEyeConfig:
    """
    Load Neptune Eye configuration from YAML file.
    Creates a default configuration file if none exists.
    
    Args:
        config_path: Path to config file. If None, uses default config.yaml in project root.
        
    Returns:
        NeptuneEyeConfig: Parsed and validated configuration object.
        
    Raises:
        yaml.YAMLError: If config file has invalid YAML syntax.
        ValueError: If config values are invalid.
        RuntimeError: If unable to create default configuration file.
    """
    if config_path is None:
        root_dir = find_project_root()
        config_path = root_dir / "config.yaml"
    
    if not config_path.exists():
        print(f"Configuration file not found at {config_path}")
        print(f"Creating default configuration file...")
        _create_default_config_file(config_path)
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in config file {config_path}: {e}")
    
    if not isinstance(config_data, dict):
        raise ValueError(f"Config file must contain a YAML object/dictionary, got {type(config_data)}")
    
    # Validate and parse configuration sections
    try:
        general_config = GeneralConfig(
            confidence=float(config_data["general"]["confidence"]),
            headless=bool(config_data["general"]["headless"]),
            source=_map_input_source(config_data["general"]["source"]),
            camera_index=int(config_data["general"]["camera_index"]),
            movie_path=config_data["general"]["movie_path"]
        )
        
        expert_config = ExpertConfig(
            model_size=_map_model_size(config_data["expert"]["model_size"]),
            fp16=bool(config_data["expert"]["fp16"]),
            iou_threshold=float(config_data["expert"]["iou_threshold"]),
            image_size=int(config_data["expert"]["image_size"]),
            override_model_path=config_data["expert"]["override_model_path"],
            override_device=_map_device(config_data["expert"]["override_device"])
        )
        
        # Resolve paths and store in configs
        expert_config.resolved_model_path = _resolve_model_path(expert_config)
        general_config.movie_path = _resolve_movie_path(general_config)
        
        # Print general configuration
        print("General Configuration:")
        print(f"Confidence: {general_config.confidence}")
        print(f"Headless: {general_config.headless}")
        print(f"Source: {general_config.source.value}")
        if general_config.source == InputSource.MOVIE:
            print(f"Movie Path: {general_config.movie_path}")
        elif general_config.source == InputSource.CAMERA:
            print(f"Camera Index: {general_config.camera_index}")
        
        return NeptuneEyeConfig(
            general=general_config,
            expert=expert_config
        )
        
    except KeyError as e:
        raise ValueError(f"Missing required configuration key: {e}")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid configuration value: {e}")


def validate_config(config: NeptuneEyeConfig) -> None:
    """
    Validate configuration values for logical consistency.
    
    Args:
        config: Configuration object to validate.
        
    Raises:
        ValueError: If configuration values are invalid or inconsistent.
    """
    # Validate confidence threshold
    if not 0.0 <= config.general.confidence <= 1.0:
        raise ValueError(f"Confidence must be between 0.0 and 1.0, got {config.general.confidence}")
    
    # Validate IoU threshold
    if not 0.0 <= config.expert.iou_threshold <= 1.0:
        raise ValueError(f"IoU threshold must be between 0.0 and 1.0, got {config.expert.iou_threshold}")
    
    # Validate image size
    if config.expert.image_size <= 0:
        raise ValueError(f"Image size must be positive, got {config.expert.image_size}")
    
    # Validate camera index
    if config.general.camera_index < 0:
        raise ValueError(f"Camera index must be non-negative, got {config.general.camera_index}")
    
    # Validate model path if provided
    if config.expert.override_model_path is not None:
        model_path = Path(config.expert.override_model_path)
        if not model_path.exists():
            raise ValueError(f"Override model path does not exist: {model_path}")
    
    # Validate movie path if using movie input
    if config.general.source == InputSource.MOVIE:
        movie_path = Path(config.general.movie_path)
        if not movie_path.exists():
            raise ValueError(f"Movie file does not exist: {movie_path}")