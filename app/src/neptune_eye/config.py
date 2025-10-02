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


class InputSource(Enum):
    """Choose the input source for video processing."""
    CAMERA = "CAMERA"
    MOVIE = "MOVIE"


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    size: YoloModelSize
    fp16: bool
    override_device: Optional[InferenceDevice]
    override_model_path: Optional[str]
    confidence: float
    image_size: int
    iou_threshold: float

@dataclass
class InputConfig:
    """Input source configuration parameters."""
    source: InputSource
    camera_index: int
    movie_path: Optional[str]

@dataclass
class NeptuneEyeConfig:
    """Complete Neptune Eye configuration."""
    model: ModelConfig
    input: InputConfig


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


def _create_default_config_content() -> str:
    """Create default configuration file content."""
    content = """# Neptune Eye Configuration
# This file contains all configuration parameters for the Neptune Eye object detection system

# YOLO Model Configuration
model:
  size: "YOLO11S"                    # Options: YOLO11N, YOLO11S, YOLO11M
  fp16: false                        # True to use FP16 precision for better performance
  confidence: 0.5                    # Confidence threshold for detections (0.0 - 1.0)
  iou_threshold: 0.45                # IoU threshold for NMS (Non-Maximum Suppression)
  image_size: 640                    # Input image size for YOLO model
  override_model_path: null          # Custom model path (null to use default), e.g., "models/pytorch/yolo11s_maritime_15.pt"
  override_device: null              # Options: null (auto-detect), "NVIDIA_GPU", "M1_GPU", "CPU"

# Input Source Configuration
input:
  source: "CAMERA"                    # Options: "CAMERA", "MOVIE"
  camera_index: 0                     # Camera index (0 for default/built-in, 1+ for external cameras)
  movie_path: null                    # Absolute path to movie file
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
        model_config = ModelConfig(
            size=_map_model_size(config_data["model"]["size"]),
            fp16=bool(config_data["model"]["fp16"]),
            override_model_path=config_data["model"]["override_model_path"],
            confidence=float(config_data["model"]["confidence"]),
            override_device=_map_device(config_data["model"]["override_device"]),
            image_size=int(config_data["model"]["image_size"]),
            iou_threshold=float(config_data["model"]["iou_threshold"])
        )
                
        input_config = InputConfig(
            source=_map_input_source(config_data["input"]["source"]),
            camera_index=int(config_data["input"]["camera_index"]),
            movie_path=str(config_data["input"]["movie_path"])
        )
        
        return NeptuneEyeConfig(
            model=model_config,
            input=input_config,
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
    if not 0.0 <= config.model.confidence <= 1.0:
        raise ValueError(f"Model confidence must be between 0.0 and 1.0, got {config.model.confidence}")
    
    # Validate IoU threshold
    if not 0.0 <= config.model.iou_threshold <= 1.0:
        raise ValueError(f"IoU threshold must be between 0.0 and 1.0, got {config.model.iou_threshold}")
    
    # Validate image size
    if config.model.image_size <= 0:
        raise ValueError(f"Image size must be positive, got {config.model.image_size}")
    
    # Validate camera index
    if config.input.camera_index < 0:
        raise ValueError(f"Camera index must be non-negative, got {config.input.camera_index}")
    
    # Validate model path if provided
    if config.model.override_model_path is not None:
        root_dir = find_project_root()
        model_path = root_dir / config.model.override_model_path
        if not model_path.exists():
            raise ValueError(f"Override model path does not exist: {model_path}")
    
    # Validate movie path if using movie input
    if config.input.source == InputSource.MOVIE:
        if not config.input.movie_path:
            raise ValueError("Movie path must be set when input source is MOVIE")
        root_dir = find_project_root()
        movie_path = root_dir / config.input.movie_path
        if not movie_path.exists():
            raise ValueError(f"Movie file does not exist: {movie_path}")