"""
Configuration module for Neptune Eye

This module handles loading and validating configuration from YAML files.
"""
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from enum import Enum

import yaml
import torch

from object_detection.yolo_object_detection import YoloModelSize, InferenceDevice
from utilites import find_project_root

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
    device: Optional[InferenceDevice]
    model_path: Optional[str]

class NeptuneEyeConfig:
    """Complete Neptune Eye configuration."""
    
    # Default movie path (from root) if none is provided
    DEFAULT_MOVIE_PATH: str = "res/movies/boat_4.MOV"
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize Neptune Eye configuration by loading from YAML file.
        Creates a default configuration file if none exists.
        
        Args:
            config_path: Path to config file. If None, uses default config.yaml in project root.
            
        Raises:
            yaml.YAMLError: If config file has invalid YAML syntax.
            ValueError: If config values are invalid.
            RuntimeError: If unable to create default configuration file.
        """
        self._general: Optional[GeneralConfig] = None
        self._expert: Optional[ExpertConfig] = None
        self.load(config_path)
    
    @staticmethod
    def _map_enum(value: str, enum_class: type[Enum], name: str) -> Enum:
        """
        Map string value to enum.
        
        Args:
            value: String value to map
            enum_class: Enum class to map to
            name: Name of the setting for error messages
            
        Returns:
            Enum member
            
        Raises:
            ValueError: If value is not valid for the enum
        """
        try:
            return enum_class[value]
        except KeyError:
            valid_values = [e.name for e in enum_class]
            raise ValueError(f"Invalid {name}: {value}. Must be one of {valid_values}")

    @staticmethod
    def _map_device(device_str: Optional[str]) -> Optional[InferenceDevice]:
        """Map string device to InferenceDevice enum, or None if not specified."""
        return None if device_str is None else NeptuneEyeConfig._map_enum(device_str, InferenceDevice, "device")

    @staticmethod
    def _detect_best_device() -> InferenceDevice:
        """Detect the best available device for inference."""
        if torch.backends.mps.is_available():
            return InferenceDevice.M1_GPU
        if torch.cuda.is_available():
            return InferenceDevice.NVIDIA_GPU
        return InferenceDevice.CPU

    @staticmethod
    def _get_nvidia_gpu_model_path(model_size: YoloModelSize, fp16: bool) -> str:
        """Get model path for NVIDIA GPU inference."""
        precision_suffix = "16fp" if fp16 else "32fp"
        
        model_paths = {
            YoloModelSize.YOLO11N: f"engine/yolo11n_{precision_suffix}.engine",
            YoloModelSize.YOLO11S: f"engine/neptunes_{precision_suffix}.engine",
        }

        return model_paths[model_size]

    @staticmethod
    def _get_pytorch_model_path(model_size: YoloModelSize) -> str:
        """Get model path for PyTorch inference (M1 GPU or CPU)."""
        model_paths = {
            YoloModelSize.YOLO11N: "pytorch/neptunen.pt",
            YoloModelSize.YOLO11S: "pytorch/neptunes.pt",
            YoloModelSize.YOLO11M: "pytorch/yolo11m.pt",
        }

        return model_paths[model_size]

    def _resolve_movie_path(self, movie_path: Optional[str]) -> str:
        """
        Resolve the absolute path to the movie file based on configuration.
        Supports both absolute paths and relative paths (relative to project root).
        
        Args:
            movie_path: Path to the movie file (can be None)
            
        Returns:
            str: Absolute path to the movie file.
        """
        if self._general.source == InputSource.MOVIE:
            # If there is no movie path provided, use default sample movie
            if movie_path is None:
                movie_path = Path(find_project_root() / NeptuneEyeConfig.DEFAULT_MOVIE_PATH)
            else:
                provided_path = Path(movie_path)
                if not provided_path.is_absolute():
                    # Relative path - resolve from project root
                    movie_path = find_project_root() / provided_path
                else:
                    movie_path = provided_path
        
        return str(movie_path.resolve())

    def _resolve_model_path(self, user_given_model_path: Optional[str]) -> str:
        """
        Resolve the absolute path to the model file. If a user-given path is provided, it is used directly.
        Else, the best model path is determined based on device and model size.
        
        Args:
            user_given_model_path: Optional user-specified model path
            
        Returns:
            str: Absolute path to the model file.
            
        Raises:
            ValueError: If model configuration is invalid or model file doesn't exist.
        """
        # Use the user given model path if provided
        if user_given_model_path is not None:
            provided_path = Path(user_given_model_path)
            absolute_path = provided_path if provided_path.is_absolute() else find_project_root() / provided_path
            
            if not absolute_path.exists():
                raise ValueError(f"User specified model file does not exist: {absolute_path}")
            return str(absolute_path.resolve())

        # Get relative model path based on device and model size
        if self._expert.device == InferenceDevice.NVIDIA_GPU:
            relative_path = NeptuneEyeConfig._get_nvidia_gpu_model_path(self._expert.model_size, self._expert.fp16)
        else:
            relative_path = NeptuneEyeConfig._get_pytorch_model_path(self._expert.model_size)
        
        # Convert to absolute path
        absolute_path = (find_project_root() / "models" / relative_path).resolve()
        
        if not absolute_path.exists():
            raise ValueError(f"Model file does not exist: {absolute_path}. "
                            f"Expected model for {self._expert.model_size.name} on {self._expert.device.name}")
        
        return str(absolute_path)
    
    @staticmethod
    def _create_default_config_content() -> str:
        """Create default configuration file content."""
        content = """# Neptune Eye Configuration
# This file contains all configuration parameters for the Neptune Eye object detection system

# General Configuration
general:
  confidence: 0.5                    # Confidence threshold for detections (0.0 - 1.0)
  headless: true                     # True to run without showing images (headless mode), false to display images
  source: "MOVIE"                    # Options: "CAMERA", "MOVIE"
  camera_index: 0                    # Camera index (0 for default/built-in, 1+ for external cameras)
  movie_path: null                   # Path to movie file. Can be relative to root or absolute. If null is set, a sample video will be used.

# Expert Configuration (Advanced Settings)
expert:
  model_size: "YOLO11S"              # Options: YOLO11N, YOLO11S, YOLO11M
  fp16: false                        # True to use FP16 precision for better performance
  iou_threshold: 0.45                # IoU threshold for NMS (Non-Maximum Suppression)
  image_size: 640                    # Input image size for YOLO model
  override_model_path: null          # Possibility to override the used model. Can be relative to root or absolute path. null is default. 
  override_device: null              # Possibility to override the device. Options: null (best detected device for inference is used), "NVIDIA_GPU", "M1_GPU", "CPU"
"""
        return content

    @staticmethod
    def _create_default_config_file(config_path: Path) -> None:
        """Create a default configuration file at the specified path."""
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(NeptuneEyeConfig._create_default_config_content())
            print(f"Created default configuration file: {config_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to create default configuration file {config_path}: {e}")

    def _validate(self) -> None:
        """
        Validate configuration values for logical consistency.
        
        Raises:
            ValueError: If configuration values are invalid or inconsistent.
        """
        # Validate confidence threshold
        if not 0.0 <= self._general.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self._general.confidence}")
        
        # Validate IoU threshold
        if not 0.0 <= self._expert.iou_threshold <= 1.0:
            raise ValueError(f"IoU threshold must be between 0.0 and 1.0, got {self._expert.iou_threshold}")
        
        # Validate image size
        if self._expert.image_size <= 0:
            raise ValueError(f"Image size must be positive, got {self._expert.image_size}")
        
        # Validate camera index
        if self._general.camera_index < 0:
            raise ValueError(f"Camera index must be non-negative, got {self._general.camera_index}")
        
        # Validate movie path if using movie input
        if self._general.source == InputSource.MOVIE:
            movie_path = Path(self._general.movie_path)
            if not movie_path.exists():
                raise ValueError(f"Movie file does not exist: {movie_path}")

    def load(self, config_path: Optional[Path] = None) -> None:
        """
        Load Neptune Eye configuration from YAML file.
        Creates a default configuration file if none exists.
        
        Args:
            config_path: Path to config file. If None, uses default config.yaml in project root.
            
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
            self._create_default_config_file(config_path)
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in config file {config_path}: {e}")
        
        if not isinstance(config_data, dict):
            raise ValueError(f"Config file must contain a YAML object/dictionary, got {type(config_data)}")
        
        # Validate and parse configuration sections
        try:
            # Parse general config (without movie_path, will be set later)
            general_config = GeneralConfig(
                confidence=float(config_data["general"]["confidence"]),
                headless=bool(config_data["general"]["headless"]),
                source=self._map_enum(config_data["general"]["source"], InputSource, "input source"),
                camera_index=int(config_data["general"]["camera_index"]),
                movie_path=None  # Will be resolved after instance creation
            )
            
            # Parse expert config (without model_path, will be set later)
            expert_config = ExpertConfig(
                model_size=self._map_enum(config_data["expert"]["model_size"], YoloModelSize, "model size"),
                fp16=bool(config_data["expert"]["fp16"]),
                iou_threshold=float(config_data["expert"]["iou_threshold"]),
                image_size=int(config_data["expert"]["image_size"]),
                device=self._map_device(config_data["expert"]["override_device"]) or self._detect_best_device(),
                model_path=None  # Will be resolved after instance creation
            )

        except KeyError as e:
            raise ValueError(f"Missing required configuration key: {e}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid configuration value: {e}")
        
        # Set the instance attributes
        self._general = general_config
        self._expert = expert_config
        
        # Now resolve paths using instance methods (which can access instance members)
        self._expert.model_path = self._resolve_model_path(
            user_given_model_path=config_data["expert"]["override_model_path"]
        )
        
        # Handle movie_path: if null in YAML, it becomes None in Python
        movie_path_value = config_data["general"]["movie_path"]
        if movie_path_value is not None:
            movie_path_value = str(movie_path_value)
        self._general.movie_path = self._resolve_movie_path(movie_path=movie_path_value)
        
        # Validate the final configuration
        self._validate()

        # Print general configuration
        self._print_config()
    
    # General configuration getters
    def get_confidence(self) -> float:
        """Get confidence threshold for detections."""
        return self._general.confidence
    
    def get_headless(self) -> bool:
        """Get headless mode setting."""
        return self._general.headless
    
    def get_source(self) -> InputSource:
        """Get input source (CAMERA or MOVIE)."""
        return self._general.source
    
    def get_camera_index(self) -> int:
        """Get camera index for camera input."""
        return self._general.camera_index
    
    def get_movie_path(self) -> str:
        """Get movie file path for movie input."""
        return self._general.movie_path
    
    # Expert configuration getters
    def get_model_size(self) -> YoloModelSize:
        """Get YOLO model size."""
        return self._expert.model_size
    
    def get_fp16(self) -> bool:
        """Get FP16 precision setting."""
        return self._expert.fp16
    
    def get_iou_threshold(self) -> float:
        """Get IoU threshold for NMS."""
        return self._expert.iou_threshold
    
    def get_image_size(self) -> int:
        """Get input image size."""
        return self._expert.image_size
    
    def get_device(self) -> InferenceDevice:
        """Get inference device."""
        return self._expert.device
    
    def get_model_path(self) -> str:
        """Get model file path."""
        return self._expert.model_path
    
    def is_input_movie(self) -> bool:
        """Check if input source is movie file."""
        return self._general.source == InputSource.MOVIE
    
    def is_input_camera(self) -> bool:
        """Check if input source is camera."""
        return self._general.source == InputSource.CAMERA
    
    def _print_config(self) -> None:
        """Print the configuration to console."""
        print("General Configuration:")
        print(f"Confidence: {self._general.confidence}")
        print(f"Headless: {self._general.headless}")
        print(f"Source: {self._general.source.value}")
        print(f"Model path: {self._expert.model_path}")
        if self._general.source == InputSource.MOVIE:
            print(f"Movie Path: {self._general.movie_path}")
        elif self._general.source == InputSource.CAMERA:
            print(f"Camera Index: {self._general.camera_index}")
