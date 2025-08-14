"""
Pipeline utilities for the generalized financial sentiment analysis pipeline.

This module provides configuration management, state tracking, and logging utilities
for the automated pipeline system.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime


class ConfigManager:
    """Manages configuration loading and access for the pipeline."""
    
    def __init__(self, config_path: str):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration JSON file
        """
        self.config_path = Path(config_path)
        self._config = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from the JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                self._config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'training.batch_size')
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_config(self) -> None:
        """Save current configuration back to file."""
        with open(self.config_path, 'w') as f:
            json.dump(self._config, f, indent=2)
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the full configuration dictionary."""
        return self._config.copy()


class StateManager:
    """Manages pipeline state and step completion tracking."""
    
    def __init__(self, state_path: str):
        """
        Initialize the state manager.
        
        Args:
            state_path: Path to the state JSON file
        """
        self.state_path = Path(state_path)
        self._state = {}
        self.load_state()
    
    def load_state(self) -> None:
        """Load state from the JSON file, create if doesn't exist. Always ensure required keys exist."""
        try:
            with open(self.state_path, 'r') as f:
                self._state = json.load(f)
        except FileNotFoundError:
            # Initialize with empty state
            self._state = {}
        # Ensure required keys exist
        if 'pipeline_id' not in self._state:
            self._state['pipeline_id'] = datetime.now().strftime('%Y%m%d_%H%M%S')
        if 'created_at' not in self._state:
            self._state['created_at'] = datetime.now().isoformat()
        if 'steps' not in self._state:
            self._state['steps'] = {}
        if 'metadata' not in self._state:
            self._state['metadata'] = {}
        self.save_state()
    
    def save_state(self) -> None:
        """Save current state to file."""
        # Ensure directory exists
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.state_path, 'w') as f:
            json.dump(self._state, f, indent=2)
    
    def mark_step_complete(self, step_name: str, **metadata) -> None:
        """
        Mark a pipeline step as completed.
        
        Args:
            step_name: Name of the completed step
            **metadata: Additional metadata to store with the step
        """
        self._state['steps'][step_name] = {
            'completed_at': datetime.now().isoformat(),
            'status': 'completed',
            **metadata
        }
        self._state['last_updated'] = datetime.now().isoformat()
        self.save_state()
    
    def mark_step_failed(self, step_name: str, error_message: str, **metadata) -> None:
        """
        Mark a pipeline step as failed.
        
        Args:
            step_name: Name of the failed step
            error_message: Error message describing the failure
            **metadata: Additional metadata to store with the step
        """
        self._state['steps'][step_name] = {
            'completed_at': datetime.now().isoformat(),
            'status': 'failed',
            'error': error_message,
            **metadata
        }
        self._state['last_updated'] = datetime.now().isoformat()
        self.save_state()
    
    def is_step_complete(self, step_name: str) -> bool:
        """
        Check if a pipeline step is completed.
        
        Args:
            step_name: Name of the step to check
            
        Returns:
            True if step is completed, False otherwise
        """
        step_info = self._state['steps'].get(step_name, {})
        return step_info.get('status') == 'completed'
    
    def get_step_info(self, step_name: str) -> Dict[str, Any]:
        """
        Get information about a pipeline step.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Step information dictionary
        """
        return self._state['steps'].get(step_name, {})
    
    def clear_step(self, step_name: str) -> None:
        """
        Clear a pipeline step (mark as not completed).
        
        Args:
            step_name: Name of the step to clear
        """
        if step_name in self._state['steps']:
            del self._state['steps'][step_name]
        self._state['last_updated'] = datetime.now().isoformat()
        self.save_state()
    
    def reset_pipeline(self) -> None:
        """Reset the entire pipeline state."""
        self._state = {
            'pipeline_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'created_at': datetime.now().isoformat(),
            'steps': {},
            'metadata': {}
        }
        self.save_state()
    
    @property
    def state(self) -> Dict[str, Any]:
        """Get the full state dictionary."""
        return self._state.copy()


class LoggingManager:
    """Manages logging configuration for pipeline components."""
    
    def __init__(self, config: ConfigManager, component_name: str):
        """
        Initialize the logging manager.
        
        Args:
            config: ConfigManager instance for accessing logging configuration
            component_name: Name of the component (used in logger name)
        """
        self.config = config
        self.component_name = component_name
        self.logger = None
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        # Get logging configuration
        log_level = self.config.get('logging.level', 'INFO')
        log_to_file = self.config.get('logging.log_to_file', True)
        log_dir = self.config.get('logging.log_dir', 'logs')
        log_format = self.config.get(
            'logging.log_format', 
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create logger
        logger_name = f"pipeline.{self.component_name}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(log_format)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if enabled)
        if log_to_file:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            
            log_file = log_path / f"{self.component_name}.log"
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(getattr(logging, log_level.upper()))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False
    
    def get_logger(self) -> logging.Logger:
        """
        Get the configured logger instance.
        
        Returns:
            Configured logger for the component
        """
        return self.logger
    
    def set_level(self, level: str) -> None:
        """
        Set the logging level.
        
        Args:
            level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        """
        log_level = getattr(logging, level.upper())
        self.logger.setLevel(log_level)
        
        for handler in self.logger.handlers:
            handler.setLevel(log_level)


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure
        
    Returns:
        Path object for the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available compute devices.
    
    Returns:
        Dictionary with device information
    """
    import torch
    
    info = {
        'cuda_available': torch.cuda.is_available(),
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        'device_count': 0,
        'device_names': [],
        'recommended_device': 'cpu'
    }
    
    if info['cuda_available']:
        info['device_count'] = torch.cuda.device_count()
        info['device_names'] = [torch.cuda.get_device_name(i) for i in range(info['device_count'])]
        info['recommended_device'] = 'cuda'
    elif info['mps_available']:
        info['recommended_device'] = 'mps'
    
    return info


def validate_model_config(model_config: Dict[str, Any]) -> bool:
    """
    Validate a model configuration dictionary.
    
    Args:
        model_config: Model configuration to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = ['name', 'model_id', 'tokenizer_id', 'num_labels']
    
    for key in required_keys:
        if key not in model_config:
            return False
    
    return True


def format_time(seconds: float) -> str:
    """
    Format time in seconds to a human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def calculate_model_size(model) -> Dict[str, Any]:
    """
    Calculate model size information.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with size information
    """
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate size in MB (assuming float32, so 4 bytes per parameter)
    size_mb = (param_count * 4) / (1024 * 1024)
    
    return {
        'total_parameters': param_count,
        'trainable_parameters': trainable_params,
        'size_mb': size_mb,
        'size_human': f"{size_mb:.1f} MB" if size_mb < 1024 else f"{size_mb/1024:.1f} GB"
    }

