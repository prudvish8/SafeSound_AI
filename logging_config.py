#!/usr/bin/env python3
"""
Centralized logging configuration for the WhatsApp chatbot backend.

This module provides a consistent logging setup across all application modules.
It configures loggers with proper formatting, levels, and handlers.
"""

import logging
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    log_to_file: bool = False,
    log_file_path: str = "app.log"
) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        level: Logging level (default: INFO)
        format_string: Custom format string for log messages
        log_to_file: Whether to log to file in addition to console
        log_file_path: Path to log file if log_to_file is True
        
    Returns:
        Configured logger instance
    """
    # Default format if none provided
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear any existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_to_file:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Configure default logging when module is imported
setup_logging() 