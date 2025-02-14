import logging
import sys
from pathlib import Path
from datetime import datetime
import colorama
from colorama import Fore, Back, Style
from typing import Optional, Union
import re
import traceback
import copy
import os

# Initialize colorama
colorama.init()

class ColoredFormatter(logging.Formatter):
    """Colored formatter for structured log output.
    
    This formatter adds color-coding, icons, timestamps, and file location
    information to log messages. It supports different color schemes for
    different log levels and includes special formatting for exceptions.

    Attributes:
        COLORS (dict): Color schemes for different log levels, including:
            - color: Foreground color
            - style: Text style (dim, normal, bright)
            - icon: Emoji icon for the log level
            - bg: Background color (for critical logs)
    """

    COLORS = {
        'DEBUG': {
            'color': Fore.CYAN,
            'style': Style.DIM,
            'icon': '🔍'
        },
        'INFO': {
            'color': Fore.GREEN,
            'style': Style.NORMAL,
            'icon': 'ℹ️'
        },
        'WARNING': {
            'color': Fore.YELLOW,
            'style': Style.BRIGHT,
            'icon': '⚠️'
        },
        'ERROR': {
            'color': Fore.RED,
            'style': Style.BRIGHT,
            'icon': '❌'
        },
        'CRITICAL': {
            'color': Fore.WHITE,
            'style': Style.BRIGHT,
            'bg': Back.RED,
            'icon': '💀'
        }
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with color and structure.

        This method formats log records with:
        - Timestamp in HH:MM:SS.mmm format
        - File location (filename:line)
        - Color-coded level name with icon
        - Color-coded message
        - Formatted exception traceback if present

        Args:
            record (logging.LogRecord): Log record to format.

        Returns:
            str: Formatted log message with color and structure.
        """
        colored_record = copy.copy(record)

        # Get color scheme
        scheme = self.COLORS.get(record.levelname, {
            'color': Fore.WHITE,
            'style': Style.NORMAL,
            'icon': '•'
        })

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]

        # Get file location
        file_location = f"{os.path.basename(record.pathname)}:{record.lineno}"

        # Build components
        components = []

        # log formatting
        components.extend([
            f"{Fore.BLUE}{timestamp}{Style.RESET_ALL}",
            f"{Fore.WHITE}{Style.DIM}{file_location}{Style.RESET_ALL}",
            f"{scheme['color']}{scheme['style']}{scheme['icon']} {record.levelname:8}{Style.RESET_ALL}",
            f"{scheme['color']}{record.msg}{Style.RESET_ALL}"
        ])

        # Add exception info
        if record.exc_info:
            components.append(
                f"\n{Fore.RED}{Style.BRIGHT}"
                f"{''.join(traceback.format_exception(*record.exc_info))}"
                f"{Style.RESET_ALL}"
            )

        return " | ".join(components)


def setup_logger(
        name: Optional[Union[str, Path]] = None,
        level: int = logging.INFO
) -> logging.Logger:
    """Set up a colored logger

    This function creates or retrieves a logger with colored output and
    automatic log interception. If a file path is provided as the name,
    it will use the filename (without extension) as the logger name.

    Args:
        name (Optional[Union[str, Path]], optional): Logger name or __file__ for
            module name. Defaults to None.
        level (int, optional): Logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Get logger name from file path
    if isinstance(name, (str, Path)) and Path(name).suffix == '.py':
        name = Path(name).stem

    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Only add handler if none exists
    if not logger.handlers:
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredFormatter())
        logger.addHandler(console_handler)
        
    return logger
