import sys
from datetime import datetime

from loguru import logger as _logger

from .constant import LOG_DIR


def define_log_level(print_level="INFO", logfile_level="DEBUG"):
    """
    Initialize the logger instance using the given log levels.

    :param print_level: Log level used to print the log.
    :param logfile_level: Log level used to save to the logfile.
    :return: None
    """
    current_date = datetime.now()
    formatted_date = current_date.strftime("%Y%m%d_%H")
    log_name = f"run_{formatted_date}"

    log_format = ('<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | '
                  '<cyan>{file}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>')

    _logger.remove()
    _logger.add(sys.stderr, level=print_level, format=log_format)
    _logger.add(f"{LOG_DIR}/{log_name}.log", level=logfile_level, format=log_format)
    return _logger


logger = define_log_level().opt(colors=True)


def error_and_raise(msg: str):
    logger.error(f"Error: {msg}")
    raise Exception(f"Error: {msg}")
