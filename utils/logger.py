from pathlib import Path


from loguru import logger

from config.app_config import LOG_DEFAULT_FILE


def configure(log_level: str = "INFO", log_file: Path = None):
    log_level_value = log_level.upper()

    if log_file:
        logger.info(f"Log file: {log_file}")
        log_file = Path(log_file).joinpath(LOG_DEFAULT_FILE)
        log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.add(log_file, level=log_level_value)
    logger.info(f"Logger set up with log level: {log_level_value}")


