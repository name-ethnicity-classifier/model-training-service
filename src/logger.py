import logging
from datetime import datetime

class CustomFormatter(logging.Formatter):
    def format(self, record):
        log_time = datetime.now().strftime("%S:%M:%H %d-%m-%Y")
        log_level = record.levelname
        message = record.getMessage()
        return f"{log_time} [{log_level}]: {message}"

def get_logger(name="CustomLogger", level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    handler = logging.StreamHandler()
    handler.setFormatter(CustomFormatter())
    
    if not logger.hasHandlers():
        logger.addHandler(handler)
    
    return logger


logger = get_logger()