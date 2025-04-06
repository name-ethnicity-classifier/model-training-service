from logger import logger
from functools import wraps
import traceback
from sqlalchemy.exc import SQLAlchemyError 
from botocore.exceptions import ClientError


def error_handler(func: callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SQLAlchemyError as e:
            logger.error(f"Database error: {e}")
        except ClientError as e:
            logger.error(f"S3 error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}. Traceback:\n{traceback.format_exc()}")
    return wrapper
