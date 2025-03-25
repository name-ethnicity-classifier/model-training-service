
from functools import wraps
import traceback
from sqlalchemy.exc import SQLAlchemyError 


class GeneralError(Exception):
    def __init__(self, error_code: str, message: str, status_code: int):
        self.error_code = error_code
        self.message = message
        self.status_code = status_code


def error_handler(func: callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except GeneralError as e:
            print(f"Custom error: {e.message}")
        except SQLAlchemyError as e:
            print(f"Database error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}. Traceback:\n{traceback.format_exc()}")
    return wrapper
