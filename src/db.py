import os
import sqlalchemy
from sqlalchemy import create_engine
from dotenv import load_dotenv
from logger import logger


def create_db_connection() -> sqlalchemy.Connection:
    load_dotenv()

    db_host = os.environ.get("POSTGRES_HOST")
    db_port = os.environ.get("POSTGRES_PORT")
    db_name = os.environ.get("POSTGRES_DB")
    db_user = os.environ.get("POSTGRES_USER")
    db_password = os.environ.get("POSTGRES_PASSWORD")

    db_uri = f"postgresql://{db_host}:{db_port}/{db_name}?user={db_user}&password={db_password}"

    pg_engine = create_engine(db_uri, echo=False)
    connection = pg_engine.connect()

    logger.info("Connected to DB.")

    return connection
