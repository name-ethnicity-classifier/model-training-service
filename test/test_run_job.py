import json
import sqlalchemy
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text, MetaData
from unittest.mock import patch
from db import create_db_connection
import pytest
from run_jobs import main
from s3 import S3Handler
from config import config


@pytest.fixture(scope="session")
def db_connection():
    connection = create_db_connection()
    yield connection
    connection.close()
    
    
@pytest.fixture
def db_session(db_connection):
    SessionLocal = sessionmaker(bind=db_connection)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(autouse=True)
def init_empty_database(db_connection):
    metadata = MetaData()
    metadata.reflect(bind=db_connection)
    metadata.drop_all(bind=db_connection)

    with open("test/test_init.sql", "r") as f:
        sql_commands = f.read()
        for statement in sql_commands.split(";"):
            if statement.strip():
                db_connection.execute(text(statement))
                
    db_connection.commit()
    yield
    

@pytest.fixture(autouse=True)
def init_testing_s3_bucket():
    S3Handler.clear_bucket(config.model_bucket)


@pytest.fixture(scope="function", autouse=True)
def mock_base_model_config():
    with patch("training.train_utils.load_model_config") as mock_config:
        with open("./test/mock/model_config.json", "r") as f:
            mock_config.return_value = json.load(f)

        yield mock_config


@pytest.fixture
def db_scenario(db_session, request):
    with open(f"test/model_scenarios/{request.param}", "r") as f:
        sql_commands = f.read()
        for statement in sql_commands.split(";"):
            if statement.strip():
                db_session.execute(text(statement))
                
    db_session.commit()


@pytest.mark.it("should train all untrained models and update S3 bucket and database")
@pytest.mark.parametrize("db_scenario", ["two_untrained_one_trained_model.sql"], indirect=True)
def test_run_jobs(db_session, db_scenario):
    untrained_model_ids = db_session.execute(sqlalchemy.text("SELECT id FROM model WHERE is_trained = false;")).all()
    untrained_model_ids = [id_[0] for id_ in untrained_model_ids]
    
    main()
    
    rows = db_session.execute(sqlalchemy.text("SELECT id, accuracy, scores, is_trained FROM model;")).all()
    for row in rows:
        model_id, accuracy, scores, is_trained = row
        
        # Only interested in the models that where previously untrained
        if model_id not in untrained_model_ids:
            continue
        
        assert accuracy != None
        assert scores != None
        assert is_trained == True
        
        expected_s3_files = [f"{model_id}/logs.json", f"{model_id}/model.pt", f"{model_id}/dataset.pickle"]
        for expected_file in expected_s3_files:
            assert S3Handler.get(config.model_bucket, expected_file)
            
            
@pytest.mark.it("should not do anything when there are no untrained models")
@pytest.mark.parametrize("db_scenario", ["one_trained_model.sql"], indirect=True)
def test_run_no_jobs(db_session, db_scenario, caplog):
    main()
    
    expected_output = "No models to train. Exiting"
    assert expected_output in caplog.text