CREATE TABLE model (
    nationalities TEXT[]                 NOT NULL,
    accuracy      DOUBLE PRECISION,
    scores        REAL[],
    is_trained    BOOLEAN DEFAULT false  NOT NULL,
    is_grouped    BOOLEAN DEFAULT false  NOT NULL,
    is_public     BOOLEAN DEFAULT false  NOT NULL,
    public_name   VARCHAR(64),
    creation_time VARCHAR(64)            NOT NULL,
    request_count INTEGER DEFAULT 0      NOT NULL,
    id            VARCHAR(40)            NOT NULL CONSTRAINT model_pk PRIMARY KEY
);

CREATE TYPE access_level AS ENUM ('admin', 'full', 'restricted');

CREATE TABLE "user" (
    email               VARCHAR(320)           NOT NULL,
    password            VARCHAR(64)            NOT NULL,
    name                VARCHAR(64)            NOT NULL,
    role                VARCHAR(32)            NOT NULL,
    signup_time         VARCHAR(64)            NOT NULL,
    verified            BOOLEAN DEFAULT false  NOT NULL,
    consented           BOOLEAN DEFAULT false  NOT NULL,
    request_count       INTEGER DEFAULT 0      NOT NULL,
    names_classified    INTEGER DEFAULT 0      NOT NULL,
    usage_description   VARCHAR(500)           NOT NULL,
    access access_level DEFAULT         'full' NOT NULL,
    id                  SERIAL                 NOT NULL CONSTRAINT user_pk PRIMARY KEY
);



CREATE TABLE user_to_model (
    user_id             INTEGER            NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    model_id            VARCHAR(40)        NOT NULL REFERENCES "model"(id) ON DELETE CASCADE,
    name                VARCHAR(64)        NOT NULL,
    description         VARCHAR(512),
    request_count       INTEGER DEFAULT 0  NOT NULL,
    id                  SERIAL             NOT NULL CONSTRAINT user_to_model_pk PRIMARY KEY
);


CREATE TABLE user_quota (
    user_id      INTEGER      NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    last_updated DATE         NOT NULL DEFAULT CURRENT_DATE,
    name_count   INTEGER      NOT NULL DEFAULT 0,
    id           SERIAL       NOT NULL CONSTRAINT user_quota_pk PRIMARY KEY
);


ALTER TABLE model OWNER TO postgres;
