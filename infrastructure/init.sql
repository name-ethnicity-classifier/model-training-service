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

INSERT INTO public.model (accuracy, nationalities, scores, is_trained, is_grouped, is_public, public_name, creation_time, id) 
VALUES (null, '{else,chinese}', null, 
        false, false, true, 'chinese_and_else', '2021-07-20 13:41:00', 'cf58c0536d2ab4fbd6a6');
INSERT INTO public.model (accuracy, nationalities, scores, is_trained, is_grouped, is_public, public_name, creation_time, id) 
VALUES (null, '{african,eastAsian,european}', null, 
        false, true, true, '3_nationality_groups', '2021-10-14 16:44:00', '08205d420e9342228e68');
INSERT INTO public.model (accuracy, nationalities, scores, is_trained, is_grouped, is_public, public_name, creation_time, id) 
VALUES (78.51, '{else,british,indian,spanish,german,italian,french,chinese,japanese,dutch,russia}', 
        '{0.3926999866962433,0.6724200248718262,0.8446999788284302,0.8085700273513794,0.7493199706077576,0.8261500000953674,0.7379400134086609,0.9672300219535828,0.9860699772834778,0.7153800129890442,0.9139000177383423}', 
        true, false, true, '10_nationalities_and_else', '2021-09-07 12:11:00', '8129a4fb8a3f0d7f9e4c');