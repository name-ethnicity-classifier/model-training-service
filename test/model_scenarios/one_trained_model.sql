
-- add trained model
INSERT INTO public.model (accuracy, nationalities, scores, is_trained, is_grouped, is_public, public_name, creation_time, id) 
VALUES (78.51, '{else,british,indian,spanish,german,italian,french,chinese,japanese,dutch,russia}', 
        '{0.3926999866962433,0.6724200248718262,0.8446999788284302,0.8085700273513794,0.7493199706077576,0.8261500000953674,0.7379400134086609,0.9672300219535828,0.9860699772834778,0.7153800129890442,0.9139000177383423}', 
        true, false, true, '10_nationalities_and_else', '2021-09-07 12:11:00', '8129a4fb8a3f0d7f9e4c');