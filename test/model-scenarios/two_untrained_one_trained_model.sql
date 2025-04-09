-- add untrained model
INSERT INTO public.model (accuracy, nationalities, scores, is_trained, is_grouped, is_public, public_name, creation_time, id) 
VALUES (null, '{else,german}', null, 
        false, false, true, 'german_and_else', '2021-07-20 13:41:00', 'dummy-id-1');

-- add trained model
INSERT INTO public.model (accuracy, nationalities, scores, is_trained, is_grouped, is_public, public_name, creation_time, id) 
VALUES (78.51, '{else,german,greek,vietnamese,indonesian,columbian,mexican,italian,japanese}', 
        '{0.3926999866962433,0.6724200248718262,0.8446999788284302,0.8085700273513794,0.7493199706077576,0.8261500000953674,0.7379400134086609,0.9672300219535828,0.9860699772834778}', 
        true, false, true, '8_nationalities_and_else', '2021-09-07 12:11:00', 'dummy-id-2');

-- add untrained model with nationality groups
INSERT INTO public.model (accuracy, nationalities, scores, is_trained, is_grouped, is_public, public_name, creation_time, id) 
VALUES (null, '{else,european,asian,southAmerican}', null, 
        false, true, true, '3_nationality_groups_else', '2021-10-14 16:44:00', 'dummy-id-3');