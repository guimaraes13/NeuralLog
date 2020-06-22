father(andrew, jennifer).
father(christopher, victoria).
father(james, victoria).
father(marco, sophia).
father(pierro, angela).
father(roberto, emilio).
-0.5::father(roberto, lucia).

mother(christine, jennifer).
mother(francesca, angela).
mother(lucia, sophia).
mother(maria, emilio).
mother(maria, lucia).
mother(penelope, victoria).
mother(victoria, charlotte).

parents(X, Y, Z) :- mother(X, Z), father(Y, Z).

learn(father).

set_predicate_parameter("father/2", initial_value, class_name, random_normal).
set_predicate_parameter("father/2", initial_value, config, mean, 0.5).
set_predicate_parameter("father/2", initial_value, config, stddev, 0.025).

set_predicate_parameter("father/2", value_constraint, class_name, partial).
set_predicate_parameter(
    "father/2", value_constraint, config, function_name, "tf.clip_by_value").
set_predicate_parameter(
    "father/2", value_constraint, config, clip_value_min, 0.0).
set_predicate_parameter(
    "father/2", value_constraint, config, clip_value_max, 1.0).
