%% Initial value parameter for the weights
%% He instantiation, states that the weight values must be drawn from a
%% standard uniform random variable multiplied by sqrt(1/s), where `s` is the
%% size of the layer, in the case of a normal network. In our case, the size is
%% the number of rules that summed to compose the output.
%% From statistics, we have that a standard normal variable X multiplied by a
%% real constant c results in a new normal random variable with mean
%% c * mean(X) and standard deviation (c^2) * std(X).
%% Since s = 4 and 10 for this program, we have that c = sqrt(1/s), thus the
%% standard deviation should be c^2 = 1/4 = 0.25 and 1/10 = 0.1.

set_predicate_parameter("w1/1", initial_value, class_name, random_normal).
set_predicate_parameter("w1/1", initial_value, config, mean, 0.0).
set_predicate_parameter("w1/1", initial_value, config, stddev, 0.25).

set_predicate_parameter("w2/1", initial_value, class_name, random_normal).
set_predicate_parameter("w2/1", initial_value, config, mean, 0.0).
set_predicate_parameter("w2/1", initial_value, config, stddev, 0.1).

set_predicate_parameter("b/1", initial_value, zero).

set_parameter(loss_function, binary_crossentropy).
set_parameter(metrics, accuracy).

set_parameter(optimizer, class_name, adagrad).
set_parameter(optimizer, config, lr, 0.1).

set_parameter(batch_size, 8).

set_parameter(shuffle, "True").

set_parameter(epochs, 50).

learn(w1).
learn(w2).
learn(b).

for i in {0..15} do
    hidden_{i}(X) :- sepal_length(X, Y), w1(h1_{i}_1).
    hidden_{i}(X) :- sepal_width(X, Y), w1(h1_{i}_2).
    hidden_{i}(X) :- petal_length(X, Y), w1(h1_{i}_3).
    hidden_{i}(X) :- petal_width(X, Y), w1(h1_{i}_4).
    hidden_{i}(X) :- b(h_{i}).
done

activation_{i}(X) :- hidden_{i}(X), sigmoid(X).

for type in setosa versicolor virginica do
    output_{type}(X) :- activation_{i}(X), w2(output_{type}_{i}).
    output_{type}(X) :- b(output_{type}).

    iris_{type}(X) :- output_{type}(X), sigmoid(X).
done
