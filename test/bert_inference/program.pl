%% Sets a checkpoint to save the best model
%% The metric mean_reciprocal_rank_validation_set, which is computed by the
%% link prediction callback defined above, will be used to decide which model
%% is the best
set_parameter(callback, validation_checkpoint, class_name, "ModelCheckpoint").
set_parameter(callback, validation_checkpoint, config, monitor, val_mean_squared_error).
set_parameter(callback, validation_checkpoint, config, save_best_only, "True").
set_parameter(validation_period, 1). %% the model will be evaluated every epoch

%% Setting the parameters of the model
%% Sets the individual loss function for each predicate
%% Default: `mean_squared_error`
set_parameter(loss_function, mean_squared_error).
set_parameter(inverse_relations, "False").
set_parameter(mask_predictions, "True").
set_parameter(recursion_depth, 1).

%% Sets a single loss function for the model
%% Default: the loss function
%% set_parameter(loss_function, binary_crossentropy).

%% Defines individual metrics for each predicate
% set_parameter(metrics, grand_father, mean_squared_error).
% set_parameter(metrics, grand_mother, mean_absolute_error).
set_parameter(metrics, all, "CategoricalAccuracy").
#set_parameter(metrics, all, "Precision"). %% defines for all predicates
#set_parameter(metrics, all, "Recall"). %% defines for all predicates
%% set_parameter(metrics, all, poisson). %% possibly, defines more metrics

%% Optionally, defines the accuracy metric for all the outputs
%% In this way, we can only define a single metric
%% set_parameter(metrics, accuracy).

%% Sets the optimizer
set_parameter(optimizer, class_name, adagrad).
set_parameter(optimizer, config, lr, 0.1).

%% Sets the regularizer
set_parameter(regularizer, l2).

%% Sets the number of epochs, default: `10`
set_parameter(epochs, 10).

%% If `True`, shuffles the train set, default: `False`
set_parameter(shuffle, "True").

%% Sets the validation period. If 1, evaluate the model on the validation set
%% every epoch
set_parameter(validation_period, 1).

%% Sets a dictionary whose keys points to ModelCheckpoint callbacks
%% In this case, the validation_checkpoint
%% It will save the program and inference for the best model saved by
%% validation_checkpoint, at the output directory with prefix: `best_`
%% Default: `None`
% set_parameter(best_model, validation_checkpoint, best_).

%% Sets the default parameters
%% If omitted, the parameters look like follows

%% Initial value parameter for variables
set_parameter(initial_value, class_name, random_normal).
set_parameter(initial_value, config, mean, 0.5).
set_parameter(initial_value, config, stddev, 0.125).

set_predicate_parameter("b/1", initial_value, zero).

%% Default parameters for network functions
%% function to get the value of a negated literal from the non-negated one
set_parameter(literal_negation_function, literal_negation_function).
set_parameter("literal_negation_function:sparse",
              "literal_negation_function:sparse").


%% function to combine the different proves of a literal
%% (FactLayers and RuleLayers). The default is to sum all the
%% proves, element-wise, by applying the `tf.math.add` function to
%% reduce the layers outputs
set_parameter(literal_combining_function, "tf.math.add").

%% function to combine different vector and get an `AND` behaviour
%% between them.
%% The default is to multiply all the paths, element-wise, by applying
%% the `tf.math.multiply` function.
set_parameter(and_combining_function, "tf.math.multiply").

%% function to combine different path from a RuleLayer. The default
%% is to multiply all the paths, element-wise, by applying the
%% `tf.math.multiply` function
set_parameter(path_combining_function, "tf.math.multiply").

%% element is used to extract the tensor value of grounded literal
%% in a rule. The default edge combining function is the element-wise
%% multiplication. Thus, the neutral element is `1.0`, represented by
%% `tf.constant(1.0)`.
set_parameter(edge_neutral_element, class_name, "tf.constant").
set_parameter(edge_neutral_element, config, value, 1.0).


%% function to extract the value of the fact based on the input.
%% The default is the element-wise multiplication implemented by the
%% `tf.math.multiply` function
set_parameter(edge_combining_function, "tf.math.multiply").

%% function to extract the value of the fact based on the input,
%% for 2d facts. The default is the dot multiplication implemented
%% by the `tf.matmul` function
set_parameter(edge_combining_function_2d, "tf.matmul").
set_parameter("edge_combining_function_2d:sparse",
              "edge_combining_function_2d:sparse").

%% function to extract the inverse of a facts. The default is the
%% transpose function implemented by `tf.transpose`
set_parameter(invert_fact_function, "tf.transpose").
set_parameter("invert_fact_function:sparse", "tf.sparse.transpose").

%% function to aggregate the input of an any predicate. The default
%% function is the `tf.reduce_sum`.
set_parameter(any_aggregation_function, any_aggregation_function).

%% function to combine the numeric terms of a fact.
%% The default function is the `tf.math.multiply`.
set_parameter(attributes_combine_function, "tf.math.multiply").

%% function to combine the weights and values of the attribute facts.
%% The default function is the `tf.math.multiply`.
set_parameter(weighted_attribute_combining_function, "tf.math.multiply").

%% function to extract the value of an atom with a constant at the
%% last term position.
%% The default function is the `tf.nn.embedding_lookup`.
set_parameter(output_extract_function, "tf.nn.embedding_lookup").

%% function to extract the value of unary prediction.
%% The default is the dot multiplication, implemented by the `tf.matmul`,
%% applied to the transpose of the literal prediction.
set_parameter(unary_literal_extraction_function,
              unary_literal_extraction_function).
