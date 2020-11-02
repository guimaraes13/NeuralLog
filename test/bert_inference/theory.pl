label("[PAD]").
label("B").
label("I").
label("O").
label("X").
label("[CLS]").
label("[SEP]").

ner(X, Y) :- bert(X, W), dense(W, Y).
#ner(X, Y) :- bert(X, W).

type_ner(X, Y) :- ner(X, Y), label(Y).
empty_entry("<EMPTY>").

# First, it applies call the BERT layer.
set_predicate_parameter("bert/2", function_value, class_name, "bert").
set_predicate_parameter("bert/2", function_value, config, model_path, "BERT_MODEL_PATH").

# Then, it applies a dense layer on the output of the BERT layer.
# The output size of the dense layer is equal to the number of labels found in
#   the label/1 predicate.
# Then, it applies a softmax function as the final output
set_predicate_parameter("dense/2", function_value, class_name, "Dense").
set_predicate_parameter("dense/2", function_value, config, units, "$label/1[0]").
set_predicate_parameter("dense/2", function_value, config, activation, "softmax").

set_parameter(dataset_class, class_name, language_dataset).
set_parameter(dataset_class, config, inverse_relations, false).
# set_parameter(dataset_class, config, vocabulary_file, "{VOCABULARY_FILE}").
set_parameter(dataset_class, config, initial_token, "[CLS]").
set_parameter(dataset_class, config, final_token, "[SEP]").
set_parameter(dataset_class, config, pad_token, "[PAD]").
set_parameter(dataset_class, config, sub_token_label, "X").
set_parameter(dataset_class, config, maximum_sentence_length, 64).
set_parameter(dataset_class, config, pad_to_maximum_length, true).
set_parameter(dataset_class, config, do_lower_case, false).

set_parameter(epochs, 5).
set_parameter(optimizer, class_name, sgd).
set_parameter(optimizer, config, lr, 0.001).
