"""
Evaluates the Iris Results.
"""

import numpy as np

from sklearn.metrics import classification_report

LABEL_FILE = "test.pl"
RESULT_FILE = "data/last_test_set.pl"

CLASSES = ["iris_setosa", "iris_versicolor", "iris_virginica"]

LABEL_INDEX = dict()
for i, clazz in enumerate(CLASSES):
    LABEL_INDEX[clazz] = i


def read_examples():
    """Reads the examples"""
    values = dict()
    for line in open(LABEL_FILE):
        line = line.strip()
        if line == "":
            continue
        if "::" in line:
            continue
        index = line.index("(")
        args = line[index + 1: line.rfind(")")]
        fields = args.split(",")
        label = fields[0].strip()
        example = fields[-1].strip()
        values[example] = label

    return values


def read_predictions():
    """Reads the predictions"""
    predictions = dict()
    for line in open(RESULT_FILE):
        line = line.strip()
        if line == "":
            continue
        value, example = line.split("::")
        value = float(value)
        index = example.index("(")
        label = example[:index]
        example = example[index + 1: example.rfind(")")]
        predictions.setdefault(example, [0.0] * 3)[LABEL_INDEX[label]] = value

    return predictions


def main():
    """The main function."""
    values = read_examples()
    predictions = read_predictions()

    y_true = []
    y_pred = []
    for example, label in values.items():
        y_true.append(LABEL_INDEX[label])
        y_pred.append(np.array(predictions[example]).argmax())

    print(classification_report(y_true, y_pred, target_names=CLASSES))


if __name__ == '__main__':
    main()
