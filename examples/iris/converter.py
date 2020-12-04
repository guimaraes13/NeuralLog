"""
Converts the Iris dataset.
"""

DATA = "iris.data"

KB_FILE = "facts.pl"
TRAIN_FILE = "train.pl"
TEST_FILE = "test.pl"

ATTRIBUTES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

CLASSES = {"setosa", "versicolor", "virginica"}

OTHER_LABELS = {}
for clazz in CLASSES:
    OTHER_LABELS[clazz] = CLASSES - {clazz}

TRAIN_SIZE = 35


def main():
    """The main function."""
    count = 0
    examples_by_class = {"setosa": 0, "versicolor": 0, "virginica": 0}
    kb = open(KB_FILE, "w")
    train = open(TRAIN_FILE, "w")
    test = open(TEST_FILE, "w")
    for line in open(DATA):
        line = line.strip()
        if line == "":
            continue
        fields = line.split(",")
        identifier = "e_{}".format(count)
        for key, value in zip(ATTRIBUTES, fields[:-1]):
            kb.write(f"{key}({identifier}, {value}).\n")
        kb.write("\n")
        label = fields[-1][fields[-1].index("-") + 1:]
        size = examples_by_class[label]

        if size < TRAIN_SIZE:
            writer = train
        else:
            writer = test
        writer.write(f"example(iris_{label}, {identifier}).\n")
        for other_label in OTHER_LABELS[label]:
            writer.write(f"0.0::example(iris_{other_label}, {identifier}).\n")
        examples_by_class[label] += 1
        count += 1

    print(f"{count} example(s).")
    kb.close()
    train.close()
    test.close()


if __name__ == '__main__':
    main()
