# NeuralLog

NeuralLog is a system to compile logic programs into Artificial Neural Networks.

## Dependencies

The system runs using the following libraries in the version described here. 
Different versions of the libraries may work as well, although it is not
 tested. 

- python 3.7;
- numpy 1.19.1;
- scipy 1.5.2;
- scikit-learn 0.23.2;
- sklearn 0.0;
- tensorflow 2.0.1;
- tensorflow-addons 0.6.0;
- ply 3.11;
- PyYAML 5.3.1;
- matplotlib 3.3.1;
- [bert-for-tf2](https://github.com/kpe/bert-for-tf2);

## Logic Syntax

The logic syntax is simple, it is based on a function-free logic similar to 
Prolog. The elements are defined as follows:
- A **variable** is represented by a string of letters, digits or underscores, 
starting with an upper case letter;
- A **constant** is represented like a variable, but starting with a lower case 
letter;
- A **term** is either a constant or a variable;
- An **atom** is represented by a string of letters, digits or underscores, 
starting with a lower case letter followed by
a n-tuple of terms between brackets; thus, we say the predicate has arity n.

With these elements we can have facts and rules. A fact is represented by an
**atom** ending with period. For instance:

`isMarried(john, mary).`

Additionally, similar to [ProbLog](https://dtai.cs.kuleuven.be/problog/), 
facts may have weights. This is represented as follows:

`0.5::isMarried(john, mary).`

And a rule is represented by an **atom** followed by an implication symbol (:-)
and a set of atoms separated by commas, forming its body, ending with a
 period, for instance:

`isParent(A, B) :- isMarried(A, C), isParent(C, B).`

## Parameter Learning

The main entry file is neurallog/run/main.py. From there, one can train and
evaluate the models.

### Training

In order to run the system, first we need to set the PYTHONPATH envelopment
variable to the system's directory.

```
cd NeuralLog
export PYTHONPATH=$(pwd)
```

Then, the following command trains a neural network, evaluates it and saves the
results.

```
python3 neurallog/run/main.py train \ 
    --program examples/family/program.pl examples/family/facts.pl \
    --train examples/family/train.pl \
    --validation examples/family/validation.pl \
    --test examples/family/test.pl \
    --logFile examples/family/data/log.txt \
    --outputPath examples/family/data \
    --lastModel last_model \
    --lastProgram last_program.pl \
    --lastInference last_ \
    --verbose
```

The parameters are as follows:

- program: the program files that includes the configuration and the
 background
knowledge containing rules and common facts;
- train: the training examples;
- validation: the validation examples;
- test: the test examples;
- logFile: path for the log file, this path is not relative to the `outputPath`;
- outputPath: the path to save the data;
- lastModel: the path to save the final model, relative to outputPath;
- lastProgram: the path to save the last program (the program from the last
model), relative to outputPath;
- lastInference: the path to save the inferences of the last model, relative to
outputPath;
- verbose: increases the log details.

This command will create the neural network based on the `program.pl` and 
`facts.pl` files. Then, it will train the network on the examples from 
`train.pl`, evaluating it, on the validation.pl examples, every
`validation_period` epoch. Finally, it will save the last learned model, the
program generated from this model and the inferences on the three sets: train,
validation and test. Where `validation_period` is defined in `program.pl`. 

Since we defined a `ModelCheckpoint` in `program.pl`, it will save the models,
based on its configuration. In this case, the model which achieves the best
mean reciprocal rank at the validation set. At the end, as we defined the
parameter `best_model` to point to this `ModelCheckpoint`, the best model
saved by it will be loaded and the program and inferences of this model will be
saved in `outputPath` with prefix `best_`, defined in `program.pl`. 

### Inference

In order to perform inference using a previous trained model, without training,
one can use the following command:

```
python3 neurallog/run/main.py train \ 
    --program examples/family/program.pl examples/family/data/best_program.pl \
    --test examples/family/test.pl \
    --loadModel examples/family/data/mean_reciprocal_rank_validation_set_best \ 
    --logFile examples/family/data/log_eval.txt \
    --outputPath examples/family/data \
    --lastProgram best_eval_program.pl \
    --lastInference best_eval_ \
    --verbose
```

By using the `loadModel` parameter to load the best saved model
(`examples/family/data/mean_reciprocal_rank_validation_set_best`), and the
`program` parameter to load the best saved program
(`examples/family/data/best_program.pl`), the model will be load from the saved
weights.

One could possibly resume training from here, but, in this case, by passing no
training set, there will be no training.

The inferences will be saved to the file: 
`outputPath` + `lastInference` + `test_set.pl`.
Since the examples are in the test set, in this case:
`examples/family/data/best_eval_test_set.pl`

### Parameters

The list of all settable parameters can be obtained by running the following
command:

```python3 neurallog/run/main.py train --help```

## References

If you use NeuralLog on your research, please, considering cite one or more 
of the following papers:

[NeuralLog: a Neural Logic Language](https://arxiv.org/abs/2105.01442)
```
@article{vGuimaraes21a,
  author    = {Victor Guimar{\~{a}}es and
                 V{\'{\i}}tor Santos Costa},
  title     = {NeuralLog: a Neural Logic Language},
  journal   = {CoRR},
  volume    = {abs/2105.01442},
  year      = {2021},
  url       = {https://arxiv.org/abs/2105.01442},
  eprinttype = {arXiv},
  eprint    = {2105.01442},
  timestamp = {Wed, 12 May 2021 15:54:31 +0200},
}
```

[Meta-Interpretive Learning meets Neural Networks](https://www.geist.re/_media/sedami:sedami2021-victorguimaraes.pdf)
```
@article{vGuimaraes21b,
  author = {Guimar{\~a}es, Victor and Costa, V{\'i}tor Santos},
  year = {2021},
  month = {08},
  title = {Meta-Interpretive Learning meets Neural Networks},
  journal = {The Semantic Data Mining Workshop, SEDAMI 2021}
}
```

[Online Learning of Logic Based Neural Network Structures](https://link.springer.com/chapter/10.1007/978-3-030-97454-1_10)
```
@inproceedings{vGuimaraes21c,
  author = {Guimar{\~a}es, Victor and Costa, V{\'i}tor Santos},
  title="Online Learning of Logic Based Neural Network Structures",
  booktitle="Inductive Logic Programming",
  year="2021",
  publisher="Springer International Publishing",
  address="Athens, Greece",
}
```
