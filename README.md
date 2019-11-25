# NeuralLog

NeuralLog is a system to compile logic programs into Artificial Neural Networks.

## Dependencies
- python3;
- tensorflow 2.0;
- antlr4-python3-runtime;

### Logic Syntax

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

With these elements we can have facts and rules. A fact is represented by an **atom** ending with period. For instance:

`isMarried(john, mary).`

Additionally, similar to [ProbLog](https://dtai.cs.kuleuven.be/problog/), 
facts may have weights. This is represented as follows:

`0.5::isMarried(john, mary).`

And a rule is represented by an **atom** followed by an implication symbol (:-) and a set of atoms separated by commas,
forming its body, ending with a period, for instance:

`isParent(A, B) :- isMarried(A, C), isParent(C, B).`

## Run

The main file is src/run/neurallog.py.

# TODO: create a run example.
# TODO: add reference for the kinship example.
