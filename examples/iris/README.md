# Family

Iris classification dataset.

## Files

iris.data: contains the original dataset, in ARFF format.

converter.py: a script to convert the iris from the ARFF to the logic format. 

facts.pl: contains the facts of the dataset, in logic format.

theory.pl: contains the configuration of the neural network, training and the
 logic theory.

train.pl, test.pl: contain the examples to train and test, respectively.

eval.py: evaluates the results.

run.sh: train the model and evaluates the results.

# References

[1] Anderson, E. (1936). The Species Problem in Iris. Annals of the Missouri
 Botanical Garden, 23, 457–509.

[2] Fisher, R. A. (1936). The Use of Multiple Measurements in Taxonomic 
Problems. Annals of Eugenics, 7, 179-188.
