# Running
export PYTHONPATH=$(cd ../..; pwd)
PYTHON_SCRIPT=${PYTHONPATH}/neurallog/run/main.py

python3 ${PYTHON_SCRIPT} train \
    --program theory.pl facts.pl \
    --train train.pl \
    --test test.pl \
    --logFile data/log.txt \
    --outputPath data \
    --lastModel last_model \
    --lastProgram last_program.pl \
    --lastInference last_ \
    --verbose

printf "\n\n"

python3 eval.py | tee data/evaluation.txt
