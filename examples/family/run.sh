# Running
export PYTHONPATH=../..
PYTHON_SCRIPT=../../src/run/main.py

python3 ${PYTHON_SCRIPT} train \
    --program program.pl facts.pl \
    --train train.pl \
    --validation validation.pl \
    --test test.pl \
    --logFile data/log.txt \
    --outputPath data \
    --lastModel last_model \
    --lastProgram last_program.pl \
    --lastInference last_ \
    --verbose
