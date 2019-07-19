#!/usr/bin/env bash
export CLASSPATH=".:/usr/local/lib/antlr-4.7.2-complete.jar:$CLASSPATH"
rm NeuralLog*.java NeuralLog*.class
java -jar /usr/local/lib/antlr-4.7.2-complete.jar NeuralLog.g4
javac *.java

