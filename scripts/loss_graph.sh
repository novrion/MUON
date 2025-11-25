#!/bin/bash
g++ -o scripts/loss_graph scripts/loss_graph.cpp
./scripts/loss_graph
rm scripts/loss_graph
pdflatex data/loss_graph.tex
rm *.aux *.log
mv loss_graph.pdf data/
rm data/loss_graph.tex
