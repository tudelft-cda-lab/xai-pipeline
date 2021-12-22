# explainable-ai-pipeline
Pipeline for running XAI experiments

Example run: 
- python main.py npy randomforest lime --xtrain=data/dataset_daniel/X_train.npy --ytrain=data/dataset_daniel/y_train.npy --xexplain=data/dataset_daniel/X_explain.npy --yexplain=data/dataset_daniel/y_explain.npy

Assumptions: 
- Input data must be split in train-test- and explain-set and provided to the file.

Add modules: 
The software is a three-stage pipeline with first a parser, then a classifier, and in the end an explainer. When adding a new module, please follow
the structure of the existing directories. The same goes for potential desired .ini files. The parser .py for example should be in src/parsers/parsers/. Then, add the parser to the factory.py module in the parser subdir dir. This way, all the module implementations are hidden away from the main module. 

One sidenote: The current parsers seem simple, and probably do not have to implemented that way. I chose to do that in case some extra implementation
to them is desired, say an automatic scaling that should take place at this stage. 