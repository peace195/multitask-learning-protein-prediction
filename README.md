## Descriptions

Protein multitask learning (secondary structure prediction, b-values prediction, solvent-accessibility prediction) can improve the prediction accuracy of protein secondary structure.

- We have to face the class imbalance problem
- "foldername_cv": 5 fold cross validation

## Data

The copyright belongs to http://rostlab.org/. It is not public.

## Multitask learning model

![alt text](https://raw.githubusercontent.com/peace195/protein-prediction/master/multitask.jpg)

## Results

* Secondary structure

![alt text](https://raw.githubusercontent.com/peace195/protein-prediction/master/multitask-learning/multitask-8states/cm1.png)

* Solvent accessibility

![alt text](https://raw.githubusercontent.com/peace195/protein-prediction/master/multitask-learning/multitask-8states/cm2.png)

* b-values

![alt text](https://raw.githubusercontent.com/peace195/protein-prediction/master/multitask-learning/multitask-8states/cm3.png)


## Prerequisites

* python 2.7
* tensorflow 1.4.0
* ProtVec

## How to run

Go into each subfolder and run the code following:

* python lstm.py

## Author

**Binh Do**

## License

This project is licensed under the MIT License

