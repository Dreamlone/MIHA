# ![miha_logo.png](https://raw.githubusercontent.com/Dreamlone/MIHA/main/images/logo.png)

# MIHA
Optimizer for configuration of hyperparameters in neural networks. 

> What does this library do? - Module can optimize hyperparameters of a neural network for a pre-defined architecture.

> What deep learning libraries can this module work with? - PyTorch.

> What algorithm is used for optimization? - An evolutionary algorithm with mutation and crossover operators is used. The neural network is continuously trained in the process of evolution.
>
## The main concept

![main_concept.png](https://raw.githubusercontent.com/Dreamlone/MIHA/main/images/main_concept.png)

## Requirements
    'python>=3.7',
    'numpy',
    'cudatoolkit==10.2',
    'torchvision==0.7.0',
    'pytorch==1.6.0'
    
## Documentation

Description of the submodules:
* [model](https://github.com/Dreamlone/MIHA/blob/main/docs/model.md)
* [log](https://github.com/Dreamlone/MIHA/blob/main/docs/log.md)
* [evolutionary](https://github.com/Dreamlone/MIHA/blob/main/docs/evolutionary.md)

## How to use

How to run the algorithm can be seen in the examples:
* [Remote sensing gap-filling example (CNN)](https://github.com/Dreamlone/MIHA/blob/main/examples/CNN_autoencoder_gapfilling.py)
* Prediction of water levels (FNN) (in progress)
* Time series forecasting (LSTM) (in progress)

## Contacts

Feel free to contact us:

* [Mikhail Kovalchuk](https://github.com/angrymuskrat) | mlhakov2011@gmail.com 

* Mikhail Sarafanov | mik_sar@mail.ru

