# ga-multimodel-deep-learning
Random Multimodel Deep Learning using GA for Text Classification

# Introduction
Building the perfect deep learning network involves a hefty amount of art to accompany sound science. One way to go about finding the right hyperparameters is through trial and error, but there’s a better way!
Hyperparameters are the variables which are set before training and determine the network structure & how the network is trained. (eg : learning rate, batch size, number of epochs). Fine tuning of hyperparameters can be done by : Search Algorithm (Manual Search, Grid Search, Random Search, etc)
Here, we try to some improvement by applying a Genetic Algorithm to evolve a network with the goal of achieving optimal hyperparameters.

# IMDB Movie reviews sentiment classification
Dataset of 25,000 movies reviews from IMDB, labeled by sentiment (positive/negative). Reviews have been preprocessed, and each review is encoded as a sequence of word indexes (integers).

# Implementation
Define range for every feature in Chromosome
1. model : DNN (1), CNN (2), LSTM (3)
2. number of layer : 1-3
3. number of neuron each layer : 8 - 512
4. embedding dimension : 8 - 256
5. epoch :  1-10
6. dropout :  0.0 – 1.0
7. batch_size : 8 - 64
8. max_length of pad sequence : 10 - 100
9. number of word for imdb : 10000 – 20000
10. Optimizer :  Adam (1), RMSprop (2), Adagrad (3),  Adadelta (4), SGD (5), Adamax (6), Nadam (7)

# Conclusion
To achieve the best performances of deep learning model , we may:
1. Fine Tune Hyper-Parameters 
2. Improve Text Pre-Processing 
3. Use Dropout Layer
