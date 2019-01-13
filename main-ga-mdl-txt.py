import os
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import LSTM
from keras.datasets import imdb
from keras.backend import tensorflow_backend as K
import tensorflow as tf
import traceback
import random

MODELS = {
        1 : 'DNN',
        2 : 'CNN',
        3 : 'LSTM'
    }

OPTIMIZERS = {
        1 : 'Adam',
        2 : 'RMSprop',
        3 : 'Adagrad',
        4 : 'Adadelta',
        5 : 'SGD',
        6 : 'Adamax',
        7 : 'Nadam',
    }

class Individual:
    obj = {}

    def __init__(self, chromosome):
        self.chromosome = tuple(chromosome)
        if self.chromosome in Individual.obj:
            self.fitness = Individual.obj[self.chromosome].fitness
            self.calculated = Individual.obj[self.chromosome].calculated
        else:
            self.fitness = float('-inf')
            self.calculated = False
        Individual.obj[self.chromosome] = self

    def __eq__(self, other):
        if isinstance(other, Individual):
            return self.chromosome == other.chromosome
        return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def cal_fitness(self):
        if not self.calculated:
            try:
                self.fitness = build_random_model(self.chromosome)
                self.calculated = True
            except Exception as exc:
                print(traceback.format_exc())
                print(exc)

def create_genes(position,size):
    if type(MIN[position]) == int:
        population = range(MIN[position],MAX[position]+1)
        if size <= len(population):
            samples = random.sample(population,size)
        else:
            samples = random.choices(population,k=size)
    else:
        samples = [random.uniform(MIN[position], MAX[position]) for _ in range(size)]
    return samples

def create_population(size):
    population = [[] for _ in range(size)]
    for i in range(N_PARAMS):
        list(map(lambda x, y: x.append(y), population, create_genes(i, size)))
    return list(map(lambda x: Individual(x), population))

def mate(par1, par2):
    child_chromosome = []
    for gp1, gp2, i in zip(par1.chromosome, par2.chromosome, range(N_PARAMS)):
        prob = random.random()
        if prob < 0.45:
            child_chromosome.append(gp1)
        elif prob < 0.90:
            child_chromosome.append(gp2)
        else:
            child_chromosome.append(create_genes(i,1)[0])
    return child_chromosome

print("HELLO AI WORLD!!")
def getModel(x):
    return MODELS.get(x, 'none')

def getOptimizer(x):
    return OPTIMIZERS.get(x, 'none')

def build_DNN_model(neurons, embedding_dims, max_features, maxlen, optimizer, dropout):
    model = Sequential()
    model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
    model.add(Flatten())

    if 0 < dropout < 1:
        model.add(Dropout(dropout))

    for x in neurons:
        model.add(Dense(x, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])
    return model

def build_CNN_model(filters, embedding_dims, max_features, maxlen, optimizer, dropout):
    kernel_size = 3

    model = Sequential()
    model.add(Embedding(max_features, embedding_dims, input_length=maxlen))

    if 0 < dropout < 1:
        model.add(Dropout(dropout))

    for x in filters:
        model.add(Conv1D(x, kernel_size, padding='valid', activation='relu', strides=1))
        model.add(MaxPooling1D(kernel_size))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def build_LSTM_model(neurons, embedding_dims, max_features, maxlen, optimizer, dropout):
    model = Sequential()
    model.add(Embedding(max_features, embedding_dims, input_length=maxlen))

    if 0 < dropout < 1:
        model.add(Dropout(dropout))

    for x in neurons:
        model.add(LSTM(x,return_sequences=True))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    return model

def build_random_model(*args):
    if len(args) == 1:
        args = args[0]
    input_model = args[0]
    optimizer = args[1]
    neurons = args[3:6]
    neurons = list(filter(lambda x: x > 0, neurons))
    neurons = neurons[:args[2]]
    max_features = args[6]
    embedding_dims = args[7]
    maxlen = args[8]
    dropout = args[9]
    batch_size = args[10]
    epoch = args[11]

    print('input model = ', getModel(input_model))
    print('optimizer = ', getOptimizer(optimizer))
    print('neurons = ', neurons)
    print('number of words = ', max_features)
    print('embedding dimension = ', embedding_dims)
    print('max length of pad sequence = ', maxlen)
    print('dropout = ', dropout)
    print('batch size = ', batch_size)
    print('epoch = ', epoch)

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    input_model = getModel(input_model)
    optimizer = getOptimizer(optimizer)

    if input_model == 'DNN':
        model = build_DNN_model(neurons, embedding_dims, max_features, maxlen, optimizer, dropout)
    elif input_model == 'CNN':
        model = build_CNN_model(neurons, embedding_dims, max_features, maxlen, optimizer, dropout)
    elif input_model == 'LSTM':
        model = build_LSTM_model(neurons, embedding_dims, max_features, maxlen, optimizer, dropout)

    model.fit(x_train, y_train,batch_size=batch_size, epochs=epoch, validation_data=(x_test, y_test), verbose=2)
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

    print('Test score:', score)
    print('Test accuracy:', acc)

    return acc

def cal_fitness(population):
    for individual in population:
        individual.cal_fitness()

POOL_SIZE=20
#model,optimizer,layer,n1,n2,n3,max_features,embedding_dim,max_len,dropout,batch,epoch
MIN = (1,1,1,8,0,0,10000,8,10,0.0,8,1)
MAX = (len(MODELS),len(OPTIMIZERS),3,356,256,256,20000,256,100,1.0,64,10)
N_PARAMS = len(MIN)

if __name__ == '__main__':
    generation = 0
    population = create_population(POOL_SIZE)
    cal_fitness(population)
    population = sorted(population, key=lambda x: x.fitness, reverse=True)
    if not os.environ.get('OMP_NUM_THREADS'):
        config = tf.ConfigProto(allow_soft_placement=True)
    else:
        num_thread = int(os.environ.get('OMP_NUM_THREADS'))
        config = tf.ConfigProto(intra_op_parallelism_threads=num_thread,
                                allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))
    for i in range(20):
        new_generation = []
        s = int((30 * POOL_SIZE) / 100)
        new_generation.extend(population[:s])
        s = int((70 * POOL_SIZE) / 100)
        for _ in range(s):
            parent1 = random.choice(
                population[:int(POOL_SIZE / 2)])
            parent2 = random.choice(
                population[:int(POOL_SIZE / 2)])
            child = Individual(mate(parent1, parent2))
            new_generation.append(child)
        population = new_generation
        cal_fitness(population)
        population = sorted(population, key=lambda x: x.fitness, reverse=True)
        generation += 1
    score = population[0].fitness
#contoh build 1 model
#build_random_model(1, 1, 128, 0, 0, 32767, 32, 100, 0.2, 32, 5)