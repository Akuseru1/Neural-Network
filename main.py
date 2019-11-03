import numpy as np
import random


class NeuralNetwork():
    def __init__(self, nombre_base_datos='iris2Clas.csv', n_hidden_neurons=3, eta=0.01, len_train_dataset=0.8):
        self.dataset = np.genfromtxt(nombre_base_datos, delimiter=',')
        self.len_train_dataset = len_train_dataset
        self.len_dataset = len(self.dataset)
        self.split_dataset()
        self.n_hidden_neurons = n_hidden_neurons
        self.eta = eta
        self.init_weights()

    def split_dataset(self):
        len_train_dataset = int(self.len_dataset * self.len_train_dataset)
        train_dataset = []
        test_dataset = []
        rows_train_dataset = random.sample(
            range(self.len_dataset), len_train_dataset)

        for r in range(self.len_dataset):
            if r in rows_train_dataset:
                train_dataset.append(self.dataset[r])

            else:
                test_dataset.append(self.dataset[r])
        self.train_dataset = np.array(train_dataset)
        self.test_dataset = np.array(test_dataset)

    def init_weights(self):
        # 4 because there 4 inputs
        self.syn0 = np.random.rand(4, self.n_hidden_neurons)
        # 1 because there 1 output
        self.syn1 = np.random.rand(self.n_hidden_neurons, 1)

    def sigmoid(self, x, derivada=False):
        layer = []
        if(derivada == True):
            for i in x:
                layer.append(i*(1-i))
            return layer
        for i in x:
            layer.append(1/(1+np.exp(-i)))
        return layer

    def forward_propagation(self):
        # del input al hidden
        self.l1 = self.sigmoid(
            np.matmul(self.train_dataset[:, :-1], self.syn0))
        self.output = self.sigmoid(np.matmul(self.l1, self.syn1))

    def backwards_propagation(self):
        # error queda con los errores del 80%
        self.error1 = [ i - j for (i, j) in zip(self.train_dataset[:, -1], self.output)]
        self.error0 = [i - j for (i, j) in zip(self.train_dataset[:, -1], self.l1)]
        derivadas1 = self.sigmoid(self.output, True)
        derivadas0 = self.sigmoid(self.l1, True)
        output_delta = [i * j * self.eta for (i, j) in zip(self.error1, derivadas1)]
        l1_delta = [i * j * self.eta for (i, j) in zip(self.error0, derivadas0)]

        self.syn0 += np.matmul(self.train_dataset[:, :-1].T, l1_delta)
        self.syn1 += np.matmul(self.train_dataset[:, :-1].T, output_delta)

    def train(self):
        self.forward_propagation()
        self.backwards_propagation()

    def run(self, iter=1000):
        primera_vez = True
        print("Pesos hacia hidden: ", self.syn0)
        print("Pesos hacia output: ", self.syn0)
        for i in range(iter):
            self.train()
            print("Iteracion: ", iter)
            print("Error0: ", self.error0)
            print("Error1: ", self.error1)
            if(primera_vez):
                error0_inicial = self.error0
                error1_inicial = self.error1
        print("Pesos hacia hidden: ", self.syn0)
        print("Pesos hacia output: ", self.syn0)
        print("Error0 inicial: ", error0_inicial)
        print("Error1 inicial: ", error1_inicial)
        print("Error0 final: ", self.error0)
        print("Error1 final: ", self.error1)


s = NeuralNetwork()
s.run()
"""
# sigmoid function
def nonlin(x, deriv=False):
    if(deriv == True):
        return x*(1-x)
    return 1/(1+np.exp(-x))


# Si se quiere ingresar mas patrones, se ingresan entre llaves separadas
# por comas  EJ: [ [1,0,1],[1,0,1]] tanto en entrdas com en salidas
# input dataset
X = np.array([[1, 0, 1]])

# output dataset
y = np.array([[1]])

# seed random numbers to make calculation
# deterministic (just a good practice)

# cambiar la semilla influye mucho en el error
np.random.seed(1)
# Si cambio el numero de entradas o ocultas debo modidicar el No. pesos
# Ejemplo cambio  a 4 entradas syn0 -> (4x2)...
# Si cambio capas ocultas no. pesos syn1 cambia

# initialize weights randomly with mean 0
# SIN BIAS
syn0 = np.array([[0.2, -0.3],
                 [0.4, 0.1],
                 [-0.5, 0.2]])
#syn0 = 2*np.random.random((3,2)) - 1
syn1 = np.array([[-0.3], [-0.2]])
#syn1 = 2*np.random.random((2,1)) - 1
print('Pesos: ', '\n')
print(syn0, '\n')
print(syn1, '\n')


eta = 0.9

for iter in range(1):
        # forward propagation
        # Feed forward through layers 0, 1 and 2
    l0 = X
    neta1 = np.dot(l0, syn0)
    print('Netas Intermedias:', '\n', neta1)
    l1 = nonlin(np.dot(l0, syn0))
    print('Salidas Intermedias:', '\n', l1)

    neta2 = np.dot(l1, syn1)
    print('Netas Salidas:', '\n', neta2)
    l2 = nonlin(np.dot(l1, syn1))
    print('Salida red:', '\n', l2)

    # how much did we miss?
    # how much did we miss the target value?
    l2_error = y - l2
# ENTRENAMIENTO
    if (iter % 10000) == 0:
        print('Error Promedio Absoluto:' + str(np.mean(np.abs(l2_error))))

    # in what direction is the target value?
    # were we really sure? if so, don't change too much.

    # NO multiply how much we missed by the
    # NO slope of the sigmoid at the values in l1
    l2_delta = l2_error * nonlin(l2, deriv=True)*eta

    print('l2_delta:', '\n', l2_delta, '\n')

    # how much did each l1 value contribute to the l2 error
    # (according to the weights)?

    l1_error = l2_delta.dot(syn1.T)
    print('l1_error:', '\n', l1_error, '\n')

    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1, deriv=True)*eta
    print('l1_delta:', '\n', l1_delta, '\n')
    # update weights
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print('\n', 'Output After Training:')
print('Salida red:', '\n', l2)
print('Error:' + str(np.mean(np.abs(l2_error))))
print('\n', 'pesos salida', '\n', syn1, '\n')
print('\n', 'pesos entrada', '\n', syn0)
# Con formateo
# print "%5d%10s" %(1,'a')
"""
