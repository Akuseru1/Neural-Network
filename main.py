import numpy as np
import math
import random
import matplotlib.pyplot as plt


class NeuralNetwork():
    def __init__(self, dataset_name='iris2Clas.csv', n_hidden_neurons=3, eta=0.35, itera=15000, percentage_train_dataset=0.8, id_activate_function=1):
        self.dataset = np.genfromtxt(dataset_name, delimiter=',')
        self.dataset = self.normalization(self.dataset[1:])
        self.len_dataset = len(self.dataset)
        self.percentage_train_dataset = percentage_train_dataset
        self.split_dataset()
        self.n_hidden_neurons = n_hidden_neurons
        self.eta = eta
        self.itera = itera
        self.activation_functions = [
            {
                'fn': lambda x: x,
                'dfn': lambda x: np.ones((len(x), 1))
            },

            {
                'fn': lambda x: 1/(1 + np.exp(-x)),
                'dfn': lambda x: x*(1 - x)
            },

            {
                'fn': lambda x: 2/(1 + np.exp(-x)) - 1,
                'dfn': lambda x: 2*np.exp(-x)/((1 + np.exp(-x)) ** 2)
            },

            {
                'fn': lambda x: np.array([[(1 if v >= 0 else 0) for v in i] for i in x]),
                'dfn': lambda x: np.zeros((len(x), 1))
            }
        ]
        self.activate_function = self.activation_functions[3]
        self.init_weights()

    def split_dataset(self):
        len_train_dataset = int(
            self.len_dataset * self.percentage_train_dataset)
        train_dataset = []
        test_dataset = []
        y_train_dataset = []
        y_test_dataset = []
        rows_train_dataset = random.sample(
            range(self.len_dataset), len_train_dataset)

        for r in range(self.len_dataset):
            if r in rows_train_dataset:

                train_dataset.append(self.dataset[r][:4])
                y_train_dataset.append([self.dataset[r][-1]])
            else:
                test_dataset.append(self.dataset[r][:4])
                y_test_dataset.append([self.dataset[r][-1]])

        self.train_dataset = np.array(train_dataset)
        self.test_dataset = np.array(test_dataset)
        self.y_train_dataset = np.array(y_train_dataset)
        self.y_test_dataset = np.array(y_test_dataset)

    def init_weights(self):
        np.random.seed(2)
        # 4 because there 4 inputs
        self.syn0 = 2 * np.random.rand(4, self.n_hidden_neurons) - 1
        # 1 because there 1 output
        self.syn1 = 2 * np.random.rand(self.n_hidden_neurons, 1) - 1

    def normalization(self, dataset):
        minimum_values = []
        maximum_values = []
        normalized = []
        for i in range(len(dataset[0])):
            # saca el minimo de todos los datos de cada columna
            minimum_values.append(min(dataset[:, i]))
            maximum_values.append(max(dataset[:, i]))

        n = lambda x: [ (x[y] - minimum_values[y]) / (maximum_values[y] - minimum_values[y]) for y in range(len(x))] # funcion min-max
        
        for i in range(len(dataset)):
            normalized.append(n(dataset[i]))
        normalized = np.array(normalized)
        return normalized

    def train(self):
        error_promedio = []
        error_promedio_test = []
        epoca = []
        input_data = self.train_dataset
        for itera in range(self.itera):
            neta1 = np.dot(input_data, self.syn0)
            # print(f'Netas Intermedias: \n, {neta1 }')
            l1 = self.activate_function['fn'](neta1)
            print(f'Salidas Intermedias: \n { l1 }')

            neta2 = np.dot(l1, self.syn1)

            print(f'Netas Salidas: \n { neta2 }')
            l2 = self.activate_function['fn'](neta2)
            print(f'Salida red: \n { l2 }')

            # how much did we miss?
            # how much did we miss the target value?
            l2_error = self.y_train_dataset - l2
            # ENTRENAMIENTO
            error_promedio.append(np.mean(np.abs(l2_error)))
            epoca.append(itera)
            if (itera % 10000) == 0:
                print(
                    f'Error Promedio Absoluto: {str(error_promedio[-1])}')

            # in what direction is the target value?
            # were we really sure? if so, don't change too much.

            # NO multiply how much we missed by the
            # NO slope of the sigmoid at the values in l1
            l2_delta = l2_error * self.activate_function['dfn'](l2)*self.eta
            print(f'l2_delta: \n {l2_delta} \n')

            # how much did each l1 value contribute to the l2 error
            # (according to the weights)?

            l1_error = l2_delta.dot(self.syn1.T)
            print(f'l1_error: \n {l1_error} \n')

            # in what direction is the target l1?
            # were we really sure? if so, don't change too much.
            l1_delta = l1_error * self.activate_function['dfn'](l1)*self.eta

            print(f'l1_delta: \n {l1_delta} \n')
            print(f'l1_error: \n {l1_error} \n')

            # datos de TEST
            neta_test = np.dot(self.test_dataset, self.syn0)
            activate_neta_test = self.activate_function['fn'](neta_test)
            neta2_test = np.dot(activate_neta_test, self.syn1)
            output_test = self.activate_function['fn'](neta2_test)
            error_test = self.y_test_dataset - output_test
            error_promedio_test.append(np.mean(np.abs(error_test)))
            # update weights
            print(f' weights 0 before: \n { self.syn0 }')
            print(f' weight 1 before : \n { self.syn1 }')
            self.syn1 += l1.T.dot(l2_delta)
            self.syn0 += input_data.T.dot(l1_delta)
            print(f' weights 0 after: \n { self.syn0 }')
            print(f' weight 1 after : \n { self.syn1 }')

        print('\n Output After Training:')
        print(f'Salida red: \n, {l2}')
        print(
            f'Promedio de error con el dataset de entrenamiento: {str(np.mean(np.abs(l2_error)))} \n')
        print(f'pesos salida  \n {self.syn1} \n')
        print(f'pesos entrada \n  {self.syn0} \n')
        print(f'Salida para test dataset: {output_test}')
        print(f'Y real para test dataset: {self.y_test_dataset}')
        print(
            f'Promedio de error con el dataset de prueba {str(error_promedio_test[-1])}')

        plt.xlabel('Iteraciones')
        plt.ylabel('Error promedio')
        plt.plot(epoca, error_promedio, label='Trained')
        plt.plot(epoca, error_promedio_test, label='Test')
        plt.legend(loc='upper right')
        plt.show()


def read_parameters():
    dt = input('Ingrese el nombre del dataset (default: iris2Clas.csv)')
    function_id = input('Ingrese la función que desea utilizar \n 0:Identidad \n 1:Sigmoide \n 2: \n 3: (default Sigmoide)')
    eta = input('Ingrese el valor del del eta (default: 0.001 para función identidad, 0.35 en otro caso)')
    iteraciones = input('Ingrese el número de épocas (default: 15000)')
    n_neurons = input('Ingrese el número de neuronas en la capa oculta (default: 3)')
    

    if not dt.strip():
        dt = 'iris2Clas.csv'

    if not function_id.strip():
        function_id = 1
    else:
        function_id = int(function_id)

    if not eta.strip():
        eta = 0.35 if function_id else 0.001
    else:
        eta = float(eta)
        if not function_id and eta > 0.001:
            print("ETA demasiado alto para usar la función identidad como función de activación, esto llevará a nan")

    if not n_neurons.strip():
        n_neurons = 3
    else:
        n_neurons = int(n_neurons)

    if not iteraciones.strip():
        iteraciones = 15000
    else:
        iteraciones = int(iteraciones)

    return dt, eta, iteraciones, n_neurons, function_id


dt, eta, itera, n_neurons, funcion = read_parameters()
s = NeuralNetwork(dataset_name=dt, eta=eta, itera=itera,
                  n_hidden_neurons=n_neurons, id_activate_function=funcion)
s.train()
