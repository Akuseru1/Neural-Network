import numpy as np
import random


class NeuralNetwork():
    def __init__(self, dataset_name='iris2Clas.csv', n_hidden_neurons=1, eta=0.35, percentage_train_dataset=0.8, id_activate_function=1):
        self.dataset = np.genfromtxt(dataset_name, delimiter=',')
        self.dataset = self.dataset[1:]
        self.len_dataset = len(self.dataset)
        self.percentage_train_dataset = percentage_train_dataset
        self.split_dataset()
        self.n_hidden_neurons = n_hidden_neurons
        self.eta = eta
        self.layers = []
        self.activation_functions = [
            {
                'fn': lambda x: x,
                'dfn': lambda _: 1
            },

            {
                'fn': lambda x: 1/(1 + np.exp(-x)),
                'dfn': lambda x: x*(1 - x)
            },

            {
                'fn': lambda x: 2/(1 + np.exp(-x)) - 1,
                'dfn': lambda x: 2*np.exp(-x)/((1 + np.exp(-x)) ^ 2)
            },

            {
                'fn': lambda x: 1 if x >= 0 else 0,
                'dfn': lambda _: 0
            }
        ]
        self.activate_function = self.activation_functions[id_activate_function]
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
        self.syn0 = np.random.rand(4, self.n_hidden_neurons)
        # 1 because there 1 output
        self.syn1 = np.random.rand(self.n_hidden_neurons, 1)

    def train(self):
        for iter in range(15000):
            input_data = self.train_dataset
            neta1 = np.dot(input_data, self.syn0)
            print(f'Netas Intermedias: \n, {neta1 }')
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
            if (iter % 10000) == 0:
                print(
                    f'Error Promedio Absoluto: {str(np.mean(np.abs(l2_error)))}')

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
            # update weights
            self.syn1 += l1.T.dot(l2_delta)
            self.syn0 += input_data.T.dot(l1_delta)

        print('\n Output After Training:')
        print(f'Salida red: \n, {l2}')
        print(
            f'Promedio de error con el dataset de entrenamiento: {str(np.mean(np.abs(l2_error)))} \n')
        print(f'pesos salida  \n {self.syn1} \n')
        print(f'pesos entrada \n  {self.syn0} \n')
        neta_test = np.dot(self.test_dataset, self.syn0)
        activate_neta_test = self.activate_function['fn'](neta_test)
        neta2_test = np.dot(activate_neta_test, self.syn1)
        output_test = self.activate_function['fn'](neta2_test)
        error_test = self.y_test_dataset - output_test
        print(f'Salida para test dataset: {output_test}')
        print(f'Y real para test dataset: {self.y_test_dataset}')
        print(
            f'Promedio de error con el dataset de prueba {str(np.mean(np.abs(error_test)))}')


s = NeuralNetwork()
s.train()

