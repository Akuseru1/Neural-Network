import numpy as np
import random
import matplotlib.pyplot as plt


class NeuralNetwork():
    def __init__(self, dataset_name='iris2Clas.csv', n_hidden_neurons=1, eta=0.35, itera=15000, percentage_train_dataset=0.8, id_activate_function=1):
        self.dataset = np.genfromtxt(dataset_name, delimiter=',')
        self.dataset = self.dataset[1:]
        self.len_dataset = len(self.dataset)
        self.percentage_train_dataset = percentage_train_dataset
        self.split_dataset()
        self.n_hidden_neurons = n_hidden_neurons
        self.eta = eta
        self.itera = itera
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
        error_promedio = []
        error_promedio_test = []
        epoca = []
        for itera in range(self.itera):
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

            # datos de TEST
            neta_test = np.dot(self.test_dataset, self.syn0)
            activate_neta_test = self.activate_function['fn'](neta_test)
            neta2_test = np.dot(activate_neta_test, self.syn1)
            output_test = self.activate_function['fn'](neta2_test)
            error_test = self.y_test_dataset - output_test
            error_promedio_test.append(np.mean(np.abs(error_test)))
            # update weights
            self.syn1 += l1.T.dot(l2_delta)
            self.syn0 += input_data.T.dot(l1_delta)

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
    db = input('ingrese el nombre de la base de datos (default: iris2Clas.csv)')
    eta = input('Ingrese el valor del del eta (default: 0.35)')
    iteraciones = input('Ingrese el número de iteraciones (default: 15000)')
    n_neurons = input(
        'Ingrese el número de neuronas en la capa oculta (default: 3)')

    if not db.strip():
        db = 'iris2Clas.csv'

    if not eta.strip():
        eta = 0.35
    else:
        eta = float(eta)

    if not n_neurons.strip():
        n_neurons = 3
    else:
        n_neurons = int(n_neurons)

    if not iteraciones.strip():
        iteraciones = 1500
    else:
        iteraciones = int(iteraciones)

    return db, eta, iteraciones, n_neurons


db, eta, itera, n_neurons = read_parameters()
s = NeuralNetwork(dataset_name=db, eta=eta, itera=itera,
                  n_hidden_neurons=n_neurons)
s.train()
