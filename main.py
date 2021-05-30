import os
import tensorflow as tf
import tensorflow_addons as tfa
tfa.register.register_all()
import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rng
import seaborn as sns
import pickle


class PokeMetric:
    def __init__(self, width, height, n_channel, support_dataset, support_size=151):
        self.width = width
        self.height = height
        self.n_channel = n_channel
        self.support = [] #для хранения обучающих пар изображений
        self.model = None
        self.test_images = [] #для хранения тестовых пар изображений
        self.test_labels = []
        self.current_epoch = 0 #для отслеживания текущей эпохи обучения
        self.support_size = support_size
        self.load_support_images(support_dataset)

    #загрузка обучающих изображений
    def load_support_images(self, dataset_location):
        self.support = []
        labels = []
        for data in os.listdir(dataset_location):
            name = data.split('.')[0]
            loading_mode = 0 if self.n_channel == 1 else 1
            if name.isdigit() and int(name) <= self.support_size:
                image = cv2.imread(dataset_location + data, loading_mode)
                image = cv2.resize(image, (self.width, self.height))
                self.support.append(image / 255.)
                labels.append(int(name))
        self.support = [self.support[i] for i in np.argsort(labels)]

    #загрузка тестовых изображений
    def load_val_images(self, dataset_location):
        self.test_images = []
        self.test_labels = []
        loading_mode = 0 if self.n_channel == 1 else 1
        for data in os.listdir(dataset_location):
            name = data.split('-')[0]
            if int(name) <= self.support_size:
                image = cv2.imread(dataset_location + data, loading_mode)
                image = cv2.resize(image, (self.width, self.height))
                self.test_images.append(image / 255.)
                self.test_labels.append(int(name) - 1)
        self.test_labels = np.array(self.test_labels)

    #строим модель сети
    def build_model(self, drop_rate=0, kernel_reg=0, std_init=0.01):
        input_shape = (self.width, self.height, self.n_channel)
        first_input = tf.keras.layers.Input(input_shape)
        second_input = tf.keras.layers.Input(input_shape)

        w_init = tf.keras.initializers.RandomNormal(0, std_init)
        b_init = tf.keras.initializers.RandomNormal(0.5, std_init)
        regul = tf.keras.regularizers.l2(kernel_reg)

        convnet = tf.keras.models.Sequential()
        #слой свертки из 64 фильтров размером 10х10 с последующим применением функции ReLU
        convnet.add(tf.keras.layers.Conv2D(64, (10, 10), activation='relu', input_shape=input_shape,
                                           kernel_initializer=w_init, kernel_regularizer=regul))
        #слой субдискретизации
        convnet.add(tf.keras.layers.MaxPooling2D())
        #слой свертки из 128 фильтров размером 7х7 с применением ReLU
        convnet.add(tf.keras.layers.Conv2D(128, (7, 7), activation='relu',
                                           kernel_initializer=w_init, kernel_regularizer=regul,
                                           bias_initializer=b_init))
        #слой субдискретизации
        convnet.add(tf.keras.layers.MaxPooling2D())
        #слой свертки из 128 фильтров размером 4х4 с применением ReLU
        convnet.add(tf.keras.layers.Conv2D(128, (4, 4), activation='relu',
                                           kernel_initializer=w_init, kernel_regularizer=regul,
                                           bias_initializer=b_init))
        #слой субдискретизации
        convnet.add(tf.keras.layers.MaxPooling2D())
        #слой свертки из 256 фильтров размером 4х4 с применением ReLU
        convnet.add(tf.keras.layers.Conv2D(256, (4, 4), activation='relu',
                                           kernel_initializer=w_init, kernel_regularizer=regul,
                                           bias_initializer=b_init))
        #исключающий слой
        convnet.add(tf.keras.layers.Dropout(drop_rate))
        #сглаживающий слой
        convnet.add(tf.keras.layers.Flatten())
        #полносвязный слой с 4096 нейронами и сигмоидальная функция активации
        convnet.add(tf.keras.layers.Dense(4096, activation="sigmoid",
                                          kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.1),
                                          kernel_regularizer=regul,
                                          bias_initializer=tf.keras.initializers.RandomNormal(0, 0.1)))
        #исключающий слой
        convnet.add(tf.keras.layers.Dropout(drop_rate))

        encoded_f = convnet(first_input)
        encoded_s = convnet(second_input)
        merged = tf.math.abs(encoded_f - encoded_s)
        prediction = tf.keras.layers.Dense(1, activation='sigmoid',
                                           kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.1),
                                           kernel_regularizer=regul,
                                           bias_initializer=tf.keras.initializers.RandomNormal(0, 0.1))(merged)

        self.model = tf.keras.Model(inputs=[first_input, second_input], outputs=prediction)
        self.current_epoch = 0

    def compile(self, optimizer, loss):
        self.model.compile(optimizer, loss)

    #получение выборки заданного размера
    def get_batch(self, batch_size=32):
        # получаем выборку обучающих пар, одинаковые по типу изображения помечаются 1, различные - 0
        pairs = [[], []]
        train_support = np.array(
            tfa.image.rotate(np.array(self.support).reshape(self.support_size, self.width, self.height, self.n_channel),
                             rng.randn(self.support_size)))
        # добавляем случайные нулевые значения, чтобы избежать необходимости переобучения
        train_support += (0.1 * rng.randn(self.support_size, self.width, self.height, self.n_channel) + 0.5) * (
                    train_support == 0)
        ref = rng.randint(0, self.support_size, batch_size // 2)
        indices = list(np.arange(self.support_size))
        pairs[0] = [train_support[i] for i in ref]
        pairs[1] = [train_support[i] for i in ref]
        for i in range(batch_size // 2):
            pairs[0].append(train_support[ref[i]])
            indices.remove(ref[i])
            pairs[1].append(train_support[rng.choice(indices)])
            indices.append(ref[i])
        pairs[0] = np.array(pairs[0])
        pairs[1] = np.array(pairs[1])
        labels = np.array([1] * (batch_size // 2) + [0] * (batch_size // 2))
        del train_support
        return pairs, labels

    #получение тестовой выборки заданного размера
    def get_val_batch(self, batch_size):
        # получаем выборку тестовых пар, одно изображение принадлежит изображениям обучающей выборки, другое - загружается из папки tcg
        pairs = [[], []]
        ref = rng.randint(0, self.support_size, batch_size // 2)
        pairs[0] = [self.support[i] if i in self.test_labels else self.support[0] for i in ref]
        pairs[1] = [self.test_images[rng.choice(np.where(self.test_labels == i)[0])] if i in self.test_labels
                    else self.test_images[rng.choice(np.where(self.test_labels == 0)[0])] for i in ref]
        for i in range(batch_size // 2):
            pairs[0].append(self.support[ref[i]])
            pairs[1].append(self.test_images[rng.choice(np.where(self.test_labels != i)[0])])
        pairs[0] = np.array(pairs[0])
        pairs[1] = np.array(pairs[1])
        labels = np.array([1] * (batch_size // 2) + [0] * (batch_size // 2))
        return pairs, labels

    #функция обучения (количество обучающих выборок, размер выборки, частота сохранения, шаг)
    def train(self, n_batch, batch_size, saving_period=100, val_step=100):
        val_loss_track = []
        train_loss_track = []
        for i in range(n_batch):
            x, y = self.get_batch(batch_size)
            train_loss = self.model.train_on_batch(x, y)
            del x
            del y
            if i % saving_period == 0 and i != 0:
                self.model.save("poke" + str(self.current_epoch + i)) #сохранение файла обучения
            if i % val_step == 0:
                print(i + self.current_epoch)
                train_loss_track.append(train_loss)
                x_val, y_val = self.get_val_batch(batch_size)
                val_loss = self.model.test_on_batch(x_val, y_val)
                del x_val
                del y_val
                val_loss_track.append(val_loss)

        self.current_epoch += n_batch
        return train_loss_track, val_loss_track

    #функция тестирования (размер тестовой выборки), результат - процент верных решений программы от общего количества
    def test(self, test_size):
        correct = 0
        results = []
        selected_indices = rng.randint(0, len(self.test_images), test_size)
        selected = ([self.test_images[i] for i in selected_indices], [self.test_labels[i] for i in selected_indices])
        for i in range(test_size):
            r = self.prediction(selected[0][i], test=True)
            if r == selected[1][i]:
                correct += 1
            results.append((r, selected[1][i]))
        return (correct / test_size) * 100, results

    #полное тестирование на всей обучающей выборке
    def self_test(self):
        correct = 0
        results = []
        for i in range(self.support_size):
            r = self.prediction(self.support[i], test=True)
            if r == i:
                correct += 1
            results.append((r, i))
        return (correct / self.support_size) * 100, results

    #проверка изображения img
    #в результате исполнения функция выдает предполагаемый тип изображения
    def prediction(self, img, test=False):
        if not test:
            img = cv2.resize(img, (self.width, self.height)) / 255.
        pairs = [np.array([img] * self.support_size), np.array(self.support)]
        probs = self.model.predict(pairs)
        return np.argmax(probs)

    #определяет принадлежность двух изображений одному классу
    def check(self, ind1, ind2, test=False):
	img1 = self.test_images[ind1]
	print(img1)
	img2 = self.test_images[ind2]
	print(img2)
	return self.prediction(self, img1, test) == self.prediction(self, img2, test)

    #восстановление файла обучения
    def restore(self, i, from_input=None):
        if from_input:
            path = "poke" + str(i)
        else:
            path = "poke" + str(i)
        self.model = tf.keras.models.load_model(path)
        self.current_epoch = i


support_dataset = "pokemon/pokemon-a/"
test_dataset = "pokemon/pokemon-tcg-images/"
inputs = []
values = []
width = 128
height = 128
pokemetric = PokeMetric(width, height, 1, support_dataset, 151)
pokemetric.load_val_images(test_dataset)

pokemetric.build_model(drop_rate=0.3, kernel_reg=0.0002)
pokemetric.compile(tf.keras.optimizers.Adam(lr=0.00006), tf.losses.BinaryCrossentropy())
t, v = [], []
# pokemetric.restore(5000)
# t = t[:pokemetric.current_epoch]
# v = t[:pokemetric.current_epoch]
tbis, vbis = pokemetric.train(5001, 32, saving_period=5000)
t += tbis
v += vbis
plt.plot(t)
plt.plot(v)
plt.legend(['обучение', 'тестирование'])
# plt.show()
# if pokemetric.check(125,22,test = True):
#	print('Тип совпадает')
# else:
#	print('Тип не совпадает')
# if pokemetric.check(rng.randint(0, len(self.test_images), test_size), rng.randint(0, len(self.test_images), test_size), test = True):
#	print('Тип совпадает')
# else:
#	print('Тип не совпадает')


def results_to_heatmap(r):
    hm = np.zeros((501,501))
    for i in range(len(r)):
        hm[r[i][0],r[i][1]] += 1
    return hm

def results_to_histogram(r):
    histo_true, histo_pred = [0]*501, [0]*501
    for i in range(len(r)):
        histo_true[r[i][1]] += 1
        histo_pred[r[i][0]] += 1
    return histo_true, histo_pred
r = pokemetric.test(500)
print(r[0])
sns.set_style()
plt.figure(figsize=(15,15))
sns.heatmap(results_to_heatmap(r[1]))
plt.show()
