import io
import numpy
import matplotlib.pyplot as plt
import math
from collections import Counter
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from glob import glob
from classes import ImageData
from classes import ImageDistance
from random import randint

class_to_index = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "J": 9,
                  "K": 10, "L": 11, "M": 12, "N": 13, "O": 14, "P": 15, "Q": 16, "R": 17, "S": 18,
                  "T": 19, "U": 20, "V": 21, "W": 22, "X": 23, "Y": 24, "Z": 25, "a": 26, "b": 27,
                  "c": 28, "d": 29, "e": 30, "f": 31, "g": 32, "h": 33, "i": 34, "j": 35, "k": 36,
                  "l": 37, "m": 38, "n": 39, "o": 40, "p": 41, "q": 42, "r": 43, "s": 44, "t": 45,
                  "u": 46, "v": 47, "w": 48, "x": 49, "y": 50, "z": 51}

'''
Processa todas as imagens em uma determinada pasta e retorna uma lista de objetos ImageData, onde cada objeto contem o 
array de caracteristicas e a classificação de uma imagem;
Além de retornar uma lista, a função também escreve os arrays e a classificação em um arquivo txt.
'''


def folder_processing(classification, folder_path, file):
    data_list = []
    for filepath in glob(folder_path + '\\**'):
        image = imread(filepath)
        resized_image = resize(image, (80, 80), anti_aliasing=True)
        histograms_data = hog(resized_image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(10, 10),
                              feature_vector=True)
        data = ImageData(classification, histograms_data)
        file.write(data.__repr__() + "\n")
        data_list.append(data)

    return data_list


'''
Retorna uma lista com todas as imagens da pasta "characters" processadas e transcrevidas em objetos ImageData;
Também retorna uma lista de listas. Cada uma dessas listas possui todos os objetos ImageData com uma classificação.
'''


def process_characters():
    count_to_class = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
                      10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "H", 18: "I",
                      19: "J", 20: "K", 21: "L", 22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R",
                      28: "S", 29: "T", 30: "U", 31: "V", 32: "W", 33: "X", 34: "Y", 35: "Z", 36: "a",
                      37: "b", 38: "c", 39: "d", 40: "e", 41: "f", 42: "g", 43: "h", 44: "i", 45: "j",
                      46: "k", 47: "l", 48: "m", 49: "n", 50: "o", 51: "p", 52: "q", 53: "r", 54: "s",
                      55: "t", 56: "u", 57: "v", 58: "w", 59: "x", 60: "y", 61: "z"}

    file = io.open("data.txt", "w")
    image_data_list = []
    separated_classes = []
    count = 10
    for path in glob('characters\\**'):
        c = folder_processing(count_to_class[count], path, file)
        separated_classes.append(c)
        image_data_list = image_data_list + c
        count += 1
    file.close()

    return image_data_list, separated_classes


'''
Lê os dados de um arquivo txt;
Retorna uma lista com todas as imagens da pasta "characters" processadas e transcrevidas em objetos ImageData;
Também retorna uma lista de listas. Cada uma dessas listas possui todos os objetos ImageData com uma classificação.
'''


def get_data_from_file(file):
    image_data_list = []
    separated_characters = []
    character_data = []
    count = 0
    for line in file:
        if (line != ""):
            string_list = line.split()
            classification = string_list.pop()

            np = numpy.array(string_list)
            npfloat = np.astype(numpy.float)

            image_data_obj = ImageData(classification, npfloat)

            character_data.append(image_data_obj)

            image_data_list.append(image_data_obj)
        else:
            print("Found one blank line")

        count += 1
        if (count > 54):
            count = 0
            separated_characters.append(character_data)
            character_data = []

    return image_data_list, separated_characters


'''
Calcula a distância pondereada para o algoritmo KNN utilizando w = inverso da distância euclidiana elevado a potência de 2
'''


def p_euclidian(input_characteristics_array, database_characteristics_array):
    s = 0.0  # Variável que vai armazenar a somatória
    for i in range(0, len(input_characteristics_array)):
        s += ((input_characteristics_array[i] - database_characteristics_array[i]) ** 2)
    euclidian = math.sqrt(s)
    w = 1/euclidian**2

    for i in range(0, len(input_characteristics_array)):
        s += w * (input_characteristics_array[i] - database_characteristics_array[i]) ** 2
    pond = math.sqrt(s)

    return pond


'''
Calcula a distância euclidiana
'''


def euclidian_distance(input_characteristics_array, database_characteristics_array):
    s = 0.0  # Variável que vai armazenar a somatória
    for i in range(0, len(input_characteristics_array)):
        s += ((input_characteristics_array[i] - database_characteristics_array[i]) ** 2)
    euclidian = math.sqrt(s)
    return euclidian


'''
Calcula a distância manhattan
'''


def manhattan_distance(input_characteristics_array, database_characteristics_array):
    m = 0.0  # Variável que vai armazenar a somatória
    for i in range(0, len(input_characteristics_array)):
        m += abs(input_characteristics_array[i] - database_characteristics_array[i])
    return m


'''
Utiliza o algoritmo KNN pra classificar as imagens e retorna uma matriz de confusão
'''


def knn(training, test, k):
    matrix = [0] * 52
    for a in range(0, 52):
        matrix[a] = [0] * 52

    for object in test:
        k_nearest = []
        for i in range(0, len(training)):
            # dis = euclidian_distance(object.characteristics_array, training[i].characteristics_array)
            dis = manhattan_distance(object.characteristics_array, training[i].characteristics_array)
            # dis = p_euclidian(object.characteristics_array, training[i].characteristics_array)

            if (dis == 100000):
                k_nearest = k_nearest.append(ImageDistance(training[i].classification, dis))
                break

            k_nearest.append(ImageDistance(training[i].classification, dis))
            if (len(k_nearest) > k):
                k_nearest.sort(key=lambda x: x.distance)
                k_nearest.pop()

        cl = []  # Lista de characters que vai armazenar as k classes dos k elementos mais próximos
        for i in range(0, k):
            cl.append(k_nearest[i].classification)

        # Retirar elementos da lista até que não exista mais de uma classe com a maior quantidade de repetições
        while (True):
            counter = Counter(cl)
            mc = counter.most_common()
            if (len(mc) > 1 and mc[0][1] == mc[1][1]):
                cl.pop()
            else:
                break

        print("Class = " + object.classification + "\nKNN = " + mc[0][0])
        matrix[class_to_index[object.classification]][class_to_index[mc[0][0]]] += 1

    return matrix


'''
Separa as imagens de treino e as de teste
'''


def separate_training_from_test(separated_chars, training_size):
    training = []
    test = []
    for list in separated_chars:
        for i in range(0, training_size):
            training.append(list.pop(randint(0, len(list) - 1)))
        test = test + list

    return training, test


'''
Printa a acurácia e o erro de uma matriz de confusão
'''


def evaluate(matrix):
    mistakes = 0
    successes = 0
    for i in range(0, len(matrix)):
        for j in range(0, len(matrix[i])):
            if (i == j):
                successes += matrix[i][j]
            else:
                mistakes += matrix[i][j]

    total = successes + mistakes
    accuracy = successes/total
    error = mistakes/total

    print("Acurácia = " + str(accuracy))
    print("Erro = " + str(error))


def get_training_and_test_from_file(f_test, f_train):
    test = []
    training = []
    for line in f_test:
        if (line != ""):
            string_list = line.split()
            classification = string_list.pop()

            np = numpy.array(string_list)
            npfloat = np.astype(numpy.float)

            image_data_obj = ImageData(classification, npfloat)

            test.append(image_data_obj)
        else:
            print("Found one blank line")

    for line in f_train:
        if (line != ""):
            string_list = line.split()
            classification = string_list.pop()

            np = numpy.array(string_list)
            npfloat = np.astype(numpy.float)

            image_data_obj = ImageData(classification, npfloat)

            training.append(image_data_obj)
        else:
            print("Found one blank line")

    return training, test


def plot_m(matrix):
    label = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
             "V", "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
             "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

    plt.figure(figsize=(10, 10))
    plt.matshow(matrix, fignum=1)

    x_pos = numpy.arange(len(label))
    plt.xticks(x_pos, label)

    y_pos = numpy.arange(len(label))
    plt.yticks(y_pos, label)

    plt.colorbar()
    plt.show()


def main():
    try:
        f = io.open("data.txt", "r")
    except FileNotFoundError:
        print("Processando dados")
        image_data_list, separated_classes = process_characters()
    else:
        print("Dados já processados")
        image_data_list, separated_classes = get_data_from_file(f)
        f.close()

    try:
        f = io.open("test.txt", "r")
        f2 = io.open("training.txt", "r")
    except FileNotFoundError:
        print("Separando dados")

        training, test = separate_training_from_test(separated_classes, 37)

        f = io.open("test.txt", "w")
        f2 = io.open("training.txt", "w")

        for i in range(0, len(test)):
            f.write(test[i].__repr__() + "\n")
        f.close()

        for i in range(0, len(training)):
            f2.write(training[i].__repr__() + "\n")
        f2.close()
    else:
        print("Dados separados")
        training, test = get_training_and_test_from_file(f, f2)
        f.close()
        f2.close()

    # print(len(training))
    # print(len(test))
    # print(len(image_data_list))

    k = 55
    matrix = knn(training, test, k)

    print("K = " + str(k))
    evaluate(matrix)
    plot_m(matrix)


if __name__ == "__main__":
    main()
