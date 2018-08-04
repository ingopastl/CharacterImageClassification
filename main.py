import io
import numpy
import math
from collections import Counter
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from glob import glob
from classes import ImageData
from classes import ImageDistance

'''
Processa todas as imagens em uma determinada pasta e retorna uma lista de objetos ImageData, onde cada objeto contem o 
array de caracteristicas e a classificação de uma imagem.
Além de retornar uma lista, a função também escreve os arrays e a classificação em um arquivo txt
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
Retorna uma lista com todas as imagens da pasta "characters" processadas e transcrevidas em objetos ImageData.
'''


def create_database_elements():
    mapping = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
               10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "H", 18: "I",
               19: "J", 20: "K", 21: "L", 22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R",
               28: "S", 29: "T", 30: "U", 31: "V", 32: "W", 33: "X", 34: "Y", 35: "Z", 36: "a",
               37: "b", 38: "c", 39: "d", 40: "e", 41: "f", 42: "g", 43: "h", 44: "i", 45: "j",
               46: "k", 47: "l", 48: "m", 49: "n", 50: "o", 51: "p", 52: "q", 53: "r", 54: "s",
               55: "t", 56: "u", 57: "v", 58: "w", 59: "x", 60: "y", 61: "z"}

    file = io.open("data.txt", "w")
    image_data_list = []
    count = 0
    for path in glob('characters\\**'):
        image_data_list = image_data_list + folder_processing(mapping[count], path, file)
        count += 1
    file.close()

    return image_data_list


'''
Cria um numpy array de objetos ImageData baseando-se nos dados de um arquivo txt
'''


def get_elements_from_file(file):
    image_data_list = []
    for line in file:
        if (line != ""):
            string_list = line.split()
            classification = string_list.pop()

            np = numpy.array(string_list)
            npfloat = np.astype(numpy.float)

            image_data_list.append(ImageData(classification, npfloat))
        else:
            print("Found one")
    return image_data_list


'''
Calcula a distância pondereada para o algoritmo KNN utilizando o inverso da distância euclidiana elevado a potência de 2
'''


def distance(input_characteristics_array, database_characteristics_array):
    s = 0.0  # Variável que vai armazenar a somatória
    for i in range(0, len(input_characteristics_array)):
        s += ((input_characteristics_array[i] - database_characteristics_array[i]) ** 2)
    euclidian = math.sqrt(s)
    if (euclidian == 0):
        return 1
    else:
        return (1 / euclidian) ** 2


'''
Utiliza o algoritmo KNN pra classificar as imagens na pasta "input"
'''


def knn(image_data_list, k):
    string = ""
    for filepath in glob('input\\**'):
        k_nearest = []
        image = imread(filepath)
        resized_image = resize(image, (80, 80), anti_aliasing=True)
        input_data = hog(resized_image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(10, 10),
                         feature_vector=True)

        for i in range(0, len(image_data_list)):
            dis = distance(input_data, image_data_list[i].characteristics_array)
            k_nearest.append(ImageDistance(image_data_list[i].classification, dis))
            k_nearest.sort(key=lambda x: x.distance, reverse=True)  # Inverso porque a distância é ponderada
            if (len(k_nearest) > k):
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

        print(filepath)
        print("Character =", mc[0][0], "\n")
        string += filepath + "\n" + "Character = " + mc[0][0] + "\n\n"
    return string


def main():
    try:
        f = io.open("data.txt", "r")
    except FileNotFoundError:
        print("Processando dados")
        image_data_list = create_database_elements()
    else:
        print("Dados já processados")
        image_data_list = get_elements_from_file(f)
        f.close()

    f = io.open("output.txt", "w")
    f.write(knn(image_data_list, 5))


if __name__ == "__main__":
    main()
