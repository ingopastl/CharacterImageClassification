import io
import numpy
import math
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from glob import glob
from Image import ImageData
from Image import ImageDistance

# Processa todas as imagens em uma determinada pasta e retorna uma lista de objetos ImageData, onde cada objeto contem a lista de caracteristicas e a classificação de uma imagem
# Além de retornar uma lista, a função também escreve os arrays em um arquivo txt
def folder_processing(classification, folder_path, file):
    data_list = []
    for filepath in glob(folder_path + '\\**'):
        image = imread(filepath)
        resized_image = resize(image, (80, 80), anti_aliasing=True)
        #print('Processing image ' + filepath)
        histograms_data = hog(resized_image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(10, 10), feature_vector=True)
        data = ImageData(classification, histograms_data)
        file.write(data.__repr__() + "\n")
        data_list.append(data)

    return data_list

# Retorna uma lista com todas as imagens da pasta "characters" tendo sido processadas e transcrevidas em objetos ImageData
def create_database_elements():
    map = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
           10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "H", 18: "I",
           19: "J", 20: "K", 21: "L", 22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R",
           28: "S", 29: "T", 30: "U", 31: "V", 32: "W", 33: "X", 34: "Y", 35: "Z", 36: "a",
           37: "b", 38: "c", 39: "d", 40: "e", 41: "f", 42: "g", 43: "h", 44: "i", 45: "j",
           46: "k", 47: "l", 48: "m", 49: "n", 50: "o", 51: "p", 52: "q", 53: "r", 54: "s",
           55: "t", 56: "u", 57: "v", 58: "w", 59: "x", 60: "y", 61: "z"}

    imageData_list = []
    file = io.open("data.txt", "w")
    count = 0

    for path in glob('characters\\**'):
        imageData_list = imageData_list + folder_processing(map[count], path, file)
        count += 1

    file.close()
    return imageData_list

# Cria um numpy array de objetos ImageData baseando-se nos dados de um arquivo txt
def get_elements_fromFile(file):
    imageData_list = []
    for line in file:
        if (line != ""):
            stringList = line.split()
            classification = stringList.pop()

            np = numpy.array(stringList)
            npfloat = np.astype(numpy.float)

            imageData_list.append(ImageData(classification, npfloat))
        else:
            print("Found one")
    return imageData_list

# Calcula a distância pondereada para o algoritmo KNN utilizando o inverso da distância euclidiana elevado a potência de 2
def distance(input_characteristics_array, database_characteristics_array):
    sum = 0.0
    for i in range(0, len(input_characteristics_array)):
        sum += ((input_characteristics_array[i] - database_characteristics_array[i])**2)
    euclidian = math.sqrt(sum)
    return (1/euclidian)**2

# Utiliza o algoritmo KNN pra classificar as imagens na pasta "input"
def knn(imageData_list, k):
    for filepath in glob('input\\**'):
        k_nearest = []
        image = imread(filepath)
        resized_image = resize(image, (80, 80), anti_aliasing=True)
        input_data = hog(resized_image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(10, 10), feature_vector=True)

        for i in range(0, len(imageData_list)):
            ed = distance(input_data, imageData_list[i].characteristics_array)
            k_nearest.append(ImageDistance(imageData_list[i].classification, ed))
            k_nearest.sort(key=lambda x: x.distance, reverse=True) # Inverso porque a distância é ponderada
            if (len(k_nearest) > k):
                k_nearest.pop()

        class_list = [k_nearest[0].classification]
        class_rep = [1]
        for i in range(1, k):
            flag = 0
            for j in range(0, len(class_list)):
                if (k_nearest[i].classification == class_list[j]):
                    class_rep[j] += 1
                    flag = 1
            if(flag == 0):
                class_list.append(k_nearest[i].classification)
                class_rep.append(1)

        maxVal = max(class_rep)

        print(filepath)
        print(class_list)
        print(class_rep)
        for j in range(0, k):
            if (class_rep[j] == maxVal):
                print(k_nearest[j].classification)
                break

try:
    file = io.open("data.txt", "r")
except FileNotFoundError:
    print("Processando dados")
    imageData_list = create_database_elements()
else:
    print("Dados já processados")
    imageData_list = get_elements_fromFile(file)

knn(imageData_list, 5)