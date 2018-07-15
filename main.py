from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from glob import glob
import io
import numpy
from ImageData import ImageData

# Processa todas as imagens em uma determinada pasta e retorna uma lista de objetos ImageData, onde cada objeto contem a lista de caracteristicas e a classificação de uma imagem
# Além de retornar uma lista, a função também escreve os arrays em um arquivo txt
def folder_processing(classification, folder_path, file):
    data_list = []
    for filename in glob(folder_path + '\\**'):
        image = imread(filename)
        resized_image = resize(image, (80, 80), anti_aliasing=True)
        #print('Processing image ' + filename)
        histograms_data = hog(resized_image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), feature_vector=True)
        data = ImageData(classification, histograms_data)
        file.write(data.__repr__() + "\n")
        data_list.append(data)

    return data_list

# Retorna uma lista com todas as imagens de treinamento já processadas
def get_training_elements():
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

try:
    file = io.open("data.txt", "r")
except FileNotFoundError:
    imageData_list = get_training_elements()
else:
    print("Dados já processados")
    imageData_list = get_elements_fromFile(file)