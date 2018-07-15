class ImageData:
    def __init__(self, classif, characteristics):
        self.classification = classif
        self.characteristics_array = characteristics

    def __repr__(self):
        string = ""
        for i in range(0, len(self.characteristics_array)):
            string = string + str(self.characteristics_array[i]) + " "
        return string + self.classification