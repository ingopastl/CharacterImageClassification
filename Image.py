class ImageData:
    def __init__(self, classification, characteristics):
        self.classification = classification
        self.characteristics_array = characteristics

    def __repr__(self):
        string = ""
        for i in range(0, len(self.characteristics_array)):
            string = string + str(self.characteristics_array[i]) + " "
        return string + self.classification

class ImageDistance:
    def __init__(self, classification, distance):
        self.classification = classification
        self.distance = distance