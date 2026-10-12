from pathlib import Path



class TextFileManager:


    def __init__(self, files):

        self.files = files



    def __iter__(self):

        for file in self.files:

            yield Path(file)
