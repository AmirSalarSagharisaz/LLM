class TextStream:

    def __init__(self, files, chunk_size=1024*1024):
        self.files = files
        self.chunk_size = chunk_size

    def __iter__(self):
        for file in self.files:
            with open(file, "r", encoding="utf-8") as f:
                while True:
                    chunk = f.read(self.chunk_size)
                    if not chunk:
                        break
                    yield chunk
