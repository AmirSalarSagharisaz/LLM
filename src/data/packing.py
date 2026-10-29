import numpy as np
class SequencePacker:
    def __init__(self, sequence_length):
        self.sequence_length = (sequence_length)

    def pack(self, tokens):
        total = len(tokens)
        for start in range(0, total-self.sequence_length, self.sequence_length):
            chunk = tokens[start: start+self.sequence_length+1]
            inputs = chunk[:-1]
            targets = chunk[1:]
            yield (np.array(inputs, dtype=np.int32), np.array(targets, dtype=np.int32))
