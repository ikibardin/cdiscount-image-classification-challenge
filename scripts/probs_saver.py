import numpy as np
from pandas import HDFStore,DataFrame
from config import PREDICT_PROBS_PATH

class ProbStore:
    def __init__(self, path=PREDICT_PROBS_PATH):
        self._hdf = HDFStore(path)
        self._path = path
        self._length = 0

    def saveProbs(self, data, index):
        """
        data -- a DataFrame for one image, index -- index of an image in test
        """
        assert isinstance(data, pd.DataFrame)
        self._hdf.put(str(index), data, table=True)
        self._length += 1

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        return self._hdf[str(index)]

    def __del__(self):
        self._hdf.close()
