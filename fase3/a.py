import os

import pandas as pd
import scipy.io as scipy


class QuestaoA:
    def executar(self):
        file_mat = os.path.join(os.path.dirname(__file__), "data_preg.mat")
        file_csv = os.path.join(os.path.dirname(__file__), "data_preg.csv")

        if os.path.exists(file_mat):
            mat = scipy.loadmat(file_mat)
            if "data" in mat:
                data = mat["data"]
            else:
                keys = [k for k in mat.keys() if not k.startswith("__")]
                data = mat[keys[0]]
            x = data[:, 0]
            y = data[:, 1]
        else:
            data = pd.read_csv(file_csv, header=None)
            x = data[0].values
            y = data[1].values

        return x, y


if __name__ == "__main__":
    QuestaoA().executar()
