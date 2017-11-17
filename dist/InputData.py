import numpy as np

def readLines(path):
    ret = []
    with open(path, 'r') as f:
        ret.extend(f.readlines())
    ret = [line.strip() for line in ret]
    return ret

def one_hop(label, is_test=False, segment=None):
    if segment == 0:
        return [-2, -2]
    ret = [1, 0]
    if label[0] == 1:
        ret = [0, 1]
    if is_test:
        if label[0] == -2:
            ret = [-2, -2]
        elif label[0] == 2:
            ret = [-2, -2]
    return ret

def get_features(ids, all, pssms, sdss, seqs, win_size):
    features = []
    labels = []
    test_features = []
    test_labels = []
    for id in ids:
        index = all.index(id)
        pssm = pssms[index]
        sds = sdss[index]
        seq = seqs[index]
        for i in xrange(len(seq)):
            vals = []
            for j in xrange(i - win_size, i + win_size + 1):
                if j < 0:
                    vals.append(np.zeros(23))
                elif j >= len(seqs[index]):
                    vals.append(np.zeros(23))
                else:
                    vals.append(pssm[j])
            feature = np.concatenate(vals, axis=0)
            features.append(feature)
            label = sds[i]
            labels.append(label)

            if len(sds[i]) == 3:
                if id.startswith('bmr'):
                    test_features.append(feature)
                    test_labels.append(label)
            else:
                test_features.append(feature)
                test_labels.append(label)
    features = np.reshape(np.array(features), (len(features), 1, win_size * 2 + 1, 23))
    test_features = np.reshape(np.array(test_features), (len(test_features), 1, win_size * 2 + 1, 23))
    return features, np.array(labels), test_features, np.array(test_labels)

class TestData:
    def __init__(self, ids, seqs, pssms, sdss, window_size):

        self._win_size = window_size
        self._ids = ids
        self._seqs = seqs
        pssms = pssms
        labels = sdss

        self._features, self._labels, _, _ = get_features(list(self._ids), self._ids, pssms, labels, self._seqs, self._win_size)

    def getData(self):
        return self._ids, self._seqs, self._features, self._labels