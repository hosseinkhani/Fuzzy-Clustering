
class BaseFuzzyCluster(object):
    def distance(self, x):
        raise NotImplementedError

    def update(self, xs, uis, m):
        raise NotImplementedError

