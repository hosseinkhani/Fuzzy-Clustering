
class BaseMultivariateFuzzyCluster(object):
    def distance(self, x, j):
        """
        distance from cluster to point x
        usage in membership evaluation
        :param x: data point
        :param j: jth feature
        :return: distance**2
        """
        raise NotImplementedError

    def update(self, xs, us, j, m, ci):
        """
        update its variables based on membership values
        :param xs: list of data points
        :param us: list of memberships for all clusteres
        :param j: jth feature
        :param m:
        :param ci: cluster num
        """
        self._update(xs, us[:, ci], m)

    def _update(self, xs, uis, j, m):
        """
        update its variables based on membership values
        :param xs: list of data points
        :param uis: list of memberships for this cluster
        :param j: jth feature
        :param m:
        """
        raise NotImplementedError

    def draw(self):
        """
        :return: a matplotlib patch
        """
        raise NotImplementedError

    def center(self):
        """
        :return: center point of cluster
        """
        raise NotImplementedError
