
class BaseFuzzyCluster(object):
    def distance(self, x):
        """
        distance from cluster to point x
        usage in membership evaluation
        :param x: data point
        :return: distance**2
        """
        raise NotImplementedError

    def update(self, xs, us, m, ci):
        """
        update its variables based on membership values
        :param xs: list of data points
        :param us: list of memberships for all clusteres
        :param m:
        :param ci: cluster num
        """
        self._update(xs, us[:, ci], m)

    def _update(self, xs, uis, m):
        """
        update its variables based on membership values
        :param xs: list of data points
        :param uis: list of memberships for this cluster
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
