#!/usr/bin/env python
# coding=utf-8

import timeit

normal_py_sec = timeit.timeit('sum(x*x for x in xrange(1000))', number=10000)

naive_np_sec = timeit.timeit('sum(na*na)', setup='import numpy as np; na=np.arange(1000)', number=10000)

good_np_sec = timeit.timeit('na.dot(na)', setup='import numpy as np; na=np.arange(1000)', number=10000)

print "Normal Python: {0} sec".format(normal_py_sec)

print "Naive Numpy:{0} sec".format(naive_np_sec)

print "Good Numpy: {0} sec".format(good_np_sec)


