.. -*- mode: rst -*-
How to use the .dump files
------------

These files are python data structures serialized with cPickle library. 


.. code:: python

	# the code is compatible with python2 and requires numpy
  	import cPickle as pickle
  	splits = pickle.load(open('splits.dump'))
  	print len(splits) # 10 cross-validation splits
  	print len(splits[0]) # 6 elements of each split: feature vectors belonging to the training (element0) and testing (1) data
         	             # classes of object belonging to the training (2) and testing (3) data, and
                	     # seq ids of viruses belonging to the training (4) and testing (5) data, respectively
  	print splits[0][5]   # an example list of ids belonging to the testing set of the first cross-validation split

Citation
--------
.. image:: https://zenodo.org/badge/70039462.svg
   :target: https://zenodo.org/badge/latestdoi/70039462
