from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels, DataSet
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.platform import gfile
from tensorflow.python.framework import dtypes
# CVDF mirror of http://yann.lecun.com/exdb/mnist/
DEFAULT_SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'

# a version of tensorflow.examples.tutorials.mnist.input_data.read_data_sets() that only reads the training set
def read_data_sets(train_dir, one_hot=False, dtype=dtypes.float32, reshape=True, seed=None, source_url=DEFAULT_SOURCE_URL):
    if not source_url:  # empty string check
        source_url = DEFAULT_SOURCE_URL
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'

    local_file = base.maybe_download(TRAIN_IMAGES, train_dir,
                                     source_url + TRAIN_IMAGES)
    with gfile.Open(local_file, 'rb') as f:
      train_images = extract_images(f)

    local_file = base.maybe_download(TRAIN_LABELS, train_dir,
                                     source_url + TRAIN_LABELS)
    with gfile.Open(local_file, 'rb') as f:
      train_labels = extract_labels(f, one_hot=one_hot)

    options = dict(dtype=dtype, reshape=reshape, seed=seed)

    train = DataSet(train_images, train_labels, **options)

    return base.Datasets(train=train, validation=None, test=None)