
# cifar10 dataset has 10 categories
# and all images have 32x32 size
IMAGE_SIZE = 32
NUM_CLASSES = 10
# if input image has spatial size [32, 32]

SHUFFLE_BUFFER_SIZE = 10000
PREFETCH_BUFFER_SIZE = 1000
NUM_THREADS = 4
# read here about buffer sizes:
# https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle
