import numcodecs

PREDMOTER_VERSION = "0.3.0"
COMPRESSOR = numcodecs.blosc.Blosc(cname="blosclz", clevel=9, shuffle=2)
# data/X, evaluation, seed limit?, meta?
