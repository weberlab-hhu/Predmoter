from setuptools import setup
from predmoter.core.constants import PREDMOTER_VERSION  # works without install???

# exec(open("predmoter/core/constants.py").read())  # works

setup(
   name="predmoter",
   version=PREDMOTER_VERSION,
   author="Felicitas Kindel",
   url="https://github.com/weberlab-hhu/Predmoter",
   description="Deep Learning model predicting ATAC- and ChIP-seq data",
   packages=["predmoter", "predmoter.core", "predmoter.prediction", "predmoter.utilities"],
   scripts=["Predmoter.py"]
)
# later: package_data, install_requires (packages/git), dependency_links
