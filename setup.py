from setuptools import setup
from predmoter.core.constants import PREDMOTER_VERSION

setup(
   name="predmoter",
   version=PREDMOTER_VERSION,
   author="Felicitas Kindel",
   url="https://github.com/weberlab-hhu/Predmoter",
   description="Deep Learning model predicting ATAC- and ChIP-seq data",
   packages=["predmoter", "predmoter.core", "predmoter.prediction", "predmoter.utilities"],
   package_data={"predmoter": ["testdata/*.h5", "testdata/*.fa"]},
   install_requires=["helixer @ https://github.com/weberlab-hhu/Helixer/archive/refs/heads/dev.zip"],
   dependency_links=["https://github.com/weberlab-hhu/Helixer/archive/refs/heads/dev.zip#egg=helixer"],
   scripts=["Predmoter.py", "convert2coverage.py"]
)
