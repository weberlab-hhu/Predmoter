# Manual install
## Helixer
Predmoter works hand in hand with Helixer. There are only two scripts from Helixer
that are necessary for Predmoter to work: ``fasta2h5.py`` and ``add_ngs_coverage.py``.
The first is used to create the h5 files Predmoter works with from fasta files.
If Predmoter is only used to predict, this is the only script needed. The
``add_ngs_coverage.py`` in ``helixer/evaluation`` is used to add NGS data (bam files)
to an existing h5 file created with ``fasta2h5.py`` for training and/or testing.

## Get the code
### Predmoter
```bash
# clone via HTTPS
git clone https://github.com/weberlab-hhu/Predmoter.git

# allows access via SSH
git clone git@github.com:weberlab-hhu/Predmoter.git
```
      
### Helixer
```bash
# clone via HTTPS
git clone https://github.com/weberlab-hhu/Helixer.git

# allows access via SSH
git clone git@github.com:weberlab-hhu/Helixer.git
```
       
## Virtual environment (optional)
The installation of all the python packages in a virtual environment is recommended:
https://docs.python-guide.org/dev/virtualenvs/,
https://docs.python.org/3/library/venv.html.

For example, create and activate an environment called "env":
```bash
python3 -m venv env
source env/bin/activate
# install everything

# deactivation
deactivate
```
     
## Installation
```bash
# Helixer
cd Helixer
git checkout dev  # dev branch contains --shift option for ATAC-seq
pip install -r requirements.txt
cd ..

# Predmoter
cd Predmoter
pip install -r requirements.txt
pip install .  # or 'pip install -e .', if you will be changing the code

# Helixer 
# force Helixer installation to be the manual one
cd ../Helixer
pip install .  # or 'pip install -e .', if you will be changing the code
```
