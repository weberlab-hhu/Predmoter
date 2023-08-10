# Manual install
## Helixer
Predmoter works hand in hand with Helixer. There are two scripts from Helixer
that are necessary for Predmoter to work: ``fasta2h5.py``, used to create the h5
files from fasta files, and ``add_ngs_coverage.py``, can be found in ``helixer/evaluation``
and is used to add NGS data (bam files) to an existing h5 file created with ``fasta2h5.py``
for training and/or testing. The built-in functions to convert fasta to
h5 files are also used by Predmoter to predict on fasta files without the user having
to convert them manually.    
     
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
