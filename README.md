# rdkit_supporter

[![Build Status](https://travis-ci.com/maskot1977/rdkit_installer.svg?branch=aster)](https://travis-ci.com/github/maskot1977/rdkit_installer/)


# The following commands have been deprecated

On Google Colaboratory:

Install RDKit as follows

```python
!wget -c https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
!chmod +x Miniconda3-latest-Linux-x86_64.sh
!bash ./Miniconda3-latest-Linux-x86_64.sh -b -f -p /usr/local
!conda install -q -y -c rdkit rdkit python=3.7
import sys
sys.path.append('/usr/local/lib/python3.7/site-packages/')
```


The following methods are no longer recommended

```python
!pip install git+https://github.com/maskot1977/rdkit_installer.git

from rdkit_installer import install
install.from_miniconda()
```
