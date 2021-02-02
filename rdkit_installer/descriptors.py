import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

def calc_descriptors(smiles):
  desc_names = [x[0] for x in Descriptors._descList if x[0]]
  calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

  matrix = []
  for smile in smiles:
    row = []
    mol = Chem.MolFromSmiles(smile)
    for d in calc.CalcDescriptors(mol):
        row.append(d)
    matrix.append(row)
    
  return pd.DataFrame(matrix, columns=desc_names)
