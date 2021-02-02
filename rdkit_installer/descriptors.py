import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

def calc_descriptors(smiles):
  calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
  desc_names = [x[0] for x in Descriptors._descList if x[0]]

  matrix = []
  for smile in smiles:
    row = []
    row.append(smile)
    mol = Chem.MolFromSmiles(smile)
    for d in calc.CalcDescriptors(mol):
        row.append(d)
    matrix.append(row)
    
  return pd.DataFrame(matrix, columns=desc_names)
