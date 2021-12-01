import copy

from rdkit import Chem
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Chem.Fingerprints import FingerprintMols


class Fingerprinter:
    def __init__(self):
        self.binaryfp_names = [
            "MACCSkeys",
            "Avalon",
            "Morgan2(1024bits)",
            "Morgan2F(1024bits)",
            "Morgan4(2048bits)",
            "Morgan4F(2048bits)",
            # "AtomPair",
            # "Topological",
            # "TopologicalTortion",
        ]
        self.binaryfp = [
            lambda mol: MACCSkeys.GenMACCSKeys(mol),
            lambda mol: pyAvalonTools.GetAvalonFP(mol),
            lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024),
            lambda mol: AllChem.GetMorganFingerprintAsBitVect(
                mol, 2, nBits=1024, useFeatures=True
            ),
            lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048),
            lambda mol: AllChem.GetMorganFingerprintAsBitVect(
                mol, 4, nBits=2048, useFeatures=True
            ),
            # lambda mol: Pairs.GetAtomPairFingerprintAsBitVect(mol), # クラッシュする
            # lambda mol: FingerprintMols.FingerprintMol(mol), #Topological Fingerprint # NaNを生成する
            # lambda mol: Torsions.GetTopologicalTorsionFingerprintAsIntVect(mol), # ToBitString を持ってない
        ]
        self.countfp_names = [
            "ECFP2",
            "FCFP2",
            "ECFP4",
            "FCFP4",
            "ECFP6",
            "FCFP6",
        ]
        self.countfp = [
            lambda mol: AllChem.GetMorganFingerprint(
                mol, radius=1, bitInfo=self.bit_info, useFeatures=False
            ),
            lambda mol: AllChem.GetMorganFingerprint(
                mol, radius=1, bitInfo=self.bit_info, useFeatures=True
            ),
            lambda mol: AllChem.GetMorganFingerprint(
                mol, radius=2, bitInfo=self.bit_info, useFeatures=False
            ),
            lambda mol: AllChem.GetMorganFingerprint(
                mol, radius=2, bitInfo=self.bit_info, useFeatures=True
            ),
            lambda mol: AllChem.GetMorganFingerprint(
                mol, radius=3, bitInfo=self.bit_info, useFeatures=False
            ),
            lambda mol: AllChem.GetMorganFingerprint(
                mol, radius=3, bitInfo=self.bit_info, useFeatures=True
            ),
        ]
        self.bit_info = {}
        self.bit_infos = {}
        self.vectors = []
        self.all_bit_info_keys = {}
        self.mols = []

    @property
    def names(self):
        return self.binaryfp_names + self.countfp_names

    def calc_fingerprints(self, mol, fp_type):
        return self.binaryfp[self.binaryfp_names.index(fp_type)](mol)

    def to_bit_string(self, mol, fp_type):
        return self.calc_fingerprints(mol, fp_type).ToBitString()

    def to_list(self, mol, fp_type):
        self.mols.append(mol)
        vec = [int(x) for x in self.to_bit_string(mol, fp_type)]
        return vec

    def calc_bit_info(self, mol, fp_type):
        self.mols.append(mol)
        vec = self.countfp[self.countfp_names.index(fp_type)](mol)
        if fp_type not in self.bit_infos.keys():
            self.bit_infos[fp_type] = []
        self.bit_infos[fp_type].append(copy.deepcopy(self.bit_info))
        return

    def transform(self, smiles, fp_type="MACCSkeys", refresh_mols=True):

        self.vectors = []
        if refresh_mols:
            self.mols = []

        if fp_type in self.binaryfp_names:
            for smile in smiles:
                yield self.to_list(Chem.MolFromSmiles(smile), fp_type)

        elif fp_type in self.countfp_names:
            self.bit_info = {}
            self.bit_infos[fp_type] = []
            self.all_bit_info_keys[fp_type] = []

            for smile in smiles:
                self.calc_bit_info(Chem.MolFromSmiles(smile), fp_type)

            for bit_info in self.bit_infos[fp_type]:
                self.all_bit_info_keys[fp_type] += list(bit_info)
            self.all_bit_info_keys[fp_type] = list(set(self.all_bit_info_keys[fp_type]))

            for i, smile in enumerate(smiles):
                # print(i, smile)
                vec = []
                for key in self.all_bit_info_keys[fp_type]:
                    if key in self.bit_infos[fp_type][i].keys():
                        vec.append(len(self.bit_infos[fp_type][i][key]))
                    else:
                        vec.append(0)
                yield vec

        else:
            raise ("That type of FP is not supported.")
