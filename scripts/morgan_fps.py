import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# read the CSV file into a Pandas DataFrame
df = pd.read_csv('../Data/Drug_Full_Info.csv')

# generate Morgan fingerprints for each SMILES string in the DataFrame
fps = []
for smi in df['CanonicalSMILES']:
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    arr = np.zeros((0,), dtype=np.int8)
    fp.ToBitString().replace('0', '-1').replace('1', '1')
    for bit in fp:
        arr = np.append(arr, [int(bit)])
    fps.append(arr)

# convert the list of fingerprints into a Pandas DataFrame
fps_df = pd.DataFrame(fps)

# add the fingerprints DataFrame to the input DataFrame
df = pd.concat([df, fps_df], axis=1)

# save the input DataFrame with the fingerprints as a CSV file
df.to_csv('../Data/Drug_Full_MFP_Info.csv', index=False)

