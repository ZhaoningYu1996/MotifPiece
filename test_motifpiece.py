from utils.motifpiece import MotifPiece

datasets = ["O=C1[C@@H]2C=C[C@@H](C=CC2)C1(c1ccccc1)c1ccccc1", 'CCOC(=O)c1cccc(Nc2c(-c3ccc(N(C)C)cc3)nc3cnccn32)c1', 'O=C(O)CCN1C(=O)C(O)=C(C(=O)c2ccc(Cl)cc2)[C@@H]1c1ccc(Cl)cc1']
# datasets = ["O=C1[C@@H]2C=C[C@@H](C=CC2)C1(c1ccccc1)c1ccccc1"]

motifpiece = MotifPiece(datasets, "saved_vocabulary/")

extracted_motifs = motifpiece.inference("O=C1[C@@H]2C=C[C@@H](C=CC2)C1(c1ccccc1)c1ccccc1")
print(extracted_motifs)