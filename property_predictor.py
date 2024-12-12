from DeepPurpose import CompoundPred as models
from DeepPurpose.utils import *
from DeepPurpose.dataset import *
import pandas as pd
import numpy as np

def train_model(smiles, activity):
    drug_encoding = 'rdkit_2d_normalized'
    train, val, test = data_process(X_drug = smiles, y = activity, 
			                        drug_encoding = drug_encoding,
			                        split_method='random', 
			                        random_seed = 42)

    config = generate_config(drug_encoding = drug_encoding, 
                            cls_hidden_dims = [512], 
                            train_epoch = 10, 
                            LR = 0.001, 
                            batch_size = 128,
                            )
    model = models.model_initialize(**config)
    model.train(train, val, test)       

    return model

def predict_activity(model, smile):
    
    drug_encoding = 'rdkit_2d_normalized'
    
    if type(smile) == np.ndarray:
        new_smiles = smile
    else:
        new_smiles = np.array([smile])
    
    label = np.array([0])
    
    new_smiles_processed = data_process(X_drug = new_smiles, 
                                        y = label, 
                                        drug_encoding = drug_encoding, 
                                        split_method='no_split', 
                                        random_seed = 42)

    activity = model.predict(new_smiles_processed)

    return activity

