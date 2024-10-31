from copy import deepcopy
import logging
import wandb
import pandas as pd

# Remove a key in a nested dictionary
def scrub_key(obj, bad_key):
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            if key == bad_key:
                del obj[key]
            else:
                scrub_key(obj[key], bad_key)

def scrub_dictionary(self, in_dict: dict, remove_keys: list, add_round_back=True):
    scrubbed_dict = deepcopy(in_dict)
    for key in remove_keys:
        scrub_key(scrubbed_dict, key)
    if "round" in remove_keys:
        if add_round_back:
            scrubbed_dict["round"] = self._round
    return scrubbed_dict



def save_as_csv(_round , result_dict: dict, filename="results.csv"):
    
    df = pd.json_normalize(result_dict)
    if _round == 0:
        df.to_csv(filename, mode="w", index=False, header=True)
    else:
        df.to_csv(filename, mode="a", index=False, header=False)



# def log_to_wandb(metrics: dict, metric:str, stage: str ="", actor: str = "" commit=False):

#     flat = pd.json_normalize(metrics, sep="/").to_dict(orient="records")[0]
