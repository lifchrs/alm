import wandb, os

sweep_id = "3jqna3xu"  

def train():
    os.system("python train.py")  # hydra + wandb works as expected

wandb.agent(sweep_id, function=train, project="alm_Humanoid-v4", entity="lifchrs_mbt52_4756")