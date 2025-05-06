import wandb
import hydra
import warnings
warnings.simplefilter("ignore", UserWarning)

from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path='cfgs', config_name='config', version_base=None)
def main(cfg: DictConfig):
    if cfg.benchmark == 'gym':
        from workspaces.mujoco_workspace import MujocoWorkspace as W
    else:
        raise NotImplementedError

    if cfg.wandb_log:
        project_name = 'alm_' + cfg.id
        with wandb.init(project=project_name, entity='lifchrs_mbt52_4756', config=dict(cfg), settings=wandb.Settings(start_method="thread")):
            if wandb.run:
                # Convert wandb.config to a dictionary and ensure lr is a dictionary
                wandb_config_dict = dict(wandb.config)
                if isinstance(wandb_config_dict.get('lr'), str):
                    # If lr is a string, parse it back to a dictionary
                    import ast
                    wandb_config_dict['lr'] = ast.literal_eval(wandb_config_dict['lr'])
                cfg = OmegaConf.merge(cfg, OmegaConf.create(wandb_config_dict))
            # import pdb; pdb.set_trace()
            wandb.run.name = cfg.wandb_run_name
            workspace = W(cfg)
            print("seq_len used:", cfg.seq_len)
            print("update_interval used:", cfg.update_interval)
            workspace.train()
    else:
        workspace = W(cfg)
        workspace.train()
    
if __name__ == '__main__':
    main()