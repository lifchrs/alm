import numpy as np
import utils
import gymnasium

class FetchRewardWrapper(gymnasium.Wrapper):
  def reset(self, *args, **kwargs):
    obs, info = self.env.reset(*args, **kwargs)
    return obs, info

  def step(self, action):
    obs, reward, terminated, truncated, info = self.env.step(action)
    achieved_goal = obs['achieved_goal']
    desired_goal = obs['desired_goal']
    reward = -np.linalg.norm(achieved_goal - desired_goal)
    return obs, reward, terminated, truncated, info

class CloseStartFetchWrapper(gymnasium.Wrapper):
    def __init__(self, env, offset=0.05):
        super().__init__(env)
        self.offset = offset  # How close to place gripper to goal

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # import pdb; pdb.set_trace()
        goal = obs['observation'][3:6]
        # exit()
        mujoco_env = self.unwrapped
        model = mujoco_env.model
        data = mujoco_env.data
        # Set the end effector (mocap) to be 2cm above the goal position in z
        goal_above = np.copy(goal)
        goal_above[2] += 0.005  # 2cm above in z
        data.mocap_pos[0] = goal_above

        # Step the simulation several times to let the arm follow the mocap
        for _ in range(20):
            mujoco_env._mujoco.mj_step(model, data)
        mujoco_env._mujoco.mj_forward(model, data)

        # Get fresh observation after setting mocap_pos
        if hasattr(self.env, '_get_obs'):
            obs = self.env._get_obs()
            # print("get_obs function exists")
        elif hasattr(mujoco_env, '_get_obs'):
            obs = mujoco_env._get_obs()
            # print("mujoco_env.get_obs function exists")
        else:
            # print("get_obs function does not exist")
            # Fallback: obs may not be updated correctly
            pass
        info = {}  # Optionally update this if your env provides info
        return obs, info

def make_agent(env, device, cfg):
    
    if cfg.agent == 'alm':
        
        from agents.alm import AlmAgent

        num_states = np.prod(env.observation_space.shape)
        if isinstance(env.observation_space, gymnasium.spaces.Dict):
            if cfg.concat_goal:
                num_states = np.prod(env.observation_space['observation'].shape) + np.prod(env.observation_space['desired_goal'].shape)
            else:
                num_states = np.prod(env.observation_space['observation'].shape)
        num_actions = np.prod(env.action_space.shape)
        action_low = env.action_space.low[0]
        action_high = env.action_space.high[0]

        print(num_states, num_actions)

        if cfg.id == 'Humanoid-v2':
            cfg.env_buffer_size = 1000000
        buffer_size = min(cfg.env_buffer_size, cfg.num_train_steps)

        agent = AlmAgent(device, action_low, action_high, num_states, num_actions,
                            buffer_size, cfg.gamma, cfg.tau, cfg.target_update_interval,
                            cfg.lr, cfg.max_grad_norm, cfg.batch_size, cfg.seq_len, cfg.lambda_cost,
                            cfg.expl_start, cfg.expl_end, cfg.expl_duration, cfg.stddev_clip, 
                            cfg.latent_dims, cfg.hidden_dims, cfg.model_hidden_dims,
                            cfg.wandb_log, cfg.log_interval
                            )
                            
    else:
        raise NotImplementedError

    return agent

def make_env(cfg):
    if cfg.benchmark == 'gym':
        import gymnasium as gym
        import gymnasium_robotics
        gym.register_envs(gymnasium_robotics)

        if cfg.id == 'AntTruncatedObs-v2' or cfg.id == 'HumanoidTruncatedObs-v2':
            utils.register_mbpo_environments()

        def get_env(cfg):
            env = gym.make(cfg.id, render_mode='rgb_array')
            if cfg.close_start:
                print("Close start enabled")
                env = CloseStartFetchWrapper(env)
            if cfg.reward_shaping:
                print("Reward shaping enabled")
                env = FetchRewardWrapper(env)
            # if cfg.entropy_bonus:
            #     env = EntropyBonusWrapper(env)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.reset(seed=cfg.seed)
            env.observation_space.seed(cfg.seed)
            env.action_space.seed(cfg.seed)
            return env 

        train_env = get_env(cfg)
        eval_env = get_env(cfg)
        
        # Register cleanup handlers
        import atexit
        atexit.register(train_env.close)
        atexit.register(eval_env.close)
        
        return train_env, eval_env
    
    else:
        raise NotImplementedError