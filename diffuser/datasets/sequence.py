import importlib
import pdb
from typing import Callable, List, Optional

import numpy as np
import torch

from diffuser.datasets.buffer import ReplayBuffer
from diffuser.datasets.normalization import DatasetNormalizer
from diffuser.datasets.preprocessing import get_preprocess_fn
from diffuser.utils.mask_generator import MultiAgentMaskGenerator


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        env_type: str = "d4rl",
        env: str = "hopper-medium-replay",
        n_agents: int = 2,
        horizon: int = 64,
        normalizer: str = "LimitsNormalizer",
        preprocess_fns: List[Callable] = [],
        use_action: bool = True,
        discrete_action: bool = False,
        max_path_length: int = 1000,
        max_n_episodes: int = 10000,
        termination_penalty: float = 0,
        use_padding: bool = True,  # when path_length is smaller than max_path_length
        discount: float = 0.99,
        returns_scale: float = 400.0,
        include_returns: bool = False,
        include_env_ts: bool = False,
        history_horizon: int = 0,
        agent_share_parameters: bool = False,
        use_seed_dataset: bool = False,
        decentralized_execution: bool = False,
        use_inv_dyn: bool = True,
        use_zero_padding: bool = True,
        agent_condition_type: str = "single",
        pred_future_padding: bool = False,
        seed: Optional[int] = None,
    ):
        if use_seed_dataset:
            assert (
                env_type == "mpe"
            ), f"Seed dataset only supported for MPE, not {env_type}"

        assert agent_condition_type in ["single", "all", "random"], agent_condition_type
        self.agent_condition_type = agent_condition_type

        env_mod_name = {
            "d4rl": "diffuser.datasets.d4rl",
            "mahalfcheetah": "diffuser.datasets.mahalfcheetah",
            "mamujoco": "diffuser.datasets.mamujoco",
            "mpe": "diffuser.datasets.mpe",
            "smac": "diffuser.datasets.smac_env",
            "smacv2": "diffuser.datasets.smacv2_env",
        }[env_type]
        env_mod = importlib.import_module(env_mod_name)

        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = env_mod.load_environment(env)
        self.global_feats = env.metadata["global_feats"]

        self.use_inv_dyn = use_inv_dyn
        self.returns_scale = returns_scale
        self.n_agents = n_agents
        self.horizon = horizon
        self.history_horizon = history_horizon
        self.max_path_length = max_path_length
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None, None]
        self.use_padding = use_padding
        self.use_action = use_action
        self.discrete_action = discrete_action
        self.include_returns = include_returns
        self.include_env_ts = include_env_ts
        self.decentralized_execution = decentralized_execution
        self.use_zero_padding = use_zero_padding
        self.pred_future_padding = pred_future_padding

        if env_type == "mpe":
            if use_seed_dataset:
                itr = env_mod.sequence_dataset(env, self.preprocess_fn, seed=seed)
            else:
                itr = env_mod.sequence_dataset(env, self.preprocess_fn)
        elif env_type == "smac" or env_type == "smacv2":
            itr = env_mod.sequence_dataset(env, self.preprocess_fn)
        else:
            itr = env_mod.sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(
            n_agents,
            max_n_episodes,
            max_path_length,
            termination_penalty,
            global_feats=self.global_feats,
            use_zero_padding=self.use_zero_padding,
        )
        for _, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()#fields到这里的时候，全是40，并且没有norm的那两个key,肯定的这是，normlizer的时候加上了norm，pad_future的时候加了39
        self.normalizer = DatasetNormalizer(
            fields,
            normalizer,
            path_lengths=fields["path_lengths"],
            agent_share_parameters=agent_share_parameters,
            global_feats=self.global_feats,
        )

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1] if self.use_action else 0
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths#path_lengths是1000(n eposide)个40

        self.indices = self.make_indices(fields.path_lengths)#   shape 是 39000, 4
        self.mask_generator = MultiAgentMaskGenerator(
            action_dim=self.action_dim,
            observation_dim=self.observation_dim,
            history_horizon=self.history_horizon,
            action_visible=not use_inv_dyn,
        )
        #连续的动作直接normalize就行
        if self.discrete_action:
            # smac has discrete actions, so we only need to normalize observations
            self.normalize(["observations"])
        else:
            self.normalize()

        self.pad_future()
        if self.history_horizon > 0:#mpe不用history
            self.pad_history()

        print(fields)
        '''
        observations: (1000, 40, 3, 12)
        actions: (1000, 40, 3, 2)
        rewards: (1000, 79, 3, 1)
        terminals: (1000, 79, 3, 1)
        timeouts: (1000, 40, 3, 1)
        normed_observations: (1000, 79, 3, 12)
        normed_actions: (1000, 79, 3, 2)
        '''

    def pad_future(self, keys: List[str] = None):
        if keys is None:
            keys = ["normed_observations", "rewards", "terminals"]#指的是这三个key需要pad39这个跟 最后fields打印的结果也是一致的
            if "legal_actions" in self.fields.keys:
                keys.append("legal_actions")
            if self.use_action:
                if self.discrete_action:
                    keys.append("actions")
                else:
                    keys.append("normed_actions")

        for key in keys:
            shape = self.fields[key].shape
            if self.use_zero_padding:
                self.fields[key] = np.concatenate(
                    [
                        self.fields[key],
                        np.zeros(
                            (shape[0], self.horizon - 1, *shape[2:]),
                            dtype=self.fields[key].dtype,
                        ),
                    ],
                    axis=1,
                )
            else:
                self.fields[key] = np.concatenate(
                    [
                        self.fields[key],
                        np.repeat(
                            self.fields[key][:, -1:],
                            self.horizon - 1,
                            axis=1,
                        ),
                    ],
                    axis=1,
                )

    def pad_history(self, keys: List[str] = None):
        if keys is None:
            keys = ["normed_observations", "rewards", "terminals"]
            if "legal_actions" in self.fields.keys:
                keys.append("legal_actions")
            if self.use_action:
                if self.discrete_action:
                    keys.append("actions")
                else:
                    keys.append("normed_actions")

        for key in keys:
            shape = self.fields[key].shape
            if self.use_zero_padding:
                self.fields[key] = np.concatenate(
                    [
                        np.zeros(
                            (shape[0], self.history_horizon, *shape[2:]),
                            dtype=self.fields[key].dtype,
                        ),
                        self.fields[key],
                    ],
                    axis=1,
                )
            else:
                self.fields[key] = np.concatenate(
                    [
                        np.repeat(
                            self.fields[key][:, :1],
                            self.history_horizon,
                            axis=1,
                        ),
                        self.fields[key],
                    ],
                    axis=1,
                )

    def normalize(self, keys: List[str] = None):
        """
        normalize fields that will be predicted by the diffusion model
        """
        if keys is None:
            keys = ["observations", "actions"] if self.use_action else ["observations"]#指的是，只要这两个key需要归一化，产生norm_act和norm_obs

        for key in keys:
            shape = self.fields[key].shape
            array = self.fields[key].reshape(shape[0] * shape[1], *shape[2:])
            normed = self.normalizer(array, key)
            self.fields[f"normed_{key}"] = normed.reshape(shape)

    def make_indices(self, path_lengths: np.ndarray):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint
        """

        indices = []
        for i, path_length in enumerate(path_lengths):
            if self.use_padding:#true
                max_start = path_length - 1
            else:
                max_start = path_length - self.horizon
                if max_start < 0:
                    continue

            # get `end` and `mask_end` for each `start`
            for start in range(max_start):
                end = start + self.horizon
                mask_end = min(end, path_length)#因为我path_lengh和horizon设的一样长，所以mask_end会一直为40
                indices.append((i, start, end, mask_end))
        indices = np.array(indices)#39000, 4
        return indices

    def get_conditions(self, observations: np.ndarray, agent_idx: Optional[int] = None):
        """
        condition on current observations for planning
        """

        ret_dict = {}
        if self.agent_condition_type == "single":#他在train.py里定义了，decentralized_execution为true就是single
            cond_observations = np.zeros_like(observations[: self.history_horizon + 1])
            cond_observations[:, agent_idx] = observations[
                : self.history_horizon + 1, agent_idx
            ]
            ret_dict["agent_idx"] = torch.LongTensor([[agent_idx]])
        elif self.agent_condition_type == "all":
            cond_observations = observations[: self.history_horizon + 1]
        ret_dict[(0, self.history_horizon + 1)] = cond_observations
        return ret_dict

    def __len__(self):
        if self.agent_condition_type == "single":
            return len(self.indices) * self.n_agents
        else:
            return len(self.indices)

    def __getitem__(self, idx: int, agent_idx: Optional[int] = None):
        if self.agent_condition_type == "single":
            path_ind, start, end, mask_end = self.indices[idx // self.n_agents]#indices的最后一个维度是4
            agent_mask = np.zeros(self.n_agents, dtype=bool)
            agent_mask[idx % self.n_agents] = 1
        elif self.agent_condition_type == "all":
            '''
            idx   path_ind, start, end, mask_end
            36710 [941       11     51    40]
            17386 [445       31     71    40]
            像这样的32个
            mask_end 用来确保只计算有效时间步的数据，padding的0不会计算进去
            '''
            path_ind, start, end, mask_end = self.indices[idx]#第一个是[ 0,  0, 40, 40],每计算一次loss，从39000条路径中，选出batchsize条路
            agent_mask = np.ones(self.n_agents, dtype=bool)
        elif self.agent_condition_type == "random":
            path_ind, start, end, mask_end = self.indices[idx]
            # randomly generate 0 or 1 agent_masks
            agent_mask = np.random.randint(0, 2, self.n_agents, dtype=bool)

        # shift by `self.history_horizon`
        history_start = start
        start = history_start + self.history_horizon
        end = end + self.history_horizon
        mask_end = mask_end + self.history_horizon

        observations = self.fields.normed_observations[path_ind, history_start:end]#那个normed_obs是(1000, 79, 3, 12)，所以每次getitem是(40, 3, 12)
        if self.use_action:
            if self.discrete_action:
                actions = self.fields.actions[path_ind, history_start:end]
            else:
                actions = self.fields.normed_actions[path_ind, history_start:end]#同理每个getitem是(40, 3, 2)

        if self.use_action:
            trajectories = np.concatenate([actions, observations], axis=-1)#(40, 3, 14)
        else:
            trajectories = observations

        if self.use_inv_dyn:
            cond_masks = self.mask_generator(observations.shape, agent_mask)#(40, 3, 12),只有第一个(3,12)全是true，其他的都是false
            cond_trajectories = observations.copy()
        else:
            cond_masks = self.mask_generator(trajectories.shape, agent_mask)
            cond_trajectories = trajectories.copy()
        cond_trajectories[: self.history_horizon, ~agent_mask] = 0.0
        cond = {
            "x": cond_trajectories,
            "masks": cond_masks,
        }

        loss_masks = np.zeros((observations.shape[0], observations.shape[1], 1))#(40, 3, 1)
        if self.pred_future_padding:
            loss_masks[self.history_horizon :] = 1.0#从全0变为了全1
        else:
            loss_masks[self.history_horizon : mask_end - history_start] = 1.0
        if self.use_inv_dyn:
            loss_masks[self.history_horizon, agent_mask] = 0.0#第一个3个1变成了3个0

        attention_masks = np.zeros((observations.shape[0], observations.shape[1], 1))
        attention_masks[self.history_horizon : mask_end - history_start] = 1.0
        attention_masks[: self.history_horizon, agent_mask] = 1.0
        batch = {
            "x": trajectories,  #this is  (40, 3, 14)
            "cond": cond,
            "loss_masks": loss_masks,
            "attention_masks": attention_masks,
        }

        if self.include_returns:
            rewards = self.fields.rewards[path_ind, start : -self.horizon + 1]
            discounts = self.discounts[: len(rewards)]#从1一直到0.6几，(40,1,1)
            returns = (discounts * rewards).sum(axis=0).squeeze(-1)#最后总奖励778
            returns = np.array([returns / self.returns_scale], dtype=np.float32)#return_scale是700
            batch["returns"] = returns#shape (1, 3)

        if self.include_env_ts:#false
            env_ts = (
                np.arange(history_start, start + self.horizon) - self.history_horizon
            )
            env_ts[np.where(env_ts < 0)] = self.max_path_length
            env_ts[np.where(env_ts >= self.max_path_length)] = self.max_path_length
            batch["env_ts"] = env_ts

        if "legal_actions" in self.fields.keys:#false
            batch["legal_actions"] = self.fields.legal_actions[
                path_ind, history_start:end
            ]

        return batch


class ValueDataset(SequenceDataset):
    """
    adds a value field to the datapoints for training the value function
    """

    def __init__(self, *args, discount=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.include_returns is True

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        value_batch = {
            "x": batch["x"],
            "cond": batch["cond"],
            "returns": batch["returns"].mean(axis=-1),
        }
        return value_batch


class BCSequenceDataset(SequenceDataset):
    def __init__(
        self,
        env_type: str = "d4rl",
        env: str = "hopper-medium-replay",
        n_agents: int = 2,
        normalizer: str = "LimitsNormalizer",
        preprocess_fns: List[Callable] = [],
        max_path_length: int = 1000,
        max_n_episodes: int = 10000,
        agent_share_parameters: bool = False,
    ):
        super().__init__(
            env_type=env_type,
            env=env,
            n_agents=n_agents,
            normalizer=normalizer,
            preprocess_fns=preprocess_fns,
            max_path_length=max_path_length,
            max_n_episodes=max_n_episodes,
            agent_share_parameters=agent_share_parameters,
            horizon=1,
            history_horizon=0,
            use_action=True,
            termination_penalty=0.0,
            use_padding=False,
            discount=1.0,
            include_returns=False,
        )

    def __getitem__(self, idx: int):
        path_ind, start, end, _ = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        batch = {"observations": observations, "actions": actions}
        return batch
