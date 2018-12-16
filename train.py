import numpy as np
import os

import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.multiprocessing import Pipe

from model import CnnActorCriticNetwork, RNDModel
from envs import *
from utils import RunningMeanStd, RewardForwardFilter
from arguments import get_args

from tensorboardX import SummaryWriter

def get_action(model, device, state):
    state = torch.Tensor(state).to(device)
    state = state.float()
    action_probs, value_ext, value_int = model(state)
    action_dist = Categorical(action_probs)
    action = action_dist.sample()

    return action.data.cpu().numpy().squeeze(), value_ext.data.cpu().numpy().squeeze(), value_int.data.cpu().numpy().squeeze(), action_probs.detach().cpu()


def compute_intrinsic_reward(rnd, device, next_obs):
    next_obs = torch.FloatTensor(next_obs).to(device)

    target_next_feature = rnd.target(next_obs)
    predict_next_feature = rnd.predictor(next_obs)
    intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2

    return intrinsic_reward.data.cpu().numpy()


def make_train_data(reward, done, value, gamma, gae_lambda, num_step, num_worker, use_gae):
    discounted_return = np.empty([num_worker, num_step])

    # Discounted Return
    if use_gae:
        gae = np.zeros_like([num_worker, ])
        for t in range(num_step - 1, -1, -1):
            delta = reward[:, t] + gamma * value[:, t + 1] * (1 - done[:, t]) - value[:, t]
            gae = delta + gamma * gae_lambda * (1 - done[:, t]) * gae

            discounted_return[:, t] = gae + value[:, t]

            # For Actor
        adv = discounted_return - value[:, :-1]

    else:
        running_add = value[:, -1]
        for t in range(num_step - 1, -1, -1):
            running_add = reward[:, t] + gamma * running_add * (1 - done[:, t])
            discounted_return[:, t] = running_add

        # For Actor
        adv = discounted_return - value[:, :-1]

    return discounted_return.reshape([-1]), adv.reshape([-1])


def train_model(args, device, output_size, model, rnd, optimizer, s_batch, target_ext_batch, target_int_batch, y_batch, adv_batch, next_obs_batch, old_action_probs):
    epoch = 3
    update_proportion = 0.25
    s_batch = torch.FloatTensor(s_batch).to(device)
    target_ext_batch = torch.FloatTensor(target_ext_batch).to(device)
    target_int_batch = torch.FloatTensor(target_int_batch).to(device)
    y_batch = torch.LongTensor(y_batch).to(device)
    adv_batch = torch.FloatTensor(adv_batch).to(device)
    next_obs_batch = torch.FloatTensor(next_obs_batch).to(device)

    sample_range = np.arange(len(s_batch))
    forward_mse = nn.MSELoss(reduction='none')

    with torch.no_grad():
        action_probs_old_list = torch.stack(old_action_probs).permute(1, 0, 2).contiguous().view(-1, output_size).to(device)

        m_old = Categorical(action_probs_old_list)
        log_prob_old = m_old.log_prob(y_batch)
        # ------------------------------------------------------------

    for i in range(epoch):
        np.random.shuffle(sample_range)
        for j in range(int(len(s_batch) / args.batch_size)):
            sample_idx = sample_range[args.batch_size * j:args.batch_size * (j + 1)]

            # --------------------------------------------------------------------------------
            # for Curiosity-driven(Random Network Distillation)
            predict_next_state_feature, target_next_state_feature = rnd(next_obs_batch[sample_idx])

            forward_loss = forward_mse(predict_next_state_feature, target_next_state_feature.detach()).mean(-1)
            # Proportion of exp used for predictor update
            mask = torch.rand(len(forward_loss)).to(device)
            mask = (mask < update_proportion).type(torch.FloatTensor).to(device)
            forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(device))
            # ---------------------------------------------------------------------------------

            action_probs, value_ext, value_int = model(s_batch[sample_idx])
            m = Categorical(action_probs)
            log_prob = m.log_prob(y_batch[sample_idx])

            ratio = torch.exp(log_prob - log_prob_old[sample_idx])

            surr1 = ratio * adv_batch[sample_idx]
            surr2 = torch.clamp(
                ratio,
                1.0 - args.eps,
                1.0 + args.eps) * adv_batch[sample_idx]

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_ext_loss = F.mse_loss(value_ext.sum(1), target_ext_batch[sample_idx])
            critic_int_loss = F.mse_loss(value_int.sum(1), target_int_batch[sample_idx])

            critic_loss = critic_ext_loss + critic_int_loss

            entropy = m.entropy().mean()

            optimizer.zero_grad()
            loss = actor_loss + 0.5 * critic_loss - args.entropy_coef * entropy + forward_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()


def main():
    args = get_args()
    device = torch.device('cuda' if args.cuda else 'cpu')

    env = gym.make(args.env_name)

    input_size = env.observation_space.shape  # 4
    output_size = env.action_space.n  # 2

    if 'Breakout' in args.env_name:
        output_size -= 1

    env.close()

    is_render = False
    if not os.path.exists(args.save_dir):
	    os.makedirs(args.save_dir)
    model_path = os.path.join(args.save_dir, args.env_name + '.model')
    predictor_path = os.path.join(args.save_dir, args.env_name + '.pred')
    target_path = os.path.join(args.save_dir, args.env_name + '.target')    

    writer = SummaryWriter(log_dir=args.log_dir)

    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
    discounted_reward = RewardForwardFilter(args.ext_gamma)

    model = CnnActorCriticNetwork(input_size, output_size, args.use_noisy_net)
    rnd = RNDModel(input_size, output_size)
    model = model.to(device)
    rnd = rnd.to(device)
    optimizer = optim.Adam(list(model.parameters()) + list(rnd.predictor.parameters()), lr=args.lr)
   
    if args.load_model:
        if args.cuda:
            model.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))

    works = []
    parent_conns = []
    child_conns = []
    for idx in range(args.num_worker):
        parent_conn, child_conn = Pipe()
        work = AtariEnvironment(
        	args.env_name,
            is_render, 
        	idx, 
        	child_conn, 
        	sticky_action=args.sticky_action, 
        	p=args.sticky_action_prob,
        	max_episode_steps=args.max_episode_steps)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    states = np.zeros([args.num_worker, 4, 84, 84])

    sample_env_index = 0   # Sample Environment index to log
    sample_episode = 0
    sample_rall = 0
    sample_step = 0
    sample_i_rall = 0
    global_update = 0
    global_step = 0

    # normalize observation
    print('Initializes observation normalization...')
    next_obs = []
    for step in range(args.num_step * args.pre_obs_norm_steps):
        actions = np.random.randint(0, output_size, size=(args.num_worker,))

        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)

        for parent_conn in parent_conns:
            next_state, reward, done, realdone, log_reward = parent_conn.recv()
            next_obs.append(next_state[3, :, :].reshape([1, 84, 84]))

        if len(next_obs) % (args.num_step * args.num_worker) == 0:
            next_obs = np.stack(next_obs)
            obs_rms.update(next_obs)
            next_obs = []

    print('Training...')
    while True:
        total_state, total_reward, total_done, total_next_state, total_action, total_int_reward, total_next_obs, total_ext_values, total_int_values, total_action_probs = [], [], [], [], [], [], [], [], [], []
        global_step += (args.num_worker * args.num_step)
        global_update += 1

        # Step 1. n-step rollout
        for _ in range(args.num_step):
            actions, value_ext, value_int, action_probs = get_action(model, device, np.float32(states) / 255.)

            for parent_conn, action in zip(parent_conns, actions):
                parent_conn.send(action)

            next_states, rewards, dones, real_dones, log_rewards, next_obs = [], [], [], [], [], []
            for parent_conn in parent_conns:
                next_state, reward, done, real_done, log_reward = parent_conn.recv()
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)
                real_dones.append(real_done)
                log_rewards.append(log_reward)
                next_obs.append(next_state[3, :, :].reshape([1, 84, 84]))

            next_states = np.stack(next_states)
            rewards = np.hstack(rewards)
            dones = np.hstack(dones)
            real_dones = np.hstack(real_dones)
            next_obs = np.stack(next_obs)

            # total reward = int reward + ext Reward
            intrinsic_reward = compute_intrinsic_reward(rnd, device, 
                ((next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5))
            intrinsic_reward = np.hstack(intrinsic_reward)
            sample_i_rall += intrinsic_reward[sample_env_index]

            total_next_obs.append(next_obs)
            total_int_reward.append(intrinsic_reward)
            total_state.append(states)
            total_reward.append(rewards)
            total_done.append(dones)
            total_action.append(actions)
            total_ext_values.append(value_ext)
            total_int_values.append(value_int)
            total_action_probs.append(action_probs)

            states = next_states[:, :, :, :]

            sample_rall += log_rewards[sample_env_index]

            sample_step += 1
            if real_dones[sample_env_index]:
                sample_episode += 1
                writer.add_scalar('data/reward_per_epi', sample_rall, sample_episode)
                writer.add_scalar('data/reward_per_rollout', sample_rall, global_update)
                writer.add_scalar('data/step', sample_step, sample_episode)
                sample_rall = 0
                sample_step = 0
                sample_i_rall = 0

        # calculate last next value
        _, value_ext, value_int, _ = get_action(model, device, np.float32(states) / 255.)
        total_ext_values.append(value_ext)
        total_int_values.append(value_int)
        # --------------------------------------------------

        total_state = np.stack(total_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
        total_reward = np.stack(total_reward).transpose().clip(-1, 1)
        total_action = np.stack(total_action).transpose().reshape([-1])
        total_done = np.stack(total_done).transpose()
        total_next_obs = np.stack(total_next_obs).transpose([1, 0, 2, 3, 4]).reshape([-1, 1, 84, 84])
        total_ext_values = np.stack(total_ext_values).transpose()
        total_int_values = np.stack(total_int_values).transpose()
        total_logging_action_probs = np.vstack(total_action_probs)

        # Step 2. calculate intrinsic reward
        # running mean intrinsic reward
        total_int_reward = np.stack(total_int_reward).transpose()
        total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in total_int_reward.T])
        mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
        reward_rms.update_from_moments(mean, std ** 2, count)

        # normalize intrinsic reward
        total_int_reward /= np.sqrt(reward_rms.var)
        writer.add_scalar('data/int_reward_per_epi', np.sum(total_int_reward) / args.num_worker, sample_episode)
        writer.add_scalar('data/int_reward_per_rollout', np.sum(total_int_reward) / args.num_worker, global_update)
        # -------------------------------------------------------------------------------------------

        # logging Max action probability
        writer.add_scalar('data/max_prob', total_logging_action_probs.max(1).mean(), sample_episode)

        # Step 3. make target and advantage
        # extrinsic reward calculate
        ext_target, ext_adv = make_train_data(total_reward,
                                              total_done,
                                              total_ext_values,
                                              args.ext_gamma,
                                              args.gae_lambda,
                                              args.num_step,
                                              args.num_worker,
                                              args.use_gae)

        # intrinsic reward calculate
        # None Episodic
        int_target, int_adv = make_train_data(total_int_reward,
                                              np.zeros_like(total_int_reward),
                                              total_int_values,
                                              args.int_gamma,
                                              args.gae_lambda,
                                              args.num_step,
                                              args.num_worker,
                                              args.use_gae)

        # add ext adv and int adv
        total_adv = int_adv * args.int_coef + ext_adv * args.ext_coef
        # -----------------------------------------------

        # Step 4. update obs normalize param
        obs_rms.update(total_next_obs)
        # -----------------------------------------------

        # Step 5. Training!
        train_model(args, device, output_size, model, rnd, optimizer, 
                        np.float32(total_state) / 255., ext_target, int_target, total_action,
                        total_adv, ((total_next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5),
                        total_action_probs)

        if global_step % (args.num_worker * args.num_step * args.save_interval) == 0:
            print('Now Global Step :{}'.format(global_step))
            torch.save(model.state_dict(), model_path)
            torch.save(rnd.predictor.state_dict(), predictor_path)
            torch.save(rnd.target.state_dict(), target_path)


if __name__ == '__main__':
    main()
