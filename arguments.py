import argparse
import os
import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', default='rnd',
                        help='Algorithm to use: rnd | ppo')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--num-worker', type=int, default=128,
                        help='Number of workers (CPU processes) to use (default: 16)')
    parser.add_argument('--num-step', type=int, default=128,
                        help='Number of forward steps (default: 128)')
    parser.add_argument('--eps', type=float, default=0.1,
                        help='Epsilon (default: 0.1)')
    parser.add_argument('--ext-gamma', type=float, default=0.999,
                        help='Extrinsic discount factor for rewards (default: 0.999)')
    parser.add_argument('--int-gamma', type=float, default=0.99,
                        help='Intrinsic discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=True,
                        help='use generalized advantage estimation (default: True)')
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="Lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Use GPU training (default: True)')
    parser.add_argument('--use-noisy-net', action='store_true', default=False,
                        help='Use NoisyNet (default: False)')
    parser.add_argument('--no-sticky-action', action='store_true', default=False,
                        help='Use Sticky Action (default: True)')    
    parser.add_argument("--sticky-action-prob", type=float, default=0.25,
                        help="Action probability (default: 0.25")
    parser.add_argument('--epoch', type=int, default=4,
                        help='number of epochs (default: 4)')
    parser.add_argument('--mini-batch', type=int, default=4,
                        help='Number of batches (default: 4)')
    parser.add_argument('--entropy-coef', type=float, default=0.001,
                        help='entropy term coefficient (default: 0.001)')
    parser.add_argument('--ext-coef', type=float, default=2.,
                        help='entropy term coefficient (default: 2.)')    
    parser.add_argument('--int-coef', type=float, default=1.,
                        help='entropy term coefficient (default: 1.)')  
    parser.add_argument('--max-episode-steps', type=int, default=4500,
                        help='Maximum steps per episode (default: 18000)')
    parser.add_argument('--pre-obs-norm-steps', type=int, default=50,
                        help='Number of steps for pre-normalization (default: 50)')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='Save interval, one save per n updates (default: 100)')
    parser.add_argument('--load-model', action='store_true', default=False,
                        help='Load pre-trained Model (default: False)')
    parser.add_argument('--log-dir', default=None,
                        help='Directory to save agent logs (default: runs/CURRENT_DATETIME_HOSTNAME)')
    parser.add_argument('--save-dir', default='trained_models',
                        help='Directory to save agent logs (default: trained_models)')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='Use a recurrent policy')

    parser.add_argument('--env-name', default='MontezumaRevengeNoFrameskip-v4',
                        help='Environment to train on (default: MontezumaRevengeNoFrameskip-v4)')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.batch_size = int(args.num_step * args.num_worker / args.mini_batch)
    if args.log_dir is not None:
        args.log_dir = os.path.join("runs", args.log_dir)
    args.sticky_action = not args.no_sticky_action

    print("GPU training: ", args.cuda)
    print("Sticky actions: ", args.sticky_action)

    return args
