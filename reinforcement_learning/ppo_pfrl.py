
import argparse
import functools 

import gym
import gym.spaces
import numpy as np
import torch
from torch import nn

import pfrl
from pfrl.agents import PPO
from pfrl import experiments
from pfrl import utils

import environments


def main():
    '''
    ArgumentParserの設定
    '''
    import logging
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str,
                        default='LidarBat-v0', help='Bat simulation env')
    parser.add_argument('--arch', type=str, default='FFGaussian',
                        choices=('FFSoftmax', 'FFMellowmax',
                                 'FFGaussian'))
    parser.add_argument('--bound-mean', action='store_true')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--outdir', type=str, default='data/ppo',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--steps', type=int, default=10 ** 6)
    parser.add_argument('--eval-interval', type=int, default=10000) #1万回ずつエージェントを評価する
    parser.add_argument('--eval-n-runs', type=int, default=10) #各評価で10回をサンプリング？
    parser.add_argument('--reward-scale-factor', type=float, default=1e-2) #報酬の大きさを決める要素...
    parser.add_argument('--standardize-advantages', action='store_true') #標準報酬...？
    parser.add_argument('--render', action='store_true', default=False) #出力=>図示
    parser.add_argument('--lr', type=float, default=3e-4) #lr=learning rate?
    parser.add_argument('--weight-decay', type=float, default=0.0) #過剰適合のリスクを減らすための重み減衰？？？
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument("--load-pretrained",
                        action="store_true", default=False) #以前の学習をロード
    parser.add_argument('--logger-level', type=int, default=logging.INFO) #デバッグ確認
    parser.add_argument('--monitor', action='store_true') #モニター...
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1000,
        help="Interval in timesteps between outputting log messages during training",
    ) #学習中にログを出力するまでの間隔
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of envs run in parallel."
    ) #並行して実行される環境の数
    parser.add_argument("--batch-size", type=int,
                        default=64, help="Minibatch size") #ミニバッチ学習？取り出すデータの数、学習が停滞しにくくなる
    parser.add_argument('--update-interval', type=int, default=2048) #↑でパラメータを更新する間隔？
    parser.add_argument('--batchsize', type=int, default=64) #--batch-sizeとの違い...
    parser.add_argument('--epochs', type=int, default=10) #エポック数：１つの訓練データを何回繰り返して学習させるか
    parser.add_argument('--entropy-coef', type=float, default=0.0) #エントロピー係数：損失関数に加える→探索が行われなくなるのを防ぐ
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args() #引数を解析

    logging.basicConfig(level=args.logger_level)
    # Set a random seed used in PFRL
    utils.set_random_seed(args.seed)
    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32 #条件をテスト

    args.outdir = experiments.prepare_output_dir(args, args.outdir)

    def make_env(process_idx, test):
        '''
        ArgumentParser(--env)で指定したlidar_batの呼び出し
        '''
        env = gym.make(args.env) 
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[process_idx]) #探索プロセス（関数の呼び出し時に変更可能？）
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = pfrl.wrappers.CastObservationToFloat32(env)
        if args.monitor:
            env = pfrl.wrappers.Monitor(env, args.outdir)
        # TODO
        # if not test is not here
        if args.render:
            env = pfrl.wrappers.Render(env)
        return env

    def make_batch_env(test): #mini-batch学習の環境
        '''
        mini-batch学習の環境設定
        '''
        return pfrl.envs.MultiprocessVectorEnv(
            [
                functools.partial(make_env, idx, test)
                for idx, env in enumerate(range(args.num_envs))
            ]
        )

    # Only for getting timesteps, and obs-action spaces
    sample_env = gym.make(args.env) #サンプル...？
    timestep_limit = sample_env.spec.max_episode_steps #ステップ数
    obs_space = sample_env.observation_space #状態空間？
    action_space = sample_env.action_space #行動の数？
    print("Observation space:", obs_space) #表示
    print("Action space:", action_space) #表示

    assert isinstance(action_space, gym.spaces.Box) #型の判定true/false isinstance(object,class)

    # Normalize observations based on their empirical mean and variance
    obs_normalizer = pfrl.nn.EmpiricalNormalization(
        obs_space.low.size, clip_threshold=5
    ) #正規化？

    # pulicy here magic number must be concidered again
    obs_size = obs_space.low.size #状態における最小値
    action_size = action_space.low.size #行動の最小値
    policy = torch.nn.Sequential(
        nn.Linear(obs_size, 128),
        nn.Tanh(),
        nn.Linear(128, 64),
        nn.Tanh(),
        nn.Linear(64, action_size),
        pfrl.policies.GaussianHeadWithStateIndependentCovariance(
            action_size=action_size,
            var_type="diagonal",
            var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
            var_param_init=0,  # log std = 0 => std = 1
        ),
    ) 

    vf = torch.nn.Sequential(
        nn.Linear(obs_size, 128),
        nn.Tanh(),
        nn.Linear(128, 64),
        nn.Tanh(),
        nn.Linear(64, 1),
    )

    # While the original paper initialized weights by normal distribution,
    # we use orthogonal initialization as the latest openai/baselines does.
    def ortho_init(layer, gain):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)

    ortho_init(policy[0], gain=1)
    ortho_init(policy[2], gain=1)
    ortho_init(policy[4], gain=1e-2)
    ortho_init(vf[0], gain=1)
    ortho_init(vf[2], gain=1)
    ortho_init(vf[4], gain=1)

    # Combine a policy and a value function into a single model
    model = pfrl.nn.Branched(policy, vf)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    agent = PPO(
        model,
        opt,
        obs_normalizer=obs_normalizer,
        # gpu=args.gpu,
        update_interval=args.update_interval,
        minibatch_size=args.batch_size,
        epochs=args.epochs,
        clip_eps_vf=None,
        entropy_coef=args.entropy_coef,
        standardize_advantages=True,
        gamma=0.995,
        lambd=0.97,
    )

    if args.load or args.load_pretrained:
        if args.load_pretrained:
            raise Exception("Pretrained models are currently unsupported.")
        # either load or load_pretrained must be false
        assert not args.load or not args.load_pretrained
        if args.load:
            agent.load(args.load)
        else:
            agent.load(utils.download_model(
                "PPO", args.env, model_type="final")[0])

    if args.demo:
        env = make_batch_env(True)
        eval_stats = experiments.eval_performance(
            env=env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
            max_episode_len=timestep_limit,
        )
        print(
            "n_runs: {} mean: {} median: {} stdev {}".format(
                args.eval_n_runs,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )

    else:
        experiments.train_agent_batch_with_evaluation(
            agent=agent,
            env=make_batch_env(False),
            eval_env=make_batch_env(True),
            outdir=args.outdir,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
            max_episode_len=timestep_limit,
            save_best_so_far_agent=False,
        )


if __name__ == '__main__':
    main()
