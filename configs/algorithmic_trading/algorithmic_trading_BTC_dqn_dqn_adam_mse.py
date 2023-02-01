task_name = "algorithmic_trading"
dataset_name = "BTC"
optimizer_name = "adam"
loss_name = "mse"
net_name = "dqn"
agent_name = "dqn"
work_dir = f"work_dir/{task_name}_{dataset_name}_{net_name}_{agent_name}_{optimizer_name}_{loss_name}"

_base_ = [
    f"../_base_/datasets/{task_name}/{dataset_name}.py",
    f"../_base_/environments/{task_name}/env.py",
    f"../_base_/agents/{task_name}/{agent_name}.py",
    f"../_base_/trainers/{task_name}/trainer.py",
    f"../_base_/losses/{loss_name}.py",
    f"../_base_/optimizers/{optimizer_name}.py",
    f"../_base_/nets/{net_name}.py",
]


batch_size = 64
data = dict(
    type='AlgorithmicTradingDataset',
    data_path='data/algorithmic_trading/BTC',
    train_path='data/algorithmic_trading/BTC/train.csv',
    valid_path='data/algorithmic_trading/BTC/valid.csv',
    test_path='data/algorithmic_trading/BTC/test.csv',
    test_style_path=
    'data/algorithmic_trading/BTC/test_labeled_3_24_-0.15_0.15.csv',
    tech_indicator_list=[
        'high', 'low', 'open', 'close', 'adjcp', 'zopen', 'zhigh', 'zlow',
        'zadjcp', 'zclose', 'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25', 'zd_30'
    ],
    backward_num_day=5,
    forward_num_day=5,
    future_weights=0.2,
    initial_amount=100000,
    max_volume=1,
    transaction_cost_pct=0.001,
    test_style='-1')
environment = dict(type='AlgorithmicTradingEnvironment')
agent = dict(
    type='AlgorithmicTradingDQN',
    max_step=12345,
    reward_scale=1,
    repeat_times=1,
    gamma=0.9,
    batch_size=batch_size,
    clip_grad_norm=3.0,
    soft_update_tau=0,
    state_value_tau=0.005,
    memory_capacity=2000,
    epsilon=0.9,
    target_freq=50,
    future_loss_weights=0.2)
trainer = dict(
    type='AlgorithmicTradingTrainer',
    epochs=20,
    work_dir=work_dir,
    seeds_list=(12345, ),
    batch_size=batch_size,
    horizon_len=512,
    buffer_size=1000000.0,
    num_threads=8,
    if_remove=True,
    if_discrete=True,
    if_off_policy=True,
    if_keep_save=True,
    if_over_write=False,
    if_save_buffer=False,
    eval_times=3)
loss = dict(type='MSELoss')
optimizer = dict(type='Adam', lr=0.001)
act = dict(
    type='QNet', state_dim=82, action_dim=3, dims=(64, 32), explore_rate=0.25)
cri = None