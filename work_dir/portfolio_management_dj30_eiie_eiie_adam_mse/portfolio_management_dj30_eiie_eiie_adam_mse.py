data = dict(
    type='PortfolioManagementDataset',
    data_path='data/portfolio_management/dj30',
    train_path='data/portfolio_management/dj30/train.csv',
    valid_path='data/portfolio_management/dj30/valid.csv',
    test_path='data/portfolio_management/dj30/test.csv',
    tech_indicator_list=[
        'zopen', 'zhigh', 'zlow', 'zadjcp', 'zclose', 'zd_5', 'zd_10', 'zd_15',
        'zd_20', 'zd_25', 'zd_30'
    ],
    length_day=10,
    initial_amount=100000,
    transaction_cost_pct=0.001,
    test_dynamic_path=
    'data/portfolio_management/dj30/DJI_label_by_DJIindex_3_24_-0.25_0.25.csv',
    test_dynamic='-1')
environment = dict(type='PortfolioManagementEIIEEnvironment')
agent = dict(
    type='PortfolioManagementEIIE',
    memory_capacity=1000,
    gamma=0.99,
    policy_update_frequency=500)
trainer = dict(
    type='PortfolioManagementEIIETrainer',
    epochs=10,
    work_dir='work_dir/portfolio_management_dj30_eiie_eiie_adam_mse',
    if_remove=True)
loss = dict(type='MSELoss')
optimizer = dict(type='Adam', lr=0.001)
act_net = dict(
    type='EIIEConv',
    n_input=11,
    n_output=2,
    length=10,
    kernel_size=3,
    num_layer=1,
    n_hidden=32)
cri_net = dict(
    type='EIIECritic',
    n_input=11,
    n_output=1,
    length=10,
    kernel_size=3,
    num_layer=1,
    n_hidden=32)
task_name = 'portfolio_management'
dataset_name = 'dj30'
net_name = 'eiie'
agent_name = 'eiie'
optimizer_name = 'adam'
loss_name = 'mse'
work_dir = 'work_dir/portfolio_management_dj30_eiie_eiie_adam_mse'
