data = dict(
    type='OrderExecutionDataset',
    data_path='data/order_execution/PD_BTC',
    train_path='data/order_execution/PD_BTC/train.csv',
    valid_path='data/order_execution/PD_BTC/valid.csv',
    test_path='data/order_execution/PD_BTC/test.csv',
    tech_indicator_list=[
        'high', 'low', 'open', 'close', 'adjcp', 'zopen', 'zhigh', 'zlow',
        'zadjcp', 'zclose', 'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25', 'zd_30'
    ],
    length_keeping=30,
    state_length=10,
    target_order=1,
    initial_amount=100000,
    test_dynamic_path=
    'data/order_execution/PD_BTC/test_labeled_3_24_-0.15_0.15.csv',
    test_dynamic='-1')
environment = dict(type='OrderExecutionPDEnvironment')
agent = dict(
    type='OrderExecutionPD',
    memory_capacity=100,
    gamma=0.9,
    climp=0.2,
    sample_effiency=0.5,
    memory_update_freq=10)
trainer = dict(
    type='OrderExecutionPDTrainer',
    epochs=10,
    work_dir='work_dir/order_execution_BTC_pd_pd_adam_mse',
    if_remove=True)
loss = dict(type='MSELoss')
optimizer = dict(type='Adam', lr=0.001)
net = dict(type='PDNet', input_feature=16, hidden_size=32, private_feature=2)
task_name = 'order_execution'
dataset_name = 'BTC'
net_name = 'pd'
agent_name = 'pd'
optimizer_name = 'adam'
loss_name = 'mse'
work_dir = 'work_dir/order_execution_BTC_pd_pd_adam_mse'
