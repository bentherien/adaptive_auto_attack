_base_=['./default_runtime.py']
batch_size=1024
test_batch_size=1024
epochs=76
weight_decay=2e-4
lr=0.1
momentum=0.9
epsilon=0.031
num_steps=10
step_size=0.007
beta=6.0
seed=1
log_interval=100
model_dir='./saved_models'
model_name='WRN-34-10'
save_freq=1
dataset='cifar10'


trades_loss='natural'
aaa_config='config/attack/cifar10/default_aaa_cifar10.py'
pgd_config='config/attack/cifar10/default_pgd_cifar10.py'