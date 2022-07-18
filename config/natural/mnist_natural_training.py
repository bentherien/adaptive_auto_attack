_base_=['./default_runtime.py']
batch_size=2048
test_batch_size=2048
epochs=100
weight_decay=0
lr=0.01
momentum=0.9
epsilon=0.3
num_steps=40
step_size=0.01
beta=1.0
seed=1
log_interval=100
model_dir='./saved_models'
model_name='SmallCNN'
save_freq=5
dataset='MNIST'


trades_loss='natural'
aaa_config='config/attack/mnist/default_aaa_mnist.py'
pgd_config='config/attack/mnist/default_pgd_mnist.py'