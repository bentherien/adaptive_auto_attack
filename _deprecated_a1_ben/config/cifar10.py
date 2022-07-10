model_name = 'TRADES'
dataset='cifar10'
# ep = 0.031
ep = 0.03099996
batch_size = 256
random=True
average_number=2000
device=None
kwargs=dict(
    out_restart_num = 30,
    normal_prob = 0.5,
    out_re_rule = False,
    max_iter0=200,
    warm_restart=False,
    warm_restart_num=1
)