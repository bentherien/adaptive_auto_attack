_base_ = ['./default_runtime_cifar10.py']
trades_loss='ST_linfty_compose_RT'
aaa_config='config/attack/cifar10/default_aaa_cifar10.py'
beta=3.0
teacher_file_path='bit_model_dir/bit.pth.tar'
teacher_model_name="BiT-M-R50x1"