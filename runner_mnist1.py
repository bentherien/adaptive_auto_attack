import sys
import os


             

for file_ in ["L_infty_union_RT_1_trades_mnist.py","RT_1_trades_mnist.py"]:
    command = "{} train_trades-MODIF.py -c config/defenses/mnist/{} -dn 3".format(sys.executable,file_)
    print("EXECUTING:",command)
    os.system(command)