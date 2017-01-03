#!/bin/bash

export TF_LOG_DIR="/home/ubuntu/tf/logs"
source tfdefs.sh

# startserver.py has the specifications for the cluster.
terminate_cluster
start_cluster startserver.py

echo "Executing the distributed tensorflow job from exampleSynchronousUpdate.py"
# testdistributed.py is a client that can run jobs on the cluster.
# please read testdistributed.py to understand the steps defining a Graph and
# launch a session to run the Graph

for i in 100
do
START=$(date +%s)
python exampleSynchronousUpdate.py

END=$(date +%s)
DIFF=$(( $END - $START ))
echo "Program run time : $DIFF"
done
# defined in tfdefs.sh to terminate the cluster
tensorboard --logdir=$TF_LOG_DIR
terminate_cluster
