# Synchronous Stochastic Gradient Descent using TensorFlow
This tensorflow python program runs Logistic Regression Stochastic Gradient Descent Algorithm on the input dataset that is spread across 5 VMs in a synchronous manner.

## Dataset
The dataset is the one used for the [Kaggle Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge) sponsored by Criteo Labs. It is available [here](http://pages.cs.wisc.edu/~ashenoy/CS838/).

## Methodology
* A local_gradient is computed on every device and then aggregated in a centralized device which in our case is VM-4-1.
* The local_gradient computation happens in parallel in all 5 VMs on every iteration. The aggregation is done on VM-4-1 at the end of each iteration.
* Since each iteration waits for the local_gradients to be computed on all 5 VMs before proceeding, this is a synchronous process.
* We used samples in tfrecords22 for testing our trained gradient.
* An error_rate is calculated at the end of every 1000 iterations.

## Environment
A 5 node cluster, each node with 20GB RAM and 4 cores was used to run this application.
