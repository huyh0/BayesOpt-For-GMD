# Bayesian Optimization for Efficient Molecular Generation with Variational Autoencoders

Huey Hoang

## Project Description

This paper aims to introduces a data-driven workflow and algorithm
that combines a Junction Tree Variational Autoencoder with Bayesian
optimization to generate molecules that maximize the inhibition of 3CL-
protease enzyme created by the Coronavirus. To drive the Bayesian
optimization, a property predictor model from DeepPurpose Library is used.
The dataset used is also provided by DeepPurpose containing the drugs and
the percentage of inhibition. The Junction Tree Variational Autoencoder
is used to encode the molecules and generate the new molecules found
by the Bayesian Optimization. The results found show that generating
molecules with activity is achievable, however, further tuning is required to
generate molecules that will completely inhibit the 3CL-protease enzymes.
