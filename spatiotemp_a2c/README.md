# A2C Spatiotemporal Prediction

## Description
This is an A2C implementation in which the policy attempts to predict the next state of the environment in addition to the action distribution. This idea has been attempted in [3D conv nets](https://arxiv.org/pdf/1412.0767.pdf), [RNNs](https://papers.nips.cc/paper/6689-predrnn-recurrent-neural-networks-for-predictive-learning-using-spatiotemporal-lstms), and likely elsewhere, although to the best of my knowledge it has not been attempted in the domain of reinforcement learning yet. 

Due to limited computational resources, the policy as written predicts the feature vector at a hidden layer in the policy of the next state. After a single comparative trial, the results appeared to show little improvement over the standard A2C algorithm. The idea certainly needs more exploration, however.
