# Information Graphs: filtering and visualizing labeled networks via q-state Potts model geometry

In this paper, we propose a novel framework for the post-hoc analysis and visualization of labeled networks
based on the statistical mechanics of the q-state Potts model. Our method, called Information Graph, leverages
the estimation of the critical inverse temperature and information-geometric properties of the Potts model to 
identify the most and least informative nodes in a network. These estimates are used to weight the edges according 
to their contribution to intra and inter-community structure. By extracting the minimum and maximum information spanning
trees, we isolate structurally relevant edges that respectively reinforce community cohesion and inter-community bridging.
The union of these trees yields the Information Graph (IG), which offers a filtered representation of the original network 
by preserving semantically meaningful connections while removing redundancy.This process enhances network modularity and
provides an interpretable graph abstraction for downstream tasks. In practical terms, using a simple analogy with digital 
signal/image processing, Minimum Information Trees resemble a low-pass filtering process (smooth data) whereas Maximum 
Information Trees resemble a high-pass filtering process (emphasize edges and abrupt transitions). Potential applications 
include community-aware graph visualization, interpretation of classification outputs in relational data, simplification 
of biological and social networks, filtering of large-scale information graphs in machine learning pipelines and adaptive
sampling for supervised classification.

Author: Alexandre L. M. Levada

