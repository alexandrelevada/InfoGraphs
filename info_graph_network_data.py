"""

Information Graphs: filtering and visualizing labeled networks via q-state Potts model geometry

In this paper, we propose a novel framework for the post-hoc analysis and visualization of labeled networks
based on the statistical mechanics of the q-state Potts model. Our method, called Information Graph, leverages
the estimation of the critical inverse temperature and information-geometric properties of the Potts model to 
identify the most and least informative nodes in a network. These estimates are used to weight the edges 
according to their contribution to intra and inter-community structure. By extracting the minimum and maximum 
information spanning trees, we isolate structurally relevant edges that respectively reinforce community cohesion 
and inter-community bridging. The union of these trees yields the Information Graph (IG), which offers a filtered 
representation of the original network by preserving semantically meaningful connections while removing redundancy.
This process enhances network modularity and provides an interpretable graph abstraction for downstream tasks. 
In practical terms, using a simple analogy with digital signal/image processing, Minimum Information Trees resemble 
a low-pass filtering process (smooth data) whereas Maximum Information Trees resemble a high-pass filtering process 
(emphasize edges and abrupt transitions). Potential applications include community-aware graph visualization, 
interpretation of classification outputs in relational data, simplification of biological and social networks, 
filtering of large-scale information graphs in machine learning pipelines and adaptive sampling for supervised classification.

Author: Alexandre L. M. Levada


"""
import time
import types
import warnings
import matplotlib as mpl
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
import sklearn.neighbors as sknn
import sklearn.datasets as skdata
import sklearn.utils.graph as sksp
from numpy import inf
from scipy import optimize
from networkx.convert_matrix import from_numpy_array
from scipy.io import mmread
from matplotlib.lines import Line2D

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

# Build the KNN graph
def build_KNN_Graph(dados, k):
    # Build a graph
    CompleteGraph = sknn.kneighbors_graph(dados, n_neighbors=n-1, mode='distance')
    # Adjacency matrix
    W_K = CompleteGraph.toarray()
    # NetworkX format
    K_n = nx.from_numpy_array(W_K)
    # MST
    W_mst = nx.minimum_spanning_tree(K_n)
    mst = [(u, v, d) for (u, v, d) in W_mst.edges(data=True)]
    mst_edges = []
    for edge in mst:
        edge_tuple = (edge[0], edge[1], edge[2]['weight'])
        mst_edges.append(edge_tuple)
    # Create the k-NNG
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=k, mode='distance')
    # Adjacency matrix
    W = knnGraph.toarray()
    # NetworkX format
    G = nx.from_numpy_array(W)
    # To assure the k-NNG is connected we add te MST edges
    G.add_weighted_edges_from(mst_edges)
    return G

# Optional function to normalize the curvatures to the interval [0, 1]
def normalize_curvatures(curv):
    if curv.max() != curv.min():
        k = 0.001 + (curv - curv.min())/(curv.max() - curv.min())
    else:
        k = curv
    return k

# Plot the KNN graph
def plot_network(G, target, layout, pos=0, leg=False):
    color_map = []
    node_sizes = 25*np.ones(G.number_of_nodes())
    #node_shapes = G.number_of_nodes()*['o']
    for i in range(G.number_of_nodes()):
        if target[i] == -9:
    	    color_map.append('darkcyan')
    	    node_sizes[i] *= 2
    	    #node_shapes[i] = 's'
        if target[i] == -1:
            color_map.append('black')
            node_sizes[i] *= 2
            #node_shapes[i] = 's'
        elif target[i] == 0:
        	color_map.append('blue')
        elif target[i] == 1:
           	color_map.append('red')
        elif target[i] == 2:
           	color_map.append('green')
        elif target[i] == 3:
           	color_map.append('purple')
        elif target[i] == 4:
           	color_map.append('orange')
        elif target[i] == 5:
           	color_map.append('magenta')
        elif target[i] == 6:
           	color_map.append('darkkhaki')
        elif target[i] == 7:
           	color_map.append('brown')
        elif target[i] == 8:
           	color_map.append('salmon')
        elif target[i] == 9:
           	color_map.append('cyan')        	    
    plt.figure(1)
	# Há vários layouts, mas spring é um dos mais bonitos
    if pos == 0:
    	if layout == 'spring':
	    	pos = nx.spring_layout(G, iterations=50)
    	else:
	    	pos = nx.kamada_kawai_layout(G) # ideal para plotar a árvore!
    nx.draw_networkx(G, pos, node_size=node_sizes, node_color=color_map, with_labels=False, width=0.25, alpha=0.4)

    if leg == True: 
    	legend_elements = [
        	Line2D([0], [0], marker='o', color='w', label='L-nodes', markerfacecolor='black', markersize=10),
        	Line2D([0], [0], marker='o', color='w', label='H-Nodes',  markerfacecolor='darkcyan', markersize=10)
    	]
    	plt.legend(handles=legend_elements, loc='upper right')

    plt.show()
    return pos

# Compute the first and second order Fisher local information
def FisherInformation(A, beta):
	n = A.shape[0]
	# Computes the number of labels (states of the Potts model)
	c = len(np.unique(target))
	PHIs = np.zeros(n)
	PSIs = np.zeros(n)
	for i in range(n):
		neighbors = A[i, :]
		indices = neighbors.nonzero()[0]
		labels = target[indices]
		uim = np.count_nonzero(labels==target[i])
		Uis = np.zeros(c)
		vi =  np.zeros(c)
		wi = np.zeros(c)
		Ai = np.zeros((c, c))
		Bi = np.zeros((c, c))
		# Build vectors vi and wi
		for k in range(c):
			Uis[k] = np.count_nonzero(labels==k)
			vi[k] = uim - Uis[k]
			wi[k] = np.exp(beta*Uis[k])
		# Build matrix A
		for k in range(c):
			Ai[:, k] = Uis
		# Build matrix B
		for k in range(c):
			for l in range(c):
				Bi[k, l] = Uis[k] - Uis[l]  
		# Compute the first and second order Fisher information
		PHIs[i] = np.sum( np.kron((vi*wi), (vi*wi).T) ) / np.sum( np.kron(wi, wi.T) )
		Li = Ai*Bi
		Mi = np.reshape(np.kron(wi, wi.T), (c, c))
		PSIs[i] = np.sum( Li*Mi ) / np.sum( np.kron(wi, wi.T) )
	return (PHIs, PSIs)

# Compute the information graph
def InformationGraph(A, target, K, alpha=1):
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            if A[i,j] > 0:
                if target[i] == target[j]:
                    A[i,j] = alpha*(K[i] + K[j])
                else:
                    A[i,j] = K[i] + K[j]
    return A

# Generate the minimum information tree
def MinimumInformationTree(H):
    # Networkx graph
    G = nx.from_numpy_array(H)
    # MST
    T = nx.minimum_spanning_tree(G)
    return T

# Computes the mean field approximation to critical beta
def mean_field_critical_beta(G, communities):
	# Get the degrees of all nodes
	degrees = dict(G.degree())
	# Calculate the sum of degrees
	sum_of_degrees = sum(degrees.values())
	# Calculate the sum of degrees squared
	sum_of_degrees_squared = sum(np.array(list(degrees.values()))**2)
	# Get the number of nodes
	num_nodes = G.number_of_nodes()
	# Calculate the mean degree
	d = sum_of_degrees/num_nodes
	d_2 = sum_of_degrees_squared/num_nodes
	# Computes the approximation
	beta_c = (d/(d_2 - d))*np.log(1 + np.sqrt(len(communities)))
	return beta_c

# Computes the spectral approximation to critical beta
def spectral_critical_beta(A):
	lambdas, V = np.linalg.eig(A)
	beta_c = 1/lambdas.real.max()
	return beta_c

##############################################
############# Beginning of the script
##############################################

# Network data 
name = './karate/karate.mtx'
#name = './dolphins/dolphins.mtx'
#name = './polbooks/polbooks.mtx'
#name = './lesmis/lesmis.mtx'
#name = './email/email.mtx'
#name = './USAir97/USAir97.mtx'
#name = './web-polblogs/web-polblogs.mtx'
#name = './bio-celegans/bio-celegans.mtx'
#name = './jazz/jazz.mtx'
#name = './celegansneural/celegansneural.mtx'
#name = './delaunay_n10/delaunay_n10.mtx'
#name = './football/football.mtx'

# Read network
SM = mmread(name) 	# Sparse matrix
G = nx.Graph(SM)

# Choose layout
pos = nx.spring_layout(G, iterations=500)

# Plot network
plt.figure(1)
nx.draw_networkx(G, pos, node_size=25, node_color='black', with_labels=False, width=0.25, alpha=0.4)
plt.show()

# Find communities
if 'karate' in name or 'dolphins' in name or 'polbooks' in name: 
	communities = nx.community.girvan_newman(G)	
elif 'football' in name:
	communities = nx.community.louvain_communities(G, seed=123)
else:
	communities = nx.community.greedy_modularity_communities(G)		# Clauset-Newman-Moore

if isinstance(communities, types.GeneratorType):
	communities = tuple(sorted(c) for c in next(communities))

# Extract the labels
n = G.number_of_nodes()
target = np.zeros(n)
for i, c in enumerate(communities):
	nc = len(c)
	lc = list(c)
	for j in range(nc):
		target[int(lc[j])] = i

# Save the original labels
original_target = target.copy()

print('\nMetrics for the orignal network')
print('--------------------------------')
print()
print('Number of nodes: ', G.number_of_nodes())
print('Number of edges: ', G.number_of_edges())
print('Average degree: ', sum([nx.degree(G)[i] for i in range(G.number_of_nodes())])/G.number_of_nodes())
print('Number of triangles: ', sum(nx.triangles(G).values()))
print('Modularity: ', nx.community.modularity(G, communities))
coverage, performance = nx.community.partition_quality(G, communities)
print('Coverage: ', coverage)
print('Assortativity: ', nx.degree_assortativity_coefficient(G))
print('Average betweeness centrality: ', sum(nx.betweenness_centrality(G).values())/G.number_of_nodes())
print()

# Plot network communities
pos = plot_network(G, target, 'spring', pos)

# Converto to adjacency matrix
A = nx.to_numpy_array(G)

# Critical beta
beta = mean_field_critical_beta(G, communities)
print('Critical beta (mean field): ', beta)
beta_c = spectral_critical_beta(A)
print('Critical beta (spectral): ', beta_c)
print()

#######################################
# Low and High information nodes
#######################################
# Compute the first and second order local Fisher information 
PHI, PSI = FisherInformation(A, beta)
# Approximate the local curvatures (add small value to avoid division by zero)
curvatures = -PSI/(PHI+0.001)
# Normalize curvatures
K = normalize_curvatures(curvatures)
# Order by information
smallest = K.argsort()
# Flag the the smallest with -1
target[smallest[:30]] = -1
# Invert the order
largest = smallest[::-1]
# Flag the the largest with -9
target[largest[:50]] = -9
# Plot network communities and high information nodes
pos = plot_network(G, target, 'spring', pos, leg=True)

#########################################
# Minimum Information Tree
#########################################
alpha = (np.sqrt(5) - 1)/2
#alpha = 1
IG = InformationGraph(A, original_target, K, alpha)
# Minimum information tree
MinT = MinimumInformationTree(IG)
# Plot minimum information tree
plot_network(MinT, original_target, 'spring')

#########################################
# Maximum Information Tree
#########################################
# Approximate the local curvatures (add small value to avoid division by zero)
curvatures = PSI/(PHI+0.001)
# Normalize curvatures
K = normalize_curvatures(curvatures)
# Information graph
alpha = (np.sqrt(5) - 1)/2
#alpha = 1
IG = InformationGraph(A, original_target, K, alpha)
# Minimum information tree
MaxT = MinimumInformationTree(IG)
# Plot minimum information tree
plot_network(MaxT, original_target, 'spring')

#########################################
# Information Graph
#########################################
# Information graph = composition of MinT and MaxT
R = nx.compose(MinT, MaxT)
# Plot minimum information tree
pos = plot_network(R, original_target, 'spring', pos)

print('Metrics for the information graph')
print('----------------------------------')
print()
print('Number of nodes: ', G.number_of_nodes())
print('Number of edges: ', R.number_of_edges())
print('Average degree: ', sum([nx.degree(R)[i] for i in range(R.number_of_nodes())])/R.number_of_nodes())
print('Number of triangles: ', sum(nx.triangles(R).values()))
print('Modularity: ', nx.community.modularity(R, communities))
coverage, performance = nx.community.partition_quality(R, communities)
print('Coverage: ', coverage)
print('Assortativity: ', nx.degree_assortativity_coefficient(R))
print('Average betweeness centrality: ', sum(nx.betweenness_centrality(R).values())/R.number_of_nodes())
print()

print('=> Edge compression rate: ', (G.number_of_edges() - R.number_of_edges())/G.number_of_edges())
