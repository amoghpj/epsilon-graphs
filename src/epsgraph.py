# [[id:fef71c5d-f6f2-415f-9a00-a352595a2d29][Main:1]]
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import sys
import networkx as nx
from tqdm import tqdm
import os
import umap
import pandas as pd
import logging
import numpy as np
from itertools import product, combinations
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',filename="run.log")
MINEXP = 0
MAXEXP = 50
NUMGENES = 1000
NUMCELLS = 100
NUMSTEPS = 1000
NUMPROG = 4
BRIDGEFRACTION = 0.5
NUMBRIDGE = int(BRIDGEFRACTION*NUMCELLS)
usuffix = f"-{NUMGENES}G-{NUMCELLS}C-{NUMPROG}L-{NUMSTEPS}S-{NUMBRIDGE}B"
fname_cells = f"data/simulated-scrnaseq{usuffix}.csv"
fname_dist = f"data/scrnaseq-dist{usuffix}.csv"
fname_epsilon = f"data/epsilon{usuffix}.csv"
fname_epsilon_viz = f"img/scrnaseq-eps-gcc{usuffix}.png"
logging.info("-------------------------------------------")
logging.info("!!Starting run!!")
def bfs_from_start(matrix, start, eps, V):
    n = len(matrix[:,0])
    visited = np.array([0 for i in range(n)])
    Q = [start]
    visited[Q[0]] = 1
    while len(Q) > 0 or (sum(visited)!=n): 
        for j in range(n):

            if (matrix[start, j] < eps) and\
               (V[j] == 0) and\
               (j not in Q) and\
               (visited[j] == 0):
                
                Q.append(j)
                #visited[j] = 1
        Q = Q[1:]
        if len(Q) > 1:
            start = Q[0]
            visited[Q[0]] = 1
        else:
            return(visited)
    return visited

def bfs_we_think(matrix, eps):
    n = len(matrix[:,0])
    V = np.array([0 for i in range(n)])
    start = 1
    sizes = []
    counter = 0
    while sum(V) < n: 
        visited = bfs_from_start(matrix, start, eps, V)
        sizes.append(sum(visited))
        V = V + visited
        remaining = 1-V
        for i in range(n):
            if remaining[i] == 1:
                start = i
                continue
        counter += 1
    return(max(sizes))
# def graph_initial(x,n):
#     # n is the number of nodes
#     # x is the matrix of your RNA thingies or whatever, x[i] should refer to the i-th sequence
#     return np.array([[np.linalg.norm(x[i]-x[j]) for i in range(n)] for j in range(n)])

# def bfs(matrix, row, col, visited):
#     nodes = [(row, col)]
#     while nodes:
#         row, col = nodes.pop(0)
#         # the below conditional ensures that our algorithm 
#         #stays within the bounds of our matrix.
#         if row >= len(matrix) or col >= len(matrix[0]) or row < 0 or col < 0:
#             continue
#         if (row, col) not in visited:
#             #This condition is what allows you to take a step only if the pair is at distance at most epsilon.
#             if matrix[row][col] < eps:
#                 visited.append((row, col))
#                 nodes.append((row+1, col))
#                 nodes.append((row, col+1))
#                 nodes.append((row-1, col))
#                 nodes.append((row, col-1))

# # use this
# def bfs_wrapper(matrix):
#     visited = []
#     for i in range(len(matrix)):
#         for j in range(len(matrix[0])):
#             if (i,j) not in visited:
#                 bfs(matrix, i, j, visited)
            
#     return visited

def flip_coin():
    if np.random.random() > 0.5:
        return True
    else:
        return False
def do_random_walk(progenitor, numsteps):
    cell = np.array(progenitor)
    for _ in range(numsteps):
        cell = [gene + 1 if flip_coin() else max(0, gene -1)
                for gene in cell]
    return cell
def get_progenitor(numgenes, MINEXP=0, MAXEXP=500):
    return(np.random.randint(MINEXP,MAXEXP,size=numgenes))

def get_population(progenitor, progenitorid, population_size, numsteps=10000) :
    """
    :returns: pandas DataFrame
    """
    population = []
    for _ in tqdm(range(population_size)):
        population.append(do_random_walk(progenitor, numsteps=numsteps))
    dataset = pd.DataFrame(np.array(population), 
                           columns=[f"gene{i}" for i in range(len(progenitor))],
                           index=pd.Index(list(range(population_size))))
    dataset = dataset.assign(clusterid=progenitorid , 
                             incluster=True)
    return(dataset)

def euclidean(v1, v2):
    return(np.sqrt(sum([(_v1 -_v2)**2 for _v1, _v2 in zip(v1,v2)] )))

def calculate_allvall_distances(df):
    distdict = []
    dat=df[[c for c in df.columns if "gene" in c]]
    for v1, v2 in tqdm(combinations(range(dat.shape[0]),2 )):
        d = euclidean(dat.iloc[v1],dat.iloc[v2])
        distdict.append({"tail":v1, "head":v2,"d":d})
    return(distdict)

def compute_eps_networks(df):
    """
    Create a graph from the distance matrix.
    Compute the ego graph at each radius for each node, 
    and store the size of the largest ego subgraph
    at a given epsilon
    """
    logging.info("Starting epsilon search...")
    radii = np.linspace(df["d"].min(), 
                        df["d"].max(), 
                        100)
    graphs = {}
    eps_search = []
    G = nx.from_pandas_edgelist(df, source="tail",target="head",edge_attr="d")
    matrix = nx.adjacency_matrix(G, weight="d").todense()
    for eps in tqdm(radii):
        s = bfs_we_think(matrix, eps)
        eps_search.append({"epsilon": eps, "size_gc":s})

    ## Remember, G is a complete graph by construction, so just finding the 
    ## ego graph around each 
    # allnodes = list(G.nodes())[:100]
    # print(allnodes)
    # for e in tqdm(radii):
    #     sizes = []
    #     isseen = {n:False for i,n in enumerate(allnodes)}
    #     print(e)
    #     for i,n in enumerate(allnodes):
    #         edges = [e for e in G[n].edges if e["weight"] < eps]
    #         if not isseen[n]:
    #             nodes = nx.bfs_tree(G, n, depth_limit = e).nodes()
    #             sizes.append(len(nodes))
    #             for node in nodes:
    #                 isseen[node] = True
    #         isseen[n] = True
    #         # sizes.append(len(nx.ego_graph(G, n, radius=e, 
    #         #                 distance = "d").nodes))
    #     print(sizes)
    #     sys.exit()
    #     eps_search.append({"epsilon": e, "size_gc":max(sizes)})
    epssearch = pd.DataFrame(eps_search)
    epssearch.to_csv(fname_epsilon)

def brownian_bridge(endpoints, numbridge, numgenes):
    allbridgecells = []
    for (p1idx, p1), (p2idx, p2) in combinations(zip(list(range(len(endpoints))),endpoints),2):
        bridge_population = []
        for _ in range(numbridge):
            F = np.random.random()# np.random.random(size=numgenes)
            ## create a point separated by fraction F from each of the two progenitors
            ## Do a small random walk from that position to mix things up a little, currently 10 steps
            bridge_population.append(do_random_walk(np.array(p1)*F  + np.array(p2)*(1-F),
                                                    10))
        bridgedataset = pd.DataFrame(np.array(bridge_population), 
                               columns=[f"gene{i}" for i in range(numgenes)])
        bridgedataset = bridgedataset.assign(clusterid=f"{p1idx}-{p2idx}",
                                             incluster=False)
        allbridgecells.append(bridgedataset)
    return(pd.concat(allbridgecells).reset_index(drop=True))

def even_bridge(endpoints, numbridge, numgenes):
    allbridgecells = []
    for (p1idx, p1), (p2idx, p2) in combinations(zip(list(range(len(endpoints))),endpoints),2):
        bridge_population = []
        for n in range(numbridge):
            ## create a point separated by fraction F from each of the two progenitors
            ## Do a small random walk from that position to mix things up a little, currently 10 steps
            F = float(n)/float(numbridge)
            # bridge_population.append(do_random_walk(np.array(p1)*F  + np.array(p2)*(1-F),
            #                                         10))
            bridge_population.append(np.array(p1)*F  + np.array(p2)*(1-F))
        logging.info(f"Distance between adjacent points: {euclidean(bridge_population[0],bridge_population[1])}")
        bridgedataset = pd.DataFrame(np.array(bridge_population), 
                               columns=[f"gene{i}" for i in range(numgenes)])
        bridgedataset = bridgedataset.assign(clusterid=f"{p1idx}-{p2idx}",
                                             incluster=False)
        allbridgecells.append(bridgedataset)
    return(pd.concat(allbridgecells).reset_index(drop=True))

if not os.path.exists(fname_cells):
    logging.info("Generating dataset")
    progenitorlist = [get_progenitor(NUMGENES) for _ in range(NUMPROG)]
    dflist = [get_population(progn, i, NUMCELLS, numsteps = NUMSTEPS)
              for i, progn in enumerate(progenitorlist)]
    #dflist.append(brownian_bridge(progenitorlist, NUMBRIDGE, NUMGENES))
    logging.info(f"Distance between cluster centers: {euclidean(progenitorlist[0],progenitorlist[1])}")
    dflist.append(even_bridge(progenitorlist, NUMBRIDGE, NUMGENES))
    df = pd.concat(dflist)
    df.to_csv(fname_cells,index=False)
df = pd.read_csv(fname_cells)

if not os.path.exists(fname_dist):
    logging.info("Starting all-v-all distance computation")
    distdf = pd.DataFrame(calculate_allvall_distances(df))
    distdf.to_csv(fname_dist,
                  index=False)
distdf = pd.read_csv(fname_dist)
distdf = distdf.sort_values(by=["tail","d"])

plot_umap = True
plot_epsilon = True
do_epsgraphs = True
if do_epsgraphs and not os.path.exists(fname_epsilon):
    logging.info("Computing graphs of neighborhood size epsilon")
    eps = compute_eps_networks(distdf)

# if viz_eps_emergence:
#     logging.info("Visualizing graphs at radius eps")
   
#     numcols = int(np.ceil(np.sqrt(len(graphs))))
#     figdim = 4
#     fig = plt.figure(figsize=(numcols*figdim,
#                               numcols*figdim))


#     for i, (e,graph) in enumerate(graphs.items()):
#         ax = fig.add_subplot(numcols, int(len(graphs)/numcols)+1, i+1)
#         G = nx.Graph(graph)
#         cc = list(nx.connected_components(G))
#         if len(cc) > 0:
#             size_gcc.append(len(G.subgraph(sorted(cc,\
#                                               key=len,\
#                                                   reverse=True)[0]).nodes))
#         else:
#             size_gcc.append(0)
#         eps.append(e)
#         nx.draw_spring(G, 
#                 ax=ax)
#         ax.set_title(round(e,1))
#     plt.tight_layout()
#     plt.savefig("img/scranseq-eps-graphs.png")
#     plt.close("all")

if plot_epsilon:
    epssearch = pd.read_csv(fname_epsilon)
    plt.plot(epssearch.epsilon, epssearch.size_gc,"k-",lw=4)
    plt.xlabel("$\\epsilon$")
    plt.title(f"#cells={NUMCELLS} | #genes={NUMGENES}")
    plt.ylabel("Size GCC")
    plt.savefig(fname_epsilon_viz)
    plt.close("all")                
if plot_umap:
    logging.info("Umap...")
    red = umap.UMAP()
    cols = [c for c in df.columns if 'gene' in c]
    embed =  red.fit_transform(StandardScaler().fit_transform(df[cols].values))
    df = df.assign(umap1 = embed[:,0],
                   umap2 = embed[:,1])
    g = sns.scatterplot(data=df, x="umap1",
                    y="umap2",
                    hue="clusterid")
    g.legend(fancybox=False, framealpha=0.)
    plt.savefig(f"img/scrnaseq-umap{usuffix}.png")
    plt.close()
logging.info("Finished all tasks!")
# Main:1 ends here
