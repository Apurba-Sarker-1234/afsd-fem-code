import sys
sys.path.insert(0, '.')
from gamma_afsd import domain_mgr
import numpy as np
domain = domain_mgr(filename='cylinder.k')
nodes = domain.nodes.get() if hasattr(domain.nodes, 'get') else np.asarray(domain.nodes)
print('FEM node X range:', nodes[:,0].min(), 'to', nodes[:,0].max())
print('FEM node Y range:', nodes[:,1].min(), 'to', nodes[:,1].max())
print('FEM node Z range:', nodes[:,2].min(), 'to', nodes[:,2].max())
