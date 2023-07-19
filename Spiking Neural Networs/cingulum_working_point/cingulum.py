''' ###########################################################################
#                     SIMULATION OF THE NETWORK                               #
############################################################################'''

import numpy as np
import os
import sys
import time as tm
import copy
import pandas
from functions import *
from network import *
import scipy
import file_management
import nest

current_folder = os.getcwd()

'''############################################################################
#                                  Inputs                                     #
############################################################################'''
input1 = int(sys.argv[1])  # subject
input2 = int(sys.argv[2])  # coupling factor
# input3 = 20.0   # float(sys.argv[2])  # velocity
# input4 = 1000.0 # float(sys.argv[3])  # firing rate of the poisson noise background
# input5 = 0.0    # float(sys.argv[4])  # constant current injection to each population
nest.print_time=True
'''############################################################################
#                             Subjects matrices                               #
############################################################################'''
cingulum_rois = ['Frontal_Mid_2_L', 'Frontal_Mid_2_R',
                 'Insula_L', 'Insula_R',
                 'Cingulate_Ant_L', 'Cingulate_Ant_R', 'Cingulate_Post_L', 'Cingulate_Post_R',
                 'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L',
                 'ParaHippocampal_R', 'Amygdala_L', 'Amygdala_R',
                 'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L',
                 'Parietal_Inf_R', 'Precuneus_L', 'Precuneus_R',
                 'Thalamus_L', 'Thalamus_R']

nnodes = 22 # number of nodes of the network 
bands  = ["delta","theta","alpha","beta","gamma"]
nbands = len(bands)

subjects = ['NEMOS_035', 'NEMOS_049', 'NEMOS_050', 'NEMOS_058', 'NEMOS_059',
            'NEMOS_064', 'NEMOS_065', 'NEMOS_071', 'NEMOS_075', 'NEMOS_077', 'NEMOS_AVG']

files_dir = os.path.join(current_folder, "files")
files_dir = os.path.join(files_dir, f"{subjects[input1]}_AAL2_pass")

files = ['tract_lengths.txt','weights.txt','centres.txt']

lengths = np.loadtxt(os.path.join(files_dir,files[0])) # tract length matrix 
weights = np.loadtxt(os.path.join(files_dir,files[1])) # weight matrix

weights = weights/np.max(weights) # weight matrix normalization 
SClabs = []
for line in open(os.path.join(files_dir, files[2])):
    SClabs.append( line.split()[0] )

SC_cb_idx = [SClabs.index(roi) for roi in cingulum_rois]  # find indexes in SClabs that matches cortical_rois
SClabs = np.array( SClabs )

weights = weights[:, SC_cb_idx][SC_cb_idx]
lengths = lengths[:, SC_cb_idx][SC_cb_idx]
region_labels = SClabs[SC_cb_idx]
connectivity = np.zeros((nnodes,nnodes)) # conectivity matrix: (1,0)=(connection, no connection)
connectivity[np.where(weights>0)]=1
connectivity = connectivity.astype(int)

# files_dir = os.path.join(current_folder, "FC_data")
# files_dir = os.path.join(files_dir, "FCrms_"+subjects[input1])

# cortical_nodes_list = np.loadtxt( os.path.join(files_dir,"cortical_nodes.txt") ).astype(int)
# cortical_nodes = len(cortical_nodes_list)
''' ###########################################################################
#                                  NETWORK                                    #
############################################################################'''

# Setting the parameters : 

coupling_factor = np.linspace(0,0.8,81)[input2] # input2 # coupling factor value
velocity        = 15.0 # propagation speed (delay = length/velocity )
node_currents   = np.ones(nnodes)*0.0 # constant current injection of each compartment (the same for all)
# frate           = input4 # firing rate of poisson background noise
# NMDA            = False # NMDA synapses among populations 
# all_to_all      = True # all to all connectivity among neurons in each population 
# ge              = 0.012 # excitatory conductance 
# gnoise          = 0.012 # excitatory conductance of the background noise 

# Starting the simulation: 
tick = tm.time()
nsims    = 5 # number of simulations 
runtime  = 12000.0 # simulation time 
time_cut = 2000.0   # tim
t      = [ [] for ns in range(nsims) ]
volt   = [ [] for ns in range(nsims) ]
lfp    = [ [] for ns in range(nsims) ]
times  = [ [] for ns in range(nsims) ]
spikes = [ [] for ns in range(nsims) ]
rate   = [ [ [] for j in range(nnodes) ] for ns in range(nsims) ]

for ns in range(nsims):
    seed = 1993+2*ns+1
    net = Network(nnodes=nnodes,weights=weights,lengths=lengths,coupling_factor=coupling_factor, 
    velocity=velocity, connectivity=connectivity, seed=seed, 
    node_currents=node_currents, runtime=runtime, time_cut=time_cut)
    
    print('Simulation ',ns)
    print('--------------')
    print('Network loaded')
    net.set_multimeters()
    print('--------------')
    print('Multimeters loaded')
    net.set_spike_detectors()
    print('--------------')
    print('Spike detectors loaded')
    net.run_simulation()

    t[ns], volt[ns], lfp[ns] = net.get_data()
    times[ns], spikes[ns] = net.get_spikes_new() # net.get_spikes()

    for j in range(nnodes):
        nrs = np.logical_and(spikes[ns]>= 100*j, spikes[ns]<100*(j+1))
        xaux = times[ns][nrs]
        yaux = spikes[ns][nrs]
        time_rate, rate[ns][j] = firing_rate(xaux,yaux,runtime=runtime,t0=time_cut,timebin=1.0, sg=10)

tock  = tm.time()
diff  = tock-tick
hours = int(diff/3600)
diff  = diff-hours*3600
mins  = int(diff/60)
diff  = diff-mins*60
segs  = int(diff)

print(f'Simulation time with resolution dt = {net.resolution} ms :')
print('_________________')
print(' ')
print(hours,'h', mins, 'min', segs, 's')

'''############################################################################
#           Saving time series and spikes (excitatory only)                   #
############################################################################'''

series = {}
for ns in range(nsims):
    for j in range(nnodes):
        # series[f"volt_{j}_{ns}"] = np.array(volt[ns][j],dtype=np.float32)
        series[f"lfp_{j}_{ns}"]  = np.array(lfp[ns][j],dtype=np.float32)

data_spikes = {}
for ns in range(nsims):
    data_spikes[f"tspikes_{ns}"] = times[ns].astype(np.float32)
    data_spikes[f"ncell_{ns}"]   = spikes[ns].astype(int)+1

file_management.save_lzma(data_spikes,f"data_spikes_{input1}_{input2}.lzma", parent_dir=subjects[input1])
file_management.save_lzma(series,f"series_{input1}_{input2}.lzma", parent_dir=subjects[input1])

# Each simulation will have different number of spikes, so to save them in a DataFrame it is required to fill
# the gaps with nans 
# maxsize = max([a.size for a in data_spikes.values()])
# data_pad = {k:np.pad(v, pad_width=(0,maxsize-v.size,), mode='constant', constant_values=0) for k,v in data_spikes.items()}
# df = pandas.DataFrame(data_pad)

# import matplotlib.pyplot as plt
# os.chdir(current_folder)
# for ns in range(nsims):
#     plt.figure(figsize=(20,40)) 
#     plt.plot(data_spikes[f"tspikes_{ns}"],data_spikes[f"ncell_{ns}"],'o',markersize=1.)
#     for i in range(nnodes): 
#         plt.axhline(100*i,linestyle='--',color='black')
#     #plt.xlim([1000,2000])
#     #plt.ylim([0,12000])
#     title = f"test_{input3}_{100*input2}_{input3}_{input5}_{ns}.png"
#     #plt.savefig(title, dpi=300, bbox_inches='tight')