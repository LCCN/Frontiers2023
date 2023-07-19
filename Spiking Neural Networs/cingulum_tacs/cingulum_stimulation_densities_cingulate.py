''' ###########################################################################
#                     SIMULATION OF THE NETWORK                               #
############################################################################'''

import numpy as np
import os
import sys
import time as tm
import copy
import pandas as pd
from functions import *
from network import *
import scipy
from scipy.stats import rv_histogram
import file_management
from signal_analysis import *

import nest

'''############################################################################
#                                  Inputs                                     #
############################################################################'''
input1 = int(sys.argv[1])  # subject
input2 = int(sys.argv[2])  # amplitude of the external signal 
input3 = int(sys.argv[3])  # fex 
input4 = int(sys.argv[4])  # trial block 
# input3 = 20.0   # float(sys.argv[2])  # velocity
# input4 = 1000.0 # float(sys.argv[3])  # firing rate of the poisson noise background
# input5 = 0.0    # float(sys.argv[4])  # constant current injection to each population
nest.print_time=True
'''############################################################################
#                             Subjects matrices                               #
############################################################################'''
nnodes = 22 # number of nodes of the network 
bands  = ["delta","theta","alpha","beta","gamma"]
nbands = len(bands)

subjects = ['NEMOS_035', 'NEMOS_049', 'NEMOS_050', 'NEMOS_058', 'NEMOS_059',
            'NEMOS_064', 'NEMOS_065', 'NEMOS_071', 'NEMOS_075', 'NEMOS_077', 'NEMOS_AVG']

cingulum_rois = ['Frontal_Mid_2_L', 'Frontal_Mid_2_R',
                 'Insula_L', 'Insula_R',
                 'Cingulate_Ant_L', 'Cingulate_Ant_R', 'Cingulate_Post_L', 'Cingulate_Post_R',
                 'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L',
                 'ParaHippocampal_R', 'Amygdala_L', 'Amygdala_R',
                 'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L',
                 'Parietal_Inf_R', 'Precuneus_L', 'Precuneus_R',
                 'Thalamus_L', 'Thalamus_R']

current_folder = os.getcwd()
folder_save = os.path.join(current_folder, "OzCz_densities_cingulate")
folder_save = os.path.join(folder_save, subjects[input1])

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

node1 = np.where(np.array(cingulum_rois) == "Cingulate_Ant_R")[0][0]
node2 = np.where(np.array(cingulum_rois) == "Cingulate_Post_R")[0][0]
# files_dir = os.path
# .join(current_folder, "FC_data")
# files_dir = os.path.join(files_dir, "FCrms_"+subjects[input1])

# cortical_nodes_list = np.loadtxt( os.path.join(files_dir,"cortical_nodes.txt") ).astype(int)
# cortical_nodes = len(cortical_nodes_list)

''' ###########################################################################
#         STIMULATION IN THE WOKRING POINT                                    #
############################################################################'''

folder_densities = '/home/jaime/Desktop/neuromodulacion/paper/cingulum_bundle/files/histogramas/Densities/'
filename = f"densities_OzCzModel_{subjects[input1]}.lzma"
densities = file_management.load_lzma( os.path.join( folder_densities, filename) )

weights_modulation = np.zeros((nnodes, 80))
np.random.seed(1993)
for i in range(nnodes): 
    x, y = densities[f"x_{i}"], densities[f"y_{i}"]
    #weights_modulation[i] = np.interp(u, y, x)
    dx = np.diff(x)[0]
    xn = np.array(list(x-np.diff(x)[0]/2)+[x[-1]+np.diff(x)[0]/2])
    rv = rv_histogram((y,xn))
    if i == node1:
        weights_modulation[i] = rv.rvs(size=80)#np.interp(u, y, vi x)
    else:
        weights_modulation[i] = 0.0*rv.rvs(size=80)#np.interp(u, y, vi x)

ampseq = np.linspace(0.0,100.0,21)
weights_modulation = weights_modulation*ampseq[input2]
#np.savetxt(os.path.join(folder_save,f"weight_stimulation_{subjects[input1]}_{input2}_{input3}.txt"), weights_modulation)
''' ###########################################################################
#                                  NETWORK                                    #
############################################################################'''
# Setting the parameters : 
data_working_point = pd.read_csv("cingulum_bundle_working_point_solution.txt", sep="\t",index_col=None)
frequency = np.arange(4,18.1,0.25)[input3]
#data_working_point["fpeak_cluster"].values[input1]
coupling_factor = data_working_point["coup"].values[input1]
# coupling_factor = [0.2 , 0.16, 0.2 , 0.32, 0.18, 0.33, 0.51, 0.25, 0.25, 0.25][input1]
# frequency = [10.19, 9.98, 10.23, 10.14, 10.09, 10.16, 10.02, 10.25, 10.1, 10.23][input1]

velocity        = 15.0 # propagation speed (delay = length/velocity )
node_currents   = np.ones(nnodes)*0.0 # constant current injection of each compartment (the same for all)
ns = input4
# frate           = input4 # firing rate of poisson background noise
# NMDA            = False # NMDA synapses among populations 
# all_to_all      = True # all to all connectivity among neurons in each population 
# ge              = 0.012 # excitatory conductance 
# gnoise          = 0.012 # excitatory conductance of the background noise 

# Starting the simulation: 
tick = tm.time()
nsims    = 10 # number of simulations 
runtime  = 25000.0 # simulation time 
time_cut = 4000.0   # tim
# t      = [ [] for ns in range(nsims) ]
# volt   = [ [] for ns in range(nsims) ]
# lfp    = [ [] for ns in range(nsims) ]
# times  = [ [] for ns in range(nsims) ]
# spikes = [ [] for ns in range(nsims) ]
# rate   = [ [ [] for j in range(nnodes) ] for ns in range(nsims) ]

t      = [ [] for i in range(nsims) ]
volt   = [ [] for i in range(nsims) ]
lfp    = [ [] for i in range(nsims) ]
times  = [ [] for i in range(nsims) ]
spikes = [ [] for i in range(nsims) ]
#rate   = [ [] for j in range(nnodes) ]

for ns in range(nsims):

    seed = 1993+(2*nsims+1)*input4+2*ns+1
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
    net.set_injected_currents_from_densities(weights_modulation, frequency)
    print('--------------')
    print("External magnetic field loaded")
    net.run_simulation()

    t[ns], volt[ns], lfp[ns] = net.get_data()
    times[ns], spikes[ns] = net.get_spikes_new() # net.get_spikes()
    # t, volt, lfp = net.get_data()
    # times, spikes = net.get_spikes_new() # net.get_spikes()

    # for j in range(nnodes):
    #     nrs = np.logical_and(spikes[ns]>= 100*j, spikes[ns]<100*(j+1))
    #     xaux = times[ns][nrs]
    #     yaux = spikes[ns][nrs]
    #     time_rate, rate[ns][j] = firing_rate(xaux,yaux,runtime=runtime,t0=time_cut,timebin=1.0, sg=10)

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
dt = 1
fs = 1e3/dt
nperseg = 4000/dt
noverlap = int(0.5*nperseg)
nfft = 2**12

# voy a no guardar los ficheros porque si no... esto se va un poco de madre.
phase = [ [ [] for k in range(nsims)] for j in range(nnodes)]

series = {}
for ns in range(nsims):
    ns_new = int(ns+nsims*input4)
    for j in range(nnodes):
        # series[f"volt_{j}_{ns_new}"] = np.array(volt[ns][j],dtype=np.float32)
        series[f"lfp_{j}_{ns_new}"]  = np.array(lfp[ns][j],dtype=np.float32)
        lfp_ = np.array(lfp[ns][j],dtype=np.float32)
        phase[j][ns] = bandpass_filter_and_hilbert_transform(lfp_,  fs, 10.0, 2.0, 4)[-1]  

### compute PLV ### 
phase = np.array(phase)
print(np.shape(phase))
npairs = int(nnodes*(nnodes-1)/2)
plv_alpha_list = np.zeros((npairs, nsims))
plv_alpha_matrix = np.zeros((nnodes, nnodes, nsims))
count = 0 
for j1 in range(nnodes):
    for j2 in range(j1+1,nnodes):
        for k in range(nsims):
            phi1 = phase[j1,k]
            phi2 = phase[j2,k]
            plv_value = get_plv(phi1,phi2)
            print(plv_value)
            plv_alpha_list[count,k] = plv_value
            plv_alpha_matrix[j1,j2,k] = plv_value
            plv_alpha_matrix[j2,j1,k] = plv_value
        count+=1
###################################################################

columns = ["subject","mode","trial","node", "w", "fex","fpeak", "amp_fex", "amp_fpeak","amp_norm_fex", "amp_norm_fpeak", "plv_mean","plv_std"] 

rows = dict.fromkeys(columns)
for key in columns:
    rows[key] = []

fpeak_matrix = np.zeros((nnodes,nsims))
power_fex_matrix = np.zeros((nnodes,nsims))
power_fpeak_matrix = np.zeros((nnodes,nsims))
power_norm_fex_matrix = np.zeros((nnodes,nsims))
power_norm_fpeak_matrix = np.zeros((nnodes,nsims))

print('---')
for k, ns in enumerate(range(nsims)):
    ns_new = int(ns+nsims*input4)
    for j in range(nnodes):
        rows["mode"].append(0)
        rows["subject"].append(subjects[input1])
        rows["w"].append( ampseq[input2] )
        rows["trial"].append( ns_new )
        rows["node"].append( cingulum_rois[j] ) 
        rows["fex"].append(frequency)

        lfp = series[f"lfp_{j}_{k}"]
        fr, psd1  = scipy.signal.csd( lfp,  lfp,  fs=fs, window="hann", nfft=nfft, nperseg=nperseg, noverlap = noverlap)
        x1 = (lfp-lfp.mean())/lfp.std()
        fr, npsd1  = scipy.signal.csd( x1, x1, fs=fs, window="hann", nfft=nfft, nperseg=nperseg, noverlap = noverlap)
        dfr = np.diff(fr)[0]

        fpeak = fr[np.argmax(npsd1)]
        amplitude_fpeak      = np.sum(  psd1[ (fr>=fpeak-0.5)  & (fr<=fpeak+0.5) ] )*dfr 
        amplitude_fex        = np.sum(  psd1[ (fr>=frequency-0.5) & (fr<=frequency+0.5)] )*dfr
        amplitude_norm_fpeak = np.sum( npsd1[ (fr>=fpeak-0.5)  & (fr<=fpeak+0.5) ] )*dfr
        amplitude_norm_fex   = np.sum( npsd1[ (fr>=frequency-0.5) & (fr<=frequency+0.5)] )*dfr

        #phase_volt_list.append( bandpass_filter_and_hilbert_transform(volt, fs, 10.0, 2.0, 4) ) 
        #phase[i][j][k].append(  bandpass_filter_and_hilbert_transform(lfp,  fs, 10.0, 2.0, 4) ) 
        
        rows["fpeak"].append( fpeak )
        rows["amp_fex"].append( amplitude_fex )
        rows["amp_fpeak"].append( amplitude_fpeak )
        rows["amp_norm_fex"].append( amplitude_norm_fex )
        rows["amp_norm_fpeak"].append( amplitude_norm_fpeak )
        fpeak_matrix[j,k] = fpeak 
        power_fex_matrix[j,k] = amplitude_fex
        power_fpeak_matrix[j,k] = amplitude_fpeak
        power_norm_fex_matrix[j,k] = amplitude_norm_fex
        power_norm_fpeak_matrix[j,k] = amplitude_norm_fpeak

        rows["plv_mean"].append( np.mean(np.delete( plv_alpha_matrix[j,:,k], j)) )
        rows["plv_std"].append( np.std(np.delete( plv_alpha_matrix[j,:,k], j)) )

parent_dir = folder_save#os.path.join(folder_save, subjects[input1])

title = f"stimulation_nodes_{subjects[input1]}_{input2}_{input3}_{input4}.lzma"
dfnew = pd.DataFrame(rows, columns=columns)
file_management.save_lzma( dfnew, title, parent_dir = parent_dir)
# Al no estar conectadas la plv no interesa 
#title = f"stimulation_plv_{subjects[input1]}_{input2}_{input3}_{input4}.lzma"
#file_management.save_lzma( plv_alpha_matrix, title, parent_dir = parent_dir)
#dfnew.to_csv(os.path.join(folder, title),sep="\t")

# columns = ["subject","mode","trial", "w", "fex", "fpeak", "fpeak_std", "plv_mean","plv_std", 
#            "amp_fpeak", "amp_fpeak_std",
#            "amp_fex",   "amp_fex_std",   "amp_norm_fex",  "amp_norm_fex_std",
#            "amp_norm_fpeak", "amp_norm_fpeak_std"]

# idcb = np.array([8, 9, 4, 5, 0, 15, 2, 11, 19, 3, 12, 10, 17])
# idcb = np.sort(idcb)
# rows = dict.fromkeys(rows)

# for key in columns:
#     rows[key] = [] 

# for k, ns in enumerate(range(nsims)):
#     rows["subject"].append( subjects[input1] )
#     rows["mode"].append( 0 )
#     rows["trial"].append( ns_new )
#     rows["w"].append( ampseq[input2] )
#     rows["fex"].append( frequency )
    
#     rows["fpeak"].append( np.mean( fpeak_matrix[idcb,k] ))
#     rows["fpeak_std"].append(  np.std(  fpeak_matrix[idcb,k] ))
    
#     rows["plv_mean"].append( np.mean( plv_alpha_list[idcb,k] ))
#     rows["plv_std"].append(  np.std(  plv_alpha_list[idcb,k] ))
    
#     rows["amp_fpeak"].append( np.mean( power_fpeak_matrix[idcb,k] ))
#     rows["amp_fpeak_std"].append(   np.std( power_fpeak_matrix[idcb,k] ))
#     rows["amp_norm_fpeak"].append( np.mean( power_norm_fpeak_matrix[idcb,k] ))
#     rows["amp_norm_fpeak_std"].append(   np.std( power_norm_fpeak_matrix[idcb,k] ))
    
#     # rows["amp_fbaseline"].append(    np.mean( power_fpeak_matrix[0,idcb,k] ))
#     # rows["amp_fbaseline_std"].append( np.std( power_fpeak_matrix[0,idcb,k] ))
#     # rows["amp_norm_fbaseline"].append(    np.mean( power_norm_fpeak_matrix[0,idcb,k] ))
#     # rows["amp_norm_fbaseline_std"].append( np.std( power_norm_fpeak_matrix[0,idcb,k] ))
    
#     rows["amp_fex"].append(    np.mean( power_fex_matrix[idcb,k] ))
#     rows["amp_fex_std"].append( np.std( power_fex_matrix[idcb,k] ))
#     rows["amp_norm_fex"].append(    np.mean( power_norm_fex_matrix[idcb,k] ))
#     rows["amp_norm_fex_std"].append( np.std( power_norm_fex_matrix[idcb,k] ))

# title = f"stimulation_precuneus1_{subjects[input1]}_cluster.txt"
# dfnew2 = pd.DataFrame( rows, columns=columns)
# dfnew2.to_csv(os.path.join(folder, title),  sep="\t")

# data_spikes = {}
# for ns in range(nsims):
#     ns_new = int(ns+nsims*input3)
#     data_spikes[f"tspikes_{ns_new}"] = times[ns].astype(np.float32)
#     data_spikes[f"ncell_{ns_new}"]   = spikes[ns].astype(int)+1

# parent_dir = os.path.join(current_folder, "OzCz")
# parent_dir = os.path.join(parent_dir, "precuneus1")
# parent_dir = os.path.join(parent_dir, subjects[input1])

#file_management.save_lzma(data_spikes,f"data_spikes_OzCz_{input1}_{input2}_{input3}.lzma", parent_dir=folder_save)
#file_management.save_lzma(series,f"series_OzCz_{input1}_{input2}_{input3}.lzma", parent_dir=parent_dir)

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