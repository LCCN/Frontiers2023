'''##########################################################################
# Date: 13/03/2023

# NEST VERSION 2.20. Important when checking the manual is needed. 
# (I could no install the 2.30 version)

# The neural model has been changed to include multisynapses. So NMDA can be added
# for inter-population connections. 

# Receptor type: 
# --------------
# 1: AMPA synapses among neurons of the same population
# 2: GABA synapses among neurons of the same population 
# 3: AMPA synpases among neurons of different populations
# 4: AMPA synapses noise background
# 5: After-hyperpolarization calcium-dependant current, it is not a synapse
# but the increase of Ca ions concentration increase after the spike 

This version is only one simulation with the same node receiving different stimulation.
For some reason NEST stops... 

# Parameters selected to have a ~0.4 of plv between the dynamics of the two nodes

# Esta versión es diferente de lo calculado anteriormente. En esta situación 
considero que el par de nodos son las regiones de Precuneus_L y ACC_L. 
De modo que tendré un par de nodos "diferentes" por sujeto, donde 
uso el weight correspondiente a la conexion de esas dos regiones y las 
respectivas distribuciones de campo electrico.
###########################################################################'''

from spectrum import *
#from functions_node import *
import matplotlib.gridspec as gridspec # GRIDSPEC !
from matplotlib.colorbar import Colorbar 
import matplotlib.colors as colors
import matplotlib.pyplot as plt 
from scipy.signal import hilbert, chirp
import numpy as np 
import time as tm 
import nest
import scipy 
import sys 
import os 
import pandas as pd
from scipy.stats import rv_histogram
import file_management
from signal_analysis import *

nest.print_time = False

def bimodal(x, a1, mu1, sigma1, a2, mu2, sigma2):
    y = a1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2)) + a2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2))
    dx = np.diff(x)[0]
    return y/np.sum(y*dx)
##########################################################################
''' Parameters '''
##########################################################################

#synchronization states
# input1 = 0 
# parameters=dict.fromkeys(["gee","gnoise","dI","dIexc"])
# parameters["gee"]    = [0.012,     0.0206, 0.0451]
# parameters["gnoise"] = [0.012*5.8, 0.0456, 0.0312]
# parameters["dI"]     = [0.0,       25.0,   2.5]
# parameters["dIexc"]  = [0.0,       0.0,    32.0] 

ne = 80
ni = 20
resolution = 0.1
runtime = 50000.0 #ms
time_cut = 4000.0
interval = 0.1
dt = 1.0 

gee      = 0.008#parameters["gee"][input1] #0.012#0.012#0.0445#0.012#0.12
ge_noise = 0.0675#parameters["gnoise"][input1] #5.8*gee
dI       = 0.0#parameters["dI"][input1]
dIexc    = 0.0#parameters["dIexc"][input1]
# coupling_factor = 0.15 
# g12 = g21 = coupling_factor

gei = 2.0*gee
gie = 4.0*gee
gii = 4.0*gee
prob = 0.1

tau_syn_AMPA = 3.0
tau_syn_GABA = 3.2
tau_syn_NMDA = 100.0 
E_rev_AMPA = 0.0
E_rev_GABA = -85.0 
E_rev_NMDA = 0.0 

tau_ca = 20.0
E_rev_ca = -90.0
gca = 0.1

rate = 2400.0
I_excitatory = 20.0-5.0 + dI + dIexc# 79.0+dI # 93.5#47.0 + dI # 93.5 without noise
I_inhibitory = 30.0-5.0 + dI # 79.0+dI # 86.5#42.0 + dI # 86.5 without noise

############################################################################################
''' Data structure '''
############################################################################################
input1 = int(sys.argv[1]) # subject
input2 = int(sys.argv[2]) # weight selection
input3 = int(sys.argv[3]) # ntrial

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

files_dir = '/home/jaime/Desktop/neuromodulacion/paper/cingulum_bundle/files/'
files_dir = os.path.join(files_dir, f"{subjects[0]}_AAL2_pass")
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

node1 = np.where(np.array(cingulum_rois) == "Precuneus_R")[0][0]
node2 = np.where(np.array(cingulum_rois) == "Cingulate_Post_R")[0][0]


labels = [cingulum_rois[node1],cingulum_rois[node2]]
velocity = 15.0
delay12 = delay21 = lengths[node1,node2]/velocity # delay of the node-node connection

# # loading distributions
# folder_densities = '/home/jaime/Desktop/neuromodulacion/paper/cingulum_bundle/files/histogramas/Densities/'
# filename = f"densities_OzCzModel_{subjects[input1]}.lzma"
# densities = file_management.load_lzma( os.path.join( folder_densities, filename) )
# x, y = densities[f"x_{node1}"], densities[f"y_{node1}"]

x = np.linspace(-0.2,0.2,10001)
y1 = bimodal(x, 3.5, -0.075, 0.035, 1, 0.075, 0.035)
y2 = bimodal(x, 1, -0.075, 0.035, 3.5, 0.075, 0.035)
y3 = bimodal(x, 1, 0, 0.035, 1,0, 0.035)

g = []
dx = np.diff(x)[0]
xn = np.array(list(x-np.diff(x)[0]/2)+[x[-1]+np.diff(x)[0]/2])
np.random.seed(1993)
for i,y in enumerate([y1,y2,y3]):
    rv = rv_histogram((y,xn))
    g.append(rv.rvs(size=80))
weights_modulation = g[input1]

# Setting the parameters : 
folder_wp = '/home/jaime/Desktop/neuromodulacion/paper/cingulum_bundle/'

data_working_point = pd.read_csv(os.path.join(folder_wp,"cingulum_bundle_working_point_solution.txt"), sep="\t",index_col=None)
coupling_factor = data_working_point["coup"].values[input1]
g12 = g21 = weights[node1,node2]*coupling_factor # coupling factor

############################################################################################
fex_seq     = np.arange(4.0,18.1,0.25)
weights_seq = np.linspace(0.0,2000.0,21)
fex         = fex_seq[input2]

columns = ["trial","histogram","node","weight","fex","fpeak","amplitude_fex","amplitude_fpeak","amplitude_norm_fex","amplitude_norm_fpeak","amplitude_fex_rel","amplitude_fpeak_rel","amplitude_norm_fex_rel","amplitude_norm_fpeak_rel"]
columns3 = ["trial","histogram","weight","fex","plv_lfp","plv_volt"]

data1 = dict.fromkeys(columns)
data2 = dict.fromkeys(columns)
data3 = dict.fromkeys(columns)
for key in columns:
    data1[key] = []
    data2[key] = []
for key in columns3:
    data3[key] = []

#######################################################################################################
''' Population definition '''
#######################################################################################################
params_excitatory = []
params_inhibitory = []
params_rs={'C_m': 104.0, 'g_L': 4.3, 'E_L': -65.0, 'V_th':-52.0, 'Delta_T': 0.8, 'a': -0.8, 'tau_w':88.0, 
           'b': 65.0, 'V_reset': -53.0, 'I_e': I_excitatory, 't_ref': 2.0,
           'tau_syn': [tau_syn_AMPA, tau_syn_GABA, tau_syn_NMDA, tau_syn_AMPA, tau_ca],
           'E_rev':   [E_rev_AMPA,   E_rev_GABA,   E_rev_NMDA,   E_rev_AMPA,  E_rev_ca]}
          
params_fs={'C_m': 59.0, 'g_L': 2.9, 'E_L': -62.0, 'V_th': -42.0, 'Delta_T': 3.0, 'a': 1.8, 'tau_w': 16.0, 
           'b': 61.0, 'V_reset': -54.0, 'I_e': I_inhibitory, 't_ref': 2.0,
            'tau_syn': [tau_syn_AMPA, tau_syn_GABA, tau_syn_NMDA, tau_syn_AMPA, tau_ca],
            'E_rev':   [E_rev_AMPA,   E_rev_GABA,   E_rev_NMDA,   E_rev_AMPA,   E_rev_ca]}

sg = 0.01
np.random.seed(12345)
nkeys = len(params_rs.keys())
rn = np.random.normal(0,1,ne*nkeys)
k = 0 
for i in range(ne): 
    params_excitatory.append(dict.fromkeys(params_rs.keys()))
    for key in list(params_rs.keys())[:11]:
        params_excitatory[-1][key] = params_rs[key]*(1.0+sg*rn[k])
        k+=1
    params_excitatory[-1]["E_rev"]   = params_rs["E_rev"]
    params_excitatory[-1]["tau_syn"] = params_rs["tau_syn"]

nkeys = len(params_fs.keys())
#rn = 2.0*np.random.rand(ni*nkeys)-1.0
rn = np.random.normal(0,1,ni*nkeys)
k = 0
for i in range(ni): 
    params_inhibitory.append(dict.fromkeys(params_fs.keys()))
    for key in list(params_fs.keys())[:11]:
        params_inhibitory[-1][key] = params_fs[key]*(1.0+sg*rn[k])
        k+=1
    params_inhibitory[-1]["E_rev"]   = params_fs["E_rev"]
    params_inhibitory[-1]["tau_syn"] = params_fs["tau_syn"]

time_initial = tm.time()
nest.ResetKernel()
nest.SetKernelStatus({"local_num_threads": 4})
msd = 1937 + 2*input3+1 # mas ter see d
n_vp = nest.GetKernelStatus('total_num_virtual_procs')
msdrange1 = range(msd, msd+n_vp)
pyrngs = [ np.random.RandomState(s) for s in msdrange1 ]
msdrange2 = range(msd+n_vp+1, msd+n_vp*2+1 )
nest.SetKernelStatus({ 'grng_seed': msd+n_vp,'rng_seeds' : msdrange2 })

excitatory, inhibitory = [],[]
noise_ex, noise_in = [],[]
conn_ee, conn_ei, conn_ie, conn_ii = [],[],[],[]
conn_12, conn_21 = [],[]
multimeter_ex, multimeter_in = [], []
spike_detector_ex, spike_detector_in = [],[]
signal = []

for i, weight in enumerate(weights_seq):
    excitatory.append( nest.Create("aeif_cond_alpha_multisynapse", ne, params=params_excitatory) )
    inhibitory.append( nest.Create("aeif_cond_alpha_multisynapse", ni, params=params_inhibitory) ) 

for i, weight in enumerate(weights_seq):
    noise_ex.append( nest.Create('poisson_generator', ne) ) 
    noise_in.append( nest.Create('poisson_generator', ni) )
    nest.SetStatus(noise_ex[-1], {'rate' : rate})
    nest.SetStatus(noise_in[-1], {'rate' : rate})

    nest.Connect(noise_ex[-1], excitatory[i], conn_spec='one_to_one', syn_spec = {'weight': ge_noise, 'receptor_type':4})
    nest.Connect(noise_in[-1], inhibitory[i], conn_spec='one_to_one', syn_spec = {'weight': ge_noise, 'receptor_type':4})

    # intra-connectivity (all-to-all):
    conn_ee.append( nest.Connect(excitatory[i], excitatory[i], syn_spec={'weight': gee, 'delay': 0.5, 'receptor_type':1}) )
    conn_ei.append( nest.Connect(excitatory[i], inhibitory[i], syn_spec={'weight': gei, 'delay': 0.5, 'receptor_type':1}) )
    conn_ie.append( nest.Connect(inhibitory[i], excitatory[i], syn_spec={'weight': gie, 'delay': 0.5, 'receptor_type':2}) )
    conn_ii.append( nest.Connect(inhibitory[i], inhibitory[i], syn_spec={'weight': gii, 'delay': 0.5, 'receptor_type':2}) )

    #estimulacion con distribuciones
    for ii in range(ne): # aqui hay 80 objetos
        signal.append(nest.Create('ac_generator'))
        signalparams = {'amplitude': weight*weights_modulation[ii], 'frequency': fex, "offset": 0.0, "phase": 0.0, "start": 2000.0}
        nest.SetStatus(signal[-1],signalparams)
        nest.Connect(signal[-1], [excitatory[i][ii]])

    multimeter_ex.append( nest.Create('multimeter', params={'interval': dt, 'withtime': True, 'record_from': ['V_m','g_1','g_2','g_3','g_4'] }))
    multimeter_in.append( nest.Create('multimeter', params={'interval': dt, 'withtime': True, 'record_from': ['V_m'] }) )#'record_from': ['V_m','g_ex','g_in'] })

    nest.Connect(multimeter_ex[-1], excitatory[i])
    nest.Connect(multimeter_in[-1], inhibitory[i])

    # spike_detector_ex[-1].append( nest.Create('spike_detector') )
    # spike_detector_in[-1].append( nest.Create('spike_detector') )

    # nest.Connect( excitatory[-1][0], spike_detector_ex[-1][0] )    
    # nest.Connect( inhibitory[-1][0], spike_detector_in[-1][0] )

print("listo")
nest.Simulate(runtime)


psd_matrix = []
for i, weight in enumerate(weights_seq):
    # measurements 

    dmm    = nest.GetStatus(multimeter_ex[i])[0]["events"]
    neuron = dmm['senders']
    #print(neuron)
    time   = dmm['times']
    vm     = dmm['V_m']
    g1     = dmm['g_1'] # AMPA
    g2     = dmm['g_2'] # GABA
    g3     = dmm['g_3'] # NMDA
    g4     = dmm['g_4'] # noise

    time_ = [ [] for k in range(ne)]
    vm_   = [ [] for k in range(ne)]
    g1_   = [ [] for k in range(ne)] 
    g2_   = [ [] for k in range(ne)]
    g3_   = [ [] for k in range(ne)] 
    g4_   = [ [] for k in range(ne)]
    
    for k in range(ne): 
        #sender = np.where(neuron==282*i+k+1) # por el orden en el que se ha creado los objetos
        sender = np.where(neuron==100*i+k+1)
        time_[k] = time[sender]
        vm_[k]   = vm[sender]
        g1_[k]  = g1[sender]
        g2_[k]  = g2[sender]
        g3_[k]  = g3[sender]    
        g4_[k]  = g4[sender]

    time_ = np.array(time_)
    vm_   = np.array(vm_)
    g1_  = np.array(g1_)
    g2_  = np.array(g2_)
    g3_  = np.array(g3_)
    g4_  = np.array(g4_)

    I1 = np.mean(np.abs(g1_*(vm_-E_rev_AMPA)),axis=0)
    I2 = np.mean(np.abs(g2_*(vm_-E_rev_GABA)),axis=0)
    I3 = np.mean(np.abs(g3_*(vm_-E_rev_NMDA)),axis=0)
    I4 = np.mean(np.abs(g4_*(vm_-E_rev_AMPA)),axis=0)
    average_voltage = np.mean(vm_, axis=0)
    
    print('-------------')
    print('Computing PSD')
    print('-------------')

    lfp  = (I1+I2+I3+I4)[time_[0]>=time_cut] 
    volt = np.mean(vm_, axis=0)[time_[0]>=time_cut] 
    t    = time_[0][time_[0]>=time_cut] 

    fs = 1e3/dt
    nperseg = 4000/dt
    noverlap = int(0.5*nperseg)
    nfft = 2**12

    lfp_filtered = bandpass_filter(lfp, norder=4, f_range=(8,12), fs=fs)
    volt_filtered = bandpass_filter(volt, norder=4, f_range=(8,12), fs=fs)

    fr, psd1  = scipy.signal.csd( lfp_filtered,  lfp_filtered,  fs=fs, window="hann", nfft=nfft, nperseg=nperseg, noverlap = noverlap)
    fr, psd2  = scipy.signal.csd( volt_filtered, volt_filtered, fs=fs, window="hann", nfft=nfft, nperseg=nperseg, noverlap = noverlap)
    x1 = (lfp_filtered-lfp_filtered.mean())/lfp_filtered.std()
    x2 = (volt_filtered-volt_filtered.mean())/volt_filtered.std()
    fr, npsd1  = scipy.signal.csd( x1, x1, fs=fs, window="hann", nfft=nfft, nperseg=nperseg, noverlap = noverlap)
    fr, npsd2  = scipy.signal.csd( x2, x2, fs=fs, window="hann", nfft=nfft, nperseg=nperseg, noverlap = noverlap)

    fr, psdfull  = scipy.signal.csd( lfp,  lfp,  fs=fs, window="hann", nfft=nfft, nperseg=nperseg, noverlap = noverlap)
    w = (fr>=0) & (fr<50)
    psd_matrix.append( psdfull[w] )

    dfr = np.diff(fr)[0]
    fpeak = fr[np.argmax(npsd1)]
    amplitude_fpeak      = np.sum(  psd1[ (fr>=fpeak-0.5) & (fr<=fpeak+0.5) ] )*dfr 
    amplitude_fex        = np.sum(  psd1[ (fr>=fex-0.5)   & (fr<=fex+0.5)   ] )*dfr
    amplitude_norm_fpeak = np.sum( npsd1[ (fr>=fpeak-0.5) & (fr<=fpeak+0.5) ] )*dfr
    amplitude_norm_fex   = np.sum( npsd1[ (fr>=fex-0.5)   & (fr<=fex+0.5)   ] )*dfr

    data1["trial"].append(input3)
    #data1["node"].append(j)
    data1["node"].append(labels[0])
    data1["histogram"].append(input1)
    data1["weight"].append(weight)
    data1["fex"].append(fex)
    data1["fpeak"].append(fpeak)
    data1["amplitude_fex"].append(amplitude_fex)
    data1["amplitude_fpeak"].append(amplitude_fpeak)
    data1["amplitude_norm_fex"].append(amplitude_norm_fex)
    data1["amplitude_norm_fpeak"].append(amplitude_norm_fpeak)
    data1["amplitude_fex_rel"].append(1)
    data1["amplitude_fpeak_rel"].append(1)
    data1["amplitude_norm_fex_rel"].append(1)
    data1["amplitude_norm_fpeak_rel"].append(1)

    fpeak = fr[np.argmax(npsd2)]
    amplitude_fpeak      = np.sum(  psd2[ (fr>=fpeak-0.5) & (fr<=fpeak+0.5) ] )*dfr 
    amplitude_fex        = np.sum(  psd2[ (fr>=fex-0.5)   & (fr<=fex+0.5)   ] )*dfr
    amplitude_norm_fpeak = np.sum( npsd2[ (fr>=fpeak-0.5) & (fr<=fpeak+0.5) ] )*dfr
    amplitude_norm_fex   = np.sum( npsd2[ (fr>=fex-0.5)   & (fr<=fex+0.5)   ] )*dfr

    data2["trial"].append(input3)
    data2["node"].append(labels[0])
    data2["histogram"].append(input1)
    data2["weight"].append(weight)
    data2["fex"].append(fex)
    data2["fpeak"].append(fpeak)
    data2["amplitude_fex"].append(amplitude_fex)
    data2["amplitude_fpeak"].append(amplitude_fpeak)
    data2["amplitude_norm_fex"].append(amplitude_norm_fex)
    data2["amplitude_norm_fpeak"].append(amplitude_norm_fpeak)
    data2["amplitude_fex_rel"].append(1)
    data2["amplitude_fpeak_rel"].append(1)
    data2["amplitude_norm_fex_rel"].append(1)
    data2["amplitude_norm_fpeak_rel"].append(1)

psd_matrix = np.array(psd_matrix)
df1 = pd.DataFrame( data1 ) # columns=columns )
df2 = pd.DataFrame( data2 ) # columns=columns )

time_final = tm.time() 
diff  = time_final-time_initial
horas = int(diff/3600)
diff  = diff-horas*3600
mint  = int(diff/60)
diff  = diff-mint*60
seg   = int(diff)

print('Simulation time with resolution dt = ', resolution ,' ms :')
print('_________________')
print(' ')
print(horas,'h', mint, 'min', seg, 's')

current_dir = os.getcwd() 
parent_dir = os.path.join( current_dir, "tacs_densities")
parent_dir = os.path.join( parent_dir, subjects[0])
title1 =  f"data_lfp_{input1}_{input2}_{input3}.lzma"
title2 = f"data_volt_{input1}_{input2}_{input3}.lzma"
title3 = f"data_psd_{input1}_{input2}_{input3}.lzma"

file_management.save_lzma(df1, title1, parent_dir = parent_dir)
file_management.save_lzma(df2, title2, parent_dir = parent_dir)
file_management.save_lzma(psd_matrix, title3, parent_dir = parent_dir)
