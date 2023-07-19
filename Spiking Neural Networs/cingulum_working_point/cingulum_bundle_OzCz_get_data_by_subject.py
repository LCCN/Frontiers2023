import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import scipy 
import sys 
import os 
from signal_analysis import *
import file_management

amplitude = np.linspace(0,20.0,21)
s = int(sys.argv[1]) # subject
labels = ['Frontal_Mid_2_L', 'Frontal_Mid_2_R',
                 'Insula_L', 'Insula_R',
                 'Cingulate_Ant_L', 'Cingulate_Ant_R', 'Cingulate_Post_L', 'Cingulate_Post_R',
                 'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L',
                 'ParaHippocampal_R', 'Amygdala_L', 'Amygdala_R',
                 'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L',
                 'Parietal_Inf_R', 'Precuneus_L', 'Precuneus_R',
                 'Thalamus_L', 'Thalamus_R']
# plv function 
def get_plv(phi1,phi2):
    dphi = np.mod(phi1-phi2,2*np.pi)
    return np.abs(np.sum(np.exp(1j*dphi)))/len(dphi)

def get_modulation(freq,runtime=25000,dt=1,t0=2000): 
    time = np.arange(0,runtime,dt)
    n = len(time)
    y = np.zeros(n)
    it0 = int(t0/dt)
    y[it0:] = np.sin(2*np.pi*freq*1e-3*(time[it0:]-t0))
    return hilbert_phase(y)[0][:-1]

current_folder = os.getcwd() 
folder_data = os.path.join(current_folder, "OzCz")

subjects = ['NEMOS_035', 'NEMOS_049', 'NEMOS_050', 'NEMOS_058', 'NEMOS_059',
            'NEMOS_064', 'NEMOS_065', 'NEMOS_071', 'NEMOS_075', 'NEMOS_077']

#data_working_point = np.loadtxt(os.path.join(current_folder,"cingulum_bundle_working_point_solution.txt"))
#data_working_point = np.genfromtxt("cingulum_bundle_working_point_solution.txt",dtype='str')
data_working_point = pd.read_csv("cingulum_bundle_working_point_solution.txt", sep="\t",index_col=None)
fex = data_working_point["fpeak_cluster"].values

nsubjects = 10
n = 21
nblocks = 3
nsims = 10
ntrials = nblocks*nsims
nnodes = 22
npairs = int( (nnodes*(nnodes-1))/2)
data_spikes = [] 
data_series = [] 

folder = os.path.join(folder_data, subjects[s])
for i in range(n):
    # data_spikes.append({})
    data_series.append({})
    for j in range(nblocks):
    #    aux1 = file_management.load_lzma(os.path.join(folder, f"data_spikes_OzCz_{s}_{i}_{j}.lzma"))
        aux2 = file_management.load_lzma(os.path.join(folder, f"series_OzCz_{s}_{i}_{j}.lzma")) 
        # for key in aux1.keys():
        #     data_spikes[-1][key] = aux1[key] 
        for key in aux2.keys():
            data_series[-1][key] = aux2[key] 

amplitude = np.linspace(0,20.0,n)
dt = 1
fs = 1e3/dt
nperseg = 4000/dt
noverlap = int(0.5*nperseg)
nfft = 2**12

phase = [ [ [] for j in range(nnodes)] for i in range(n)]

### compute PLV ### 
for i, amp in enumerate(amplitude):
    for j in range(nnodes):
        for k, trial in enumerate(range(ntrials)):
            lfp = data_series[i][f"lfp_{j}_{k}"]
            #phase_volt_list.append( bandpass_filter_and_hilbert_transform(volt, fs, 10.0, 2.0, 4) ) 
            phase[i][j].append(  bandpass_filter_and_hilbert_transform(lfp,  fs, 10.0, 2.0, 4)[-1] ) 

phase = np.array(phase)
plv_alpha_list = np.zeros((n, npairs, ntrials))
plv_alpha_matrix = np.zeros((n,nnodes, nnodes, ntrials))
for i in range(n):
    count = 0 
    for j1 in range(nnodes):
        for j2 in range(j1+1,nnodes):
            for k in range(ntrials):
                phi1 = phase[i,j1,k]
                phi2 = phase[i,j2,k]
                plv_value = get_plv(phi1,phi2)
                plv_alpha_list[i,count,k] = plv_value
                plv_alpha_matrix[i,j1,j2,k] = plv_value
                plv_alpha_matrix[i,j2,j1,k] = plv_value

            count+=1
###################################################################
columns = ["subject","mode","trial","node", "w", "fex","fpeak", "amp_fex", "amp_fpeak","amp_norm_fex", "amp_norm_fpeak", "plv_mean","plv_std"] 
           #"amp_fex_rel", "amp_fpeak_rel","amp_norm_fex_rel","amp_norm_fpeak_rel"]

rows = dict.fromkeys(columns)
for key in columns:
    rows[key] = []

fpeak_matrix = np.zeros((n,nnodes,ntrials))
power_fex_matrix = np.zeros((n,nnodes,ntrials))
power_fpeak_matrix = np.zeros((n,nnodes,ntrials))
power_norm_fex_matrix = np.zeros((n,nnodes,ntrials))
power_norm_fpeak_matrix = np.zeros((n,nnodes,ntrials))

for i, amp in enumerate(amplitude):
    for k, trial in enumerate(range(ntrials)):
        for j in range(nnodes):
            rows["mode"].append(0)
            rows["subject"].append(subjects[s])
            rows["w"].append( amp )
            rows["trial"].append( trial)
            rows["node"].append(labels[j]) 
            rows["fex"].append(fex[s])

            lfp = data_series[i][f"lfp_{j}_{k}"]
            fr, psd1  = scipy.signal.csd( lfp,  lfp,  fs=fs, window="hann", nfft=nfft, nperseg=nperseg, noverlap = noverlap)
            x1 = (lfp-lfp.mean())/lfp.std()
            fr, npsd1  = scipy.signal.csd( x1, x1, fs=fs, window="hann", nfft=nfft, nperseg=nperseg, noverlap = noverlap)
            dfr = np.diff(fr)[0]

            fpeak = fr[np.argmax(npsd1)]
            amplitude_fpeak      = np.sum(  psd1[ (fr>=fpeak-0.5)  & (fr<=fpeak+0.5) ] )*dfr 
            amplitude_fex        = np.sum(  psd1[ (fr>=fex[s]-0.5) & (fr<=fex[s]+0.5)] )*dfr
            amplitude_norm_fpeak = np.sum( npsd1[ (fr>=fpeak-0.5)  & (fr<=fpeak+0.5) ] )*dfr
            amplitude_norm_fex   = np.sum( npsd1[ (fr>=fex[s]-0.5) & (fr<=fex[s]+0.5)] )*dfr

            #phase_volt_list.append( bandpass_filter_and_hilbert_transform(volt, fs, 10.0, 2.0, 4) ) 
            #phase[i][j][k].append(  bandpass_filter_and_hilbert_transform(lfp,  fs, 10.0, 2.0, 4) ) 
            
            rows["fpeak"].append( fpeak )
            rows["amp_fex"].append( amplitude_fex )
            rows["amp_fpeak"].append( amplitude_fpeak )
            rows["amp_norm_fex"].append( amplitude_norm_fex )
            rows["amp_norm_fpeak"].append( amplitude_norm_fpeak )
            fpeak_matrix[i,j,k] = fpeak 
            power_fex_matrix[i,j,k] = amplitude_fex
            power_fpeak_matrix[i,j,k] = amplitude_fpeak
            power_norm_fex_matrix[i,j,k] = amplitude_norm_fex
            power_norm_fpeak_matrix[i,j,k] = amplitude_norm_fpeak

            rows["plv_mean"].append( np.mean(np.delete( plv_alpha_matrix[i,j,:,k], j)) )
            rows["plv_std"].append( np.std(np.delete( plv_alpha_matrix[i,j,:,k], j)) )
            
title = f"stimulation_OzCz_{subjects[s]}_nodes.txt"
dfnew = pd.DataFrame(rows, columns=columns)
dfnew.to_csv(os.path.join(folder, title),sep="\t")

columns = ["subject","mode","trial", "w", "fex", "fpeak", "fpeak_std", "plv_mean","plv_std", 
           "amp_fpeak", "amp_fpeak_std", "amp_fbaseline", "amp_fbaseline_std",
           "amp_fex",   "amp_fex_std",   "amp_norm_fex",  "amp_norm_fex_std",
           "amp_norm_fpeak", "amp_norm_fpeak_std", "amp_norm_fbaseline", "amp_norm_fbaseline_std"]

idcb = np.array([8, 9, 4, 5, 0, 15, 2, 11, 19, 3, 12, 10, 17])
idcb = np.sort(idcb)
rows = dict.fromkeys(rows)

for key in columns:
    rows[key] = [] 

for i in range(n): 
    for k in range(ntrials):
        rows["subject"].append( subjects[s] )
        rows["mode"].append( 0 )
        rows["trial"].append( k )
        rows["w"].append( amplitude[i] )
        rows["fex"].append( fex[s] )
        
        rows["fpeak"].append( np.mean( fpeak_matrix[i,idcb,k] ))
        rows["fpeak_std"].append(  np.std(  fpeak_matrix[i,idcb,k] ))
        
        rows["plv_mean"].append( np.mean( plv_alpha_list[i,idcb,k] ))
        rows["plv_std"].append(  np.std(  plv_alpha_list[i,idcb,k] ))
        
        rows["amp_fpeak"].append( np.mean( power_fpeak_matrix[i,idcb,k] ))
        rows["amp_fpeak_std"].append(   np.std( power_fpeak_matrix[i,idcb,k] ))
        rows["amp_norm_fpeak"].append( np.mean( power_norm_fpeak_matrix[i,idcb,k] ))
        rows["amp_norm_fpeak_std"].append(   np.std( power_norm_fpeak_matrix[i,idcb,k] ))
        
        rows["amp_fbaseline"].append(    np.mean( power_fpeak_matrix[0,idcb,k] ))
        rows["amp_fbaseline_std"].append( np.std( power_fpeak_matrix[0,idcb,k] ))
        rows["amp_norm_fbaseline"].append(    np.mean( power_norm_fpeak_matrix[0,idcb,k] ))
        rows["amp_norm_fbaseline_std"].append( np.std( power_norm_fpeak_matrix[0,idcb,k] ))
        
        rows["amp_fex"].append(    np.mean( power_fex_matrix[i,idcb,k] ))
        rows["amp_fex_std"].append( np.std( power_fex_matrix[i,idcb,k] ))
        rows["amp_norm_fex"].append(    np.mean( power_norm_fex_matrix[i,idcb,k] ))
        rows["amp_norm_fex_std"].append( np.std( power_norm_fex_matrix[i,idcb,k] ))

title = f"stimulation_OzCz_{subjects[s]}_cluster.txt"
dfnew2 = pd.DataFrame( rows, columns=columns)
dfnew2.to_csv(os.path.join(folder, title),  sep="\t")

# consistency: 
npairs_trial = int((ntrials*(ntrials-1))/2)
consistency_list = np.zeros((n,npairs_trial))
consistency_matrix = np.ones((n,ntrials, ntrials))
for i in range(n):
    count = 0
    for n1 in range(ntrials):
        for n2 in range(n1+1,ntrials):
            aux1 = plv_alpha_list[i,:,n1]
            aux2 = plv_alpha_list[i,:,n2]
            cons_value = np.corrcoef(aux1,aux2)[0,1]
            consistency_list[i,count] = cons_value 
            consistency_matrix[i,n1,n2] = cons_value
            consistency_matrix[i,n2,n1] = cons_value

for i, s in enumerate(subjects):
    title = f"consistency_matrix_{s}.txt"
    np.savetxt(os.path.join(folder, title), consistency_matrix[i])

    
