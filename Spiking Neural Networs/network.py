'''############################################################################
#                             NETWORK CLASS                                   #
############################################################################### 

# Update 25/01/2022: heterogeneity in nodes 

# Update 02/02/2022: model change to iaf_alpha_multisynapse to be able 
                     to model NMDA synapses 

# Update 07/02/2022: improvement of the abstraction of the class  


# Receptor types: 
# --------------
# 1: AMPA synapses among neurons of the same population
# 2: GABA synapses among neurons of the same population 
# 3: AMPA synapses noise background
# 4: AMPA synpases among neurons of different populations
# 5: NMDA synapses among neurons of different populations 
# 6: After-hyperpolarization calcium-dependant current, it is not a synapse
# but Ca concentration increase after each spike

############################################################################'''

import numpy as np
import copy
import nest
nest.print_time = True

def rng(x,y,size=None, dim=10**6):
    # for the generation of electrical field per node.
    samples=[]
    nLoop=0
    dx = np.diff(x)[0]

    xaux = np.arange(len(x)) # I will select the indexes
    while len(samples)<size:
        idx = np.random.choice(xaux,dim)
        x_ = x[idx]
        prop = y[idx]*dx
        idx_in = np.where(np.random.uniform(0,1,dim) <= prop)[0]
        samples +=  list(x_[idx_in])
        print(len(samples))

    if len(samples)>size:
        samples = samples[:-(len(samples)-size)]

    return samples

class Network:
    def __init__(self,nnodes, weights, lengths, coupling_factor, velocity, connectivity,seed,
                 node_currents, rate=2400.0, ge=0.012, gnoise = 0.0696, runtime=6000.0, time_cut=1000.0, 
                 NMDA=False,all_to_all=True):
        '''
        Parameters
        ----------
        nnodes : integer
            Number of populations/nodes of the network
        weights : array
            Weight matrix of the network
        lengths : arry
            Tract length matrix of the network
        coupling_factor : float
            Amplitude factor 
        velocity : float
            Propagation speed of the synapses 
        connectivity : array
            Connectivity matrix
        seed : integer
            Seed to initiallize the rng
        node_currents : array
            Additional constant current injection in each population/node
        rate : float, optional
           Firing rate poisson background noise. The default is 800.0.
        ge : float, optional
            Conductance between excitatory-excitatory connections. The default is 0.12.
        gnoise : float, optional
            Background noise excitatory conductance. The default is 0.1.
        runtime : float, optional
            Time simulation in ms. The default is 6000.0.
        time_cut : float, optional
            Time from which the simulation data is saved. The default is 1000.0.
        NMDA : boolean, optional
            Inclusion of NMDA synpases. The default is False.
        all_to_all : boolean, optional
            All-to-all intra-conectivity fashion. The default is False.

        Returns
        -------
        None.

        '''

        self.nnodes = nnodes                    # number of nodes of the network
        self.coupling_factor = coupling_factor  # coupling factor 
        self.velocity = velocity                # propagation speed
        self.weights = weights                  # weight matrix
        self.lengths = lengths                  # tract lengths matrix
        self.connectivity = connectivity        # connectivity matrix
        self.seed = seed                        # seed for the random number generator
        self.node_currents = node_currents      # additional constant current injection in each population/node
        self.rate = rate                        # firing rate poisson background noise
        self.gnoise = gnoise                    # background noise excitatory conductance
        self.gee = ge                           # conductance between excitatory-excitatory connections 
        self.gei =  2.0*self.gee                # conductance between excitatory-inhibitory connections 
        self.gie =  4.0*self.gee                # conductance between inhibitory-excitatory connections 
        self.gii =  4.0*self.gee                # conductance between inhibitory-inhibitory connections 
        self.gNMDA = 0.0*self.gee               # NMDA conductance (only among populations/nodes)
        self.NMDA = NMDA                        # to consider NMDA synapses or not
        self.all_to_all = all_to_all            # to consider all-to-all intra-population/node connectivity
        self.ne = 80                            # number of excitatory units in each population/node
        self.ni = 20                            # number of inhibitory units in each population/node
        self.C_m = [104.0, 59.0]                # membrane capacitance E,I 
        self.g_L = [4.3,2.9]                    # leak conductance E,I
        self.E_L = [-65.0,-62.0]                # leak reversal potential E,I
        self.V_th = [-52.0, -42.0]              # spike initiation threshold E,I
        self.Delta_T = [0.8, 3.0]               # slope factor E,I
        self.tau_w = [88.0, 16.0]               # adaptation time constnat E,I
        self.V_reset = [-53.0, -54.0]           # membrane potential reset value after spike E,I
        self.a = [-0.8, 1.8]                    # subthreshold adaptation E,I 
        self.b = [65.0,61.0]                    # spike-triggered adaptation E,I
        self.I_e = [20.0, 30.0] #[47.0,42.0] # [0.0,0.0]      # constant current injection in each neuron/unit
        if self.all_to_all:                     # to select all-to-all intra-population/node connectitivity
            #self.I_e = [79.0,74.0]
            self.I_e = [20.0-5,30.0-5]

        self.tau_syn_AMPA = 3.0                 # rise time of fast excitatory synaptic conductance
        self.tau_syn_GABA = 3.2                 # rise time of fast inhibitory synpatic conductance
        self.tau_syn_NMDA = 60.0                # rise time of slow excitatory synaptic conductance
        self.E_rev_AMPA = 0.0                   # fast excitatory reversal potential 
        self.E_rev_GABA = -85.0                 # fast inhibitory reversal potential 
        self.E_rev_NMDA = 0.0                   # slow excitatory reversal potential 
        self.tau_Ca = 20.0                      # rise time of the calcium concentration
        self.E_rev_K = -90.0                    # after-hyperpolarization current reversal potential
        self.gca = 0.0                          # calcium-dependant AHP conductance 
        
        self.resolution = 0.1                   # resolution of the simulation dt in ms
        self.runtime  = runtime                 # time simulation in ms
        self.interval = 1.0                     # time resolution  
        self.time_cut = time_cut                # 

        # model parameters of excitatory neurons/units: 
        self.params_ex={'C_m': self.C_m[0], 'g_L': self.g_L[0], 'E_L': self.E_L[0],
                       'V_th': self.V_th[0], 'Delta_T': self.Delta_T[0], 'a': self.a[0],
                       'tau_w': self.tau_w[0], 'b': self.b[0], 'V_reset': self.V_reset[0],
                       'I_e': self.I_e[0], 't_ref': 2.0,
                       'tau_syn': [self.tau_syn_AMPA, self.tau_syn_GABA, self.tau_syn_AMPA, self.tau_syn_AMPA, self.tau_syn_NMDA, self.tau_Ca], 
                       'E_rev':   [self.E_rev_AMPA,   self.E_rev_GABA,   self.E_rev_AMPA,   self.E_rev_AMPA,   self.E_rev_NMDA, self.E_rev_K ] }
        
        # model parameters of inhibitory neurons/units:
        self.params_in={'C_m': self.C_m[1], 'g_L': self.g_L[1], 'E_L': self.E_L[1],
                       'V_th': self.V_th[1], 'Delta_T': self.Delta_T[1], 'a': self.a[1],
                       'tau_w': self.tau_w[1], 'b': self.b[1], 'V_reset': self.V_reset[1],
                       'I_e': self.I_e[1], 't_ref': 2.0,
                       'tau_syn': [self.tau_syn_AMPA, self.tau_syn_GABA, self.tau_syn_AMPA, self.tau_syn_AMPA, self.tau_syn_NMDA, self.tau_Ca ], 
                       'E_rev':   [self.E_rev_AMPA,   self.E_rev_GABA,   self.E_rev_AMPA,   self.E_rev_AMPA,   self.E_rev_NMDA, self.E_rev_K ] }

        
        self.excitatory = [] # list of excitatory subpopulations
        self.inhibitory = [] # list of inhibitory subpopulations 
        self.noise_ex   = [] # list of poisson background noise for excitatory subpopulations
        self.noise_in   = [] # list of poisson background noise for inhibitory subpipulations 
        
        # Reset nest kernel, setting the number of threads and the seed for rng 
        nest.ResetKernel()
        nest.SetKernelStatus({"local_num_threads": 4})
        msd = self.seed 
        n_vp = nest.GetKernelStatus('total_num_virtual_procs')
        msdrange1 = range(msd, msd+n_vp)
        pyrngs = [ np.random.RandomState(s) for s in msdrange1 ]
        msdrange2 = range(msd+n_vp+1, msd+n_vp*2+1 )
        nest.SetKernelStatus({ "grng_seed": msd+n_vp,"rng_seeds" : msdrange2, "resolution": self.resolution,  })

        self.create_nodes()          # to build the populations/nodes
        self.set_internal_synapses() # intra-population connectivity 
        self.set_external_synapses() # inter-population connectivity 
       
    def create_nodes(self):
        '''
        Description
        -----------
        Function that creates the populations/nodes.
        Each node consists of ne excitatory and ni inhibitory neurons. 
        Heterogeinity was added in the paramters.

        Returns
        -------
        None.

        '''

        self.params_excitatory=[] # list of excitatory parameters per each population/node 
        self.params_inhibitory=[] # list of inhibitory parameters per each population/node

        #np.random.seed(12345) #
        sg=0.01
        nkeys = len(self.params_ex.keys())
        rn = np.random.normal(0,1,self.ne*nkeys)
        k = 0
        
        for i in range(self.ne):
            self.params_excitatory.append(dict.fromkeys(self.params_ex.keys()))
            for key in list(self.params_ex.keys())[:11]:
                self.params_excitatory[-1][key] = self.params_ex[key]*(1.0+sg*rn[k])
                k+=1
            self.params_excitatory[-1]["E_rev"]   = self.params_ex["E_rev"]
            self.params_excitatory[-1]["tau_syn"] = self.params_ex["tau_syn"]

        nkeys = len(self.params_in.keys())
        rn = np.random.normal(0,1,self.ni*nkeys)
        k = 0
        for i in range(self.ni):
            self.params_inhibitory.append(dict.fromkeys(self.params_in.keys()))
            for key in list(self.params_in.keys())[:11]:
                self.params_inhibitory[-1][key] = self.params_in[key]*(1.0+sg*rn[k])
                k+=1
            self.params_inhibitory[-1]["E_rev"]   = self.params_in["E_rev"]
            self.params_inhibitory[-1]["tau_syn"] = self.params_in["tau_syn"]
            
        params_ex = [] 
        params_in = []
        for i in range(self.nnodes):
            self.excitatory.append(nest.Create("aeif_cond_alpha_multisynapse",self.ne))
            self.inhibitory.append(nest.Create("aeif_cond_alpha_multisynapse",self.ni))

            params_ex.append(copy.deepcopy(self.params_excitatory))
            params_in.append(copy.deepcopy(self.params_inhibitory))
            for ii in range(self.ne):
                params_ex[-1][ii]["I_e"] += self.node_currents[i]
            for ii in range(self.ni):
                params_in[-1][ii]["I_e"] += self.node_currents[i]
            nest.SetStatus(self.excitatory[-1], params = params_ex[-1]) # self.params_excitatory)
            nest.SetStatus(self.inhibitory[-1], params = params_in[-1]) # self.params_inhibitory)

    def set_injected_signal(self, weights, frequency):
        '''
        Description
        -----------
        Injection of a modulation signal (sine function). 
        Only in excitatory neurons
        
        Parameters
        ----------
        weights : numpy.ndarray
            amplitude factor of the signal per node 
        frequency : float
            oscillatory frequency of the modulation signal

        Returns
        -------
        None.
        '''
        
        self.signal = []
        for i in range(self.nnodes):
            signalparams = {'amplitude': weights[i], 'frequency': frequency}
            self.signal.append(nest.Create('ac_generator'))
            nest.SetStatus(self.signal[-1], signalparams)
            nest.Connect(self.signal[-1],self.excitatory[i])
    
    def set_injected_currents_from_densities(self, weights, frequency, tstart=2000.0): 
        '''
        Description
        ----------- 
        Injection of a modulation signal (sine function) with a distribution of ampltidudes.
        Only in excitatory neurons.
        
        Parameters
        ----------
        densities : dictionary 
            statistical density of electric field's amplitude
        frequency : float
            oscillatory freqquency of the modulation signal

        Returns
        -------
        None.

        '''
        self.signal = [] 
        for i in range(self.nnodes): 
            #x, y = densities[f"x_{i}"], densities[f"y_{i}"]
            #weights = rng(x,y,self.ne)
            for j in range(self.ne):
                #signalparams = {"amplitude": weights[j], "frequency": frequency}
                self.signal.append(nest.Create('ac_generator'))
                signalparams = {'amplitude': weights[i,j], 'frequency': frequency, "offset": 0.0, "phase": 0.0, "start": tstart} # importante solo para el stimwfit
                nest.SetStatus(self.signal[-1],signalparams)
                nest.Connect(self.signal[-1], [self.excitatory[i][j]])

            
    def set_internal_synapses(self, prob=0.1):
        '''
        Description
        ----------- 
        Setting of the intra-population connections
        
        Parameters
        ----------
        prob : float, optional
            connection probability among neurons of the same population/node. The default is 0.1.

        Returns
        -------
        None.

        '''
        
        self.nee = int(self.ne*prob) # Number of excitatory-excitatory connections
        self.nei = int(self.ne*prob) # Number of excitatory-inhibitory connections
        self.nie = int(self.ni*prob) # Number of inhibitory-excitatory connections
        self.nii = int(self.ni*prob) # Number of inhibitory-inhibitory connections

        for i in range(self.nnodes):
            self.noise_ex.append(nest.Create('poisson_generator', self.ne)) # poisson generator for each excitatory subpopulation
            self.noise_in.append(nest.Create('poisson_generator', self.ni)) # poisson generator for each inhibitory subpopulation 

            nest.SetStatus(self.noise_ex[-1], {'rate' : self.rate}) 
            nest.SetStatus(self.noise_in[-1], {'rate' : self.rate})

            nest.Connect(self.noise_ex[-1], self.excitatory[i], conn_spec='one_to_one', syn_spec = {'weight': self.gnoise, 'receptor_type': 3 }) # synapses background noise -> excitatory
            nest.Connect(self.noise_in[-1], self.inhibitory[i], conn_spec='one_to_one', syn_spec = {'weight': self.gnoise, 'receptor_type': 3 }) # synapses background noise -> inhibitory
            
            # AHP-calcium dependent current not well implemented yet
            #nest.Connect(self.excitatory[i],self.excitatory[i], conn_spec='one_to_one', syn_spec = {'weight': self.gca, 'receptor_type': 6})  # AHP calcium-dependant current in excitatory neurons
            #nest.Connect(self.inhibitory[i],self.inhibitory[i], conn_spec='one_to_one', syn_spec = {'weight': self.gca, 'receptor_type': 6})  # AHP calcium-dependant current in inhibtory neurons
            
            if self.all_to_all: # all-to-all connectivity fashion
                nest.Connect(self.excitatory[i], self.excitatory[i], syn_spec={'weight': self.gee, 'delay': 0.5, 'receptor_type': 1})  # synapses excitatory -> excitatory
                nest.Connect(self.excitatory[i], self.inhibitory[i], syn_spec={'weight': self.gei, 'delay': 0.5, 'receptor_type': 1})  # synapses excitatory -> excitatory
                nest.Connect(self.inhibitory[i], self.excitatory[i], syn_spec={'weight': self.gie, 'delay': 0.5, 'receptor_type': 2})  # synapses inhibitory -> excitatory
                nest.Connect(self.inhibitory[i], self.inhibitory[i], syn_spec={'weight': self.gii, 'delay': 0.5, 'receptor_type': 2})  # synapses inhibitory -> inhibitory
            else: # cpnnectivity of prob*100 %
                nest.Connect(self.excitatory[i], self.excitatory[i], conn_spec={'rule': 'fixed_indegree', 'indegree': self.nee }, syn_spec={'weight': self.gee, 'delay': 0.5, 'receptor_type': 1}) # synapses excitatory -> excitatory
                nest.Connect(self.excitatory[i], self.inhibitory[i], conn_spec={'rule': 'fixed_indegree', 'indegree': self.nei }, syn_spec={'weight': self.gei, 'delay': 0.5, 'receptor_type': 1}) # synapses excitatory -> excitatory
                nest.Connect(self.inhibitory[i], self.excitatory[i], conn_spec={'rule': 'fixed_indegree', 'indegree': self.nie }, syn_spec={'weight': self.gie, 'delay': 0.5, 'receptor_type': 2}) # synapses inhibitory -> excitatory
                nest.Connect(self.inhibitory[i], self.inhibitory[i], conn_spec={'rule': 'fixed_indegree', 'indegree': self.nii }, syn_spec={'weight': self.gii, 'delay': 0.5, 'receptor_type': 2}) # synapses inhibitory -> inhibitory         
                
                
    def set_synapse(self, source, target, weight, delay, receptor_type, prob=0.05): 
        '''
        Description
        -----------
        Setting the synapses between neurons of different populations/nodes.ç
        Modified to be able to work with multisynapses

        Parameters
        ----------
        source : TYPE
            pre-synaptic neurons
        target : TYPE
            post-synaptic neurons
        weight : float
            post-synaptic neuron's conductance
        delay : float
            synpatic delay
        receptor_type : integer
            type of synapse 
        prob : float, optional
            connection probability among neurons of different populations/nodes. The default is 0.05.

        Returns
        -------
        None.

        '''
    
        n=int(self.ne*prob)
        syn_spec = {'weight': weight, 'delay': delay, 'receptor_type': receptor_type}
        conn_spec = {'rule': 'fixed_indegree', 'indegree': n}
        nest.Connect(source, target, syn_spec = syn_spec, conn_spec = conn_spec)
  
    def set_external_synapses(self):
        '''
        Description
        -----------
        Setting all the external synpases.
        NMDA synapses should be modified. If NMDA is included, same synpases should release AMPA and NMDA neurotransmitters (?). 

        Returns
        -------
        None.

        '''
        
        conn = np.where(self.connectivity==1)
        for _ in range(np.shape(conn)[1]):
            i,j = conn[0][_], conn[1][_]
            weight = self.weights[i,j]*self.coupling_factor
            delay  = np.round(self.lengths[i,j]/self.velocity,1)
            if delay < self.resolution:
                delay = self.resolution
            
            if self.NMDA:
                self.set_synapse(self.excitatory[j], self.excitatory[i], weight=weight,     delay=delay, receptor_type = 4, prob = 0.025) # AMPA
                self.set_synapse(self.excitatory[j], self.excitatory[i], weight=0.2*weight, delay=delay, receptor_type = 5, prob = 0.025) # NMDA
            else: 
                self.set_synapse(self.excitatory[j], self.excitatory[i], weight=weight, delay=delay, receptor_type = 4) # AMPA


    def set_multimeters(self):
        '''
        Description
        -----------
        Setting the multimeters to record the variables of interset.

        Returns
        -------
        None.

        '''
        self.multimeters=[]
        params = {'interval': self.interval, 'withtime': True, 'record_from': ['V_m','g_1','g_2','g_3','g_4','g_5'] }
        for i in range(self.nnodes):
            self.multimeters.append( nest.Create('multimeter', params=params))
            nest.Connect(self.multimeters[-1], self.excitatory[i])

    def set_spike_detectors(self):
        '''
        Description
        -----------
        Setting the spike detector to record the spike events.

        Returns
        -------
        None.

        '''
        self.spike_detectors = []
        for i in range(self.nnodes):
            self.spike_detectors.append(nest.Create('spike_detector'))
            nest.Connect(self.excitatory[i], self.spike_detectors[-1])

    def run_simulation(self):
        '''
        Running the simulation.

        Returns
        -------
        None.

        '''
        nest.Simulate(self.runtime)

    def get_data(self,save_currents=False):
        '''
        Description
        -----------
        Getting the membrane potential, the local field potential (LFP) and the incoming 
        currents of each neuron. 
        LFP is computed as the mean of the absolute value of each current

        Parameters
        ----------
        save_currents : boolean, optional
            Save or not save the incoming currents during the simulation. The default is False.

        Returns
        -------
        Returns
        -------
        times : array
            time sequence of the simulation 
        volts : array
            average membrane potential of each population/node 
        lfps: 
            local filed potential of each population/node 
        currents: 
            incoming currents of each population/node
        '''

        currents = [ [] for i in range(self.nnodes)] 
        volts = [ [] for i in range(self.nnodes)]
        lfps  = [ [] for i in range(self.nnodes)]

        naux = self.ne+self.ni
        for i in range(self.nnodes):
            currents[i] = []
            dmm    = nest.GetStatus(self.multimeters[i])[0]["events"]
            neuron = dmm['senders']
            time   = dmm['times']
            vm     = dmm['V_m']
            gAMPA_internal  = dmm['g_1']
            gGABA_internal  = dmm['g_2']
            gAMPA_noise     = dmm['g_3']
            gAMPA_external  = dmm['g_4']
            gNMDA_external  = dmm['g_5']

            time_   = [ [] for j in range(self.ne)]
            vm_     = [ [] for j in range(self.ne)]
            gAMPA_internal_  = [ [] for j in range(self.ne)]
            gGABA_internal_  = [ [] for j in range(self.ne)]
            gAMPA_noise_     = [ [] for j in range(self.ne)]
            gAMPA_external_  = [ [] for j in range(self.ne)]
            gNMDA_external_  = [ [] for j in range(self.ne)]

            for j in range(self.ne):
                sender    = np.where(neuron==naux*i+j+1)
                time_[j]  = time[sender]
                vm_[j]    = vm[sender]
                gAMPA_internal_[j] = gAMPA_internal[sender]
                gGABA_internal_[j] = gGABA_internal[sender]
                gAMPA_noise_[j]    = gAMPA_noise[sender]
                gAMPA_external_[j] = gAMPA_external[sender]
                gNMDA_external_[j] = gNMDA_external[sender]

            time_= np.array(time_)
            vm_  = np.array(vm_)
            gAMPA_internal_ = np.array(gAMPA_internal_)
            gGABA_internal_ = np.array(gGABA_internal_)
            gAMPA_noise_    = np.array(gAMPA_noise_)
            gAMPA_external_ = np.array(gAMPA_external_)
            gNMDA_external_ = np.array(gNMDA_external_)

            I_AMPA_internal = np.mean(np.abs(gAMPA_internal_*(vm_-self.E_rev_AMPA)),axis=0)
            I_GABA_internal = np.mean(np.abs(gGABA_internal_*(vm_-self.E_rev_GABA)),axis=0)
            I_AMPA_noise    = np.mean(np.abs(gAMPA_noise_*(vm_-self.E_rev_AMPA)),axis=0)
            I_AMPA_external = np.mean(np.abs(gAMPA_external_*(vm_-self.E_rev_AMPA)),axis=0)
            I_NMDA_external = np.mean(np.abs(gNMDA_external_*(vm_-self.E_rev_NMDA)),axis=0)
            
            if save_currents:
                currents[i].append(I_AMPA_internal)
                currents[i].append(I_GABA_internal)
                currents[i].append(I_AMPA_noise)
                currents[i].append(I_AMPA_external)
                currents[i].append(I_NMDA_external)
            
            average_voltage = np.mean(vm_, axis=0)

            lfps[i]  = (I_AMPA_internal + I_GABA_internal + I_AMPA_noise + I_AMPA_external + I_NMDA_external )[time_[0]>=self.time_cut]
            volts[i] = average_voltage[time_[0]>=self.time_cut]

        t = time_[0][time_[0]>=self.time_cut]
        
        
        time = t
        volts = np.array(volts)
        lfps  = np.array(lfps)
        if save_currents:
            currents = np.array(currents)
            return time, volts, lfps, currents
        else:
            return time, volts, lfps

    def get_spikes(self):
        '''
        Description
        -----------
        Getting the spikes events of the excitatory neurons of the whole network

        Returns
        -------
        times : list
            list of "nnodes" arrays with the spiking times
        spikes : list
            list of "nnodes" arrays with the labels of the firing neurons

        '''
        times, spikes = [],[]
        for i in range(self.nnodes):
            dmm = nest.GetStatus(self.spike_detectors[i])[0]['events']
            t   = np.round(dmm['times'],2)
            s   = dmm['senders']
            times.append( t[t>=self.time_cut] )
            spikes.append( s[t>=self.time_cut]  )
            times[-1] = np.array(times[-1])
            spikes[-1] = np.array(spikes[-1])

        return times, spikes

    def get_spikes_new(self): 
        '''
        Description
        -----------
        Getting the spikes events of the excitatory neurons of the whole network.
        New version of 18/11/2021

        Returns
        -------
        times : list
            list of "nnodes" arrays with the spiking times
        spikes : list
            list of "nnodes" arrays with the labels of the firing neurons
        '''
        times, spikes = [],[]
        for i in range(self.nnodes):
            dmm = nest.GetStatus(self.spike_detectors[i])[0]['events']
            t   = np.round(dmm['times'], 2)
            s   = dmm['senders']
            times.append( t[t>=self.time_cut] )
            spikes.append( s[t>=self.time_cut]  )
            times[-1] = np.array(times[-1])
            spikes[-1] = np.array(spikes[-1])

        times = np.concatenate(times)
        spikes = np.concatenate(spikes)

        return times, spikes