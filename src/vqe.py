from hamiltonian import *
from spsa import *
from optimizers import *
from ansatz import *

import numpy as np
import math
from quafu import User, Task
import matplotlib.pyplot as plt
from quafu import simulate


def submit(qc, token_ip, shots=10000):
    user = User()
    user.save_apitoken(token_ip)
    backend = 'Dongling'
    shots = shots
    task = Task()
    task.config(backend=backend, shots=shots, compile=True)
    wait=True
    qc_result = task.send(qc, wait=wait)
    sampling = qc_result.res
    return sampling

def measure_single(circuit, term, cbits):
    m = []
    for i in range(len(term)):
        if term[i][0] == 'Z':
            m = m + [int(term[i][1:])]
        elif term[i][0] == 'X':
            iqubit = eval(term[i][1:])
            m = m + [iqubit]
            circuit.h(iqubit) 
        elif term[i][0] == 'Y':
            iqubit = eval(term[i][1:])
            m = m + [iqubit]
            #circuit.sdg(iqubit)
            circuit.rz(iqubit, np.pi/2)
            circuit.rz(iqubit, np.pi/2)
            circuit.rz(iqubit, np.pi/2)
            circuit.h(iqubit)
    circuit.measure(cbits, cbits=cbits)
    return circuit

def get_energy(data, qham_data):
    energy = 0.
    #nqubits = int(math.log(len(data[0]), 2))
    #k = [f'%0{nqubits}d'%eval(bin(i)[2:]) for i in range(2**nqubits)] # [00, 01, 10, 11]
    for i in range(len(qham_data)):
        nqubits = int(math.log(len(data[i]), 2))
        k = [f'%0{nqubits}d'%eval(bin(i)[2:]) for i in range(2**nqubits)] # [00, 01, 10, 11]
        #k = [f'%0{nqubits}d'%eval(bin(i)[2:]) for i in range(2**nqubits)]
        #print(k)
        for j in range(1, len(qham_data[i]), 2):
            tmp = np.array([1]*2**nqubits)
            for t in range(len(qham_data[i][j])):
                iqubit = eval(qham_data[i][j][t][1:])
                for n in range(len(k)):
                    if k[n][iqubit] == '1':
                        tmp[n] = -tmp[n]
            #print(tmp)
            energy += qham_data[i][j-1] * sum(np.array(data[i]) * tmp)
    return energy


class vqe:
    def __init__(self, info, nlayer=1, hea_type='ry_cascade', token_ip=None, opt='Adam', simulator=False, shots=10000, etol=1e-5, maxiter='default', method='parameter_shift', 
                 grouping='QWC', mapping='jordan_wigner',reduce_two_qubit=False):
        geometry, basis, multiplicity, charge = info
        molecule = MolecularData(geometry, basis, multiplicity, charge)
        molecule.filename = "./" + molecule.filename.split("/")[-1]
        self.mol = run_pyscf(molecule, run_scf=True, run_mp2=False, run_cisd=False, run_ccsd=False, run_fci=True)

        self.nlayer = nlayer
        self.token_ip = token_ip
        self.shots = shots
        self.hea_type = hea_type
        self.mapping = mapping
        self.reduce_two_qubit = reduce_two_qubit
        self.qham = get_qham_list(self.mol, self.mapping, self.reduce_two_qubit)
        
        if self.reduce_two_qubit:
            self.nqubits = self.mol.n_qubits - 2
        else:
            self.nqubits = self.mol.n_qubits
        self.grouping = grouping
        self.electrons = self.mol.n_electrons
        self.simulator = simulator
        self.cbits = list(np.arange(self.nqubits))
        self.etol = etol
        self.method = method
        
        nparams = get_hea_nparams(self.nqubits, self.nlayer, self.hea_type)
        self.params = np.random.uniform(-2*np.pi, 2*np.pi, nparams)
        
        if self.grouping == 'QWC':
            self.qham_measure, self.num_groups, self.groups, self.para_dict, self.ecore = group_hamiltonian(self.qham)
            self.qham_data = get_qham_data(self.qham, self.qham_measure)
            
        if type(maxiter) == int:
            self.maxiter = maxiter
        else:
            self.maxiter = len(self.params) * 20
            
        if opt == 'Adam':
            self.opt = Adam()
        elif opt == 'RMSProp':
            self.opt = RMSProp()
        else:
            self.opt = SGD()
        
    def parameter_shift(self):
        amps_grad = np.zeros_like(self.params)
        for i in range(len(self.params)):
            amps_plus, amps_minus = self.params.copy(), self.params.copy()
            amps_plus[i] += np.pi/2
            amps_minus[i] -= np.pi/2
            value_plus = self.measure(amps_plus)
            value_minus = self.measure(amps_minus)
            amps_grad[i] += 0.5 * (value_plus - value_minus)
        if self.opt:
            self.params = self.opt.run(np.array(self.params), np.array(amps_grad))
        else:
            self.params = np.array(self.params) - np.array(amps_grad)
        return self.params, amps_grad, max(abs(amps_grad))
        
        
    def measure(self, amps=[]):
        def correct_samples(nqubits, samples):
            k = [f'%0{nqubits}d'%eval(bin(i)[2:]) for i in range(2**nqubits)]
            new_s = {}
            for term in k:
                if term in samples.keys():
                    new_s[f"{term}"] =  samples[f"{term}"]
                else:
                    new_s[f"{term}"] = 0
            return new_s
        
        data = []
        for i in range(len(self.qham_measure)): 
            if len(amps):
                self.qc = get_hea_ansatz(self.nqubits, self.nlayer, self.electrons, amps, self.hea_type)
            else: 
                self.qc = get_hea_ansatz(self.nqubits, self.nlayer, self.electrons, self.params, self.hea_type)
            c = measure_single(self.qc, self.qham_measure[i], self.cbits)
            if self.simulator:
                samples = simulate(c)
                data.append(samples.probabilities)
            else:
                samples = submit(self.qc, self.token_ip, self.shots)
                samples = correct_samples(self.nqubits, samples)
                tmp = []
                for key in samples.keys():
                    tmp.append(samples[key]/self.shots)
                data.append(tmp)
        return get_energy(data, self.qham_data) + self.ecore
    
    def run(self):
        energy_recoder = []
        grad_recoder = []
        grad_max_recoder = []
        amps_recoder = []
        if self.method == 'parameter_shift':
            for i in range(self.maxiter):
                e = self.measure()
                print(f'{i} iteration, energy: {e:.10f}, error with FCI: {abs(e-self.mol.fci_energy):.10f}, ||g_max||: {grad_max_recoder[-1]:.10f}')
                if abs(e - self.mol.fci_energy) < self.etol:
                    print('Success')
                    break
                energy_recoder.append(e)
                amps_recoder.append(self.params)
                self.params, amps_grad, grad_max  = self.parameter_shift()
                grad_recoder.append(amps_grad)
                grad_max_recoder.append(grad_max)
                if grad_max < 1e-12:
                    print('Grad is zero, STOP!)')
                    break
        elif self.method == 'SPSA':
            spsa = SPSA(self.measure, self.maxiter,second_order=False)
            self.params, energy = spsa.run(self.params)
            energy_recoder = energy
            print(abs(energy_recoder[-1]-self.mol.fci_energy))
        elif self.method == '2SPSA':
            spsa = SPSA(self.measure, self.maxiter, second_order=True)
            self.params, energy = spsa.run(self.params)
            energy_recoder = energy
            print(abs(energy_recoder[-1]-self.mol.fci_energy))
                
        elif self.method in ['COBYLA']:
            def callback_fn(amps):
                e = self.measure(amps)
                energy_recoder.append(e)
            print(f"Calculating use scipy.optimize.minimize.{self.method} method")
            res = scipy.optimize.minimize(self.measure, self.params, method=f'{self.method}', tol=self.etol, options={'maxiter':self.maxiter}, callback=callback_fn)
            print(res.fun)
            #e = self.measure(amps=res.x)
            #print(e)
            #error = abs(e - self.mol.fci_energy)
        return energy_recoder, amps_recoder, grad_recoder, grad_max_recoder
    


    
if __name__ == '__main__':
    basis = "sto-3g"
    multiplicity = 0
    charge = 0
    geometry = [("H", [0.0, 0.0, 0]), ("H", [0.0, 0.0, 0.74])]
    info = [geometry, basis, multiplicity, charge]
    vqe = vqe(info, reduce_two_qubit=False)
    ene_recoder, amps_recoder, gard_recoder, _ = vqe.run()
    print(ene_recoder)
