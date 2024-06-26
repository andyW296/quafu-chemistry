from quafu import QuantumCircuit


def get_hea_nparams(nqubits,nlayer,hea_type):
    if hea_type == 'ry_full' or hea_type == 'ry_linear':
        return nqubits*(nlayer+1)
    elif hea_type == 'ry_cascade':
        return nqubits*(2*nlayer+1)

def ry_cascade(nqubits, nlayer, electrons, params):
    c = QuantumCircuit(nqubits)
    for i in range(electrons):
        c.x(i)
    count = 0
    for i in range(nqubits):
        c.ry(i, params[count])
        count += 1
    for _ in range(1, nlayer+1):
        for i in range(nqubits-1):
            c.cnot(i, i+1)
        for i in range(nqubits):
            c.ry(i, params[count])
            count += 1
        for i in range(nqubits-1, 0, -1):
            c.cnot(i-1, i)
        for i in range(nqubits):
            c.ry(i, params[count])
            count += 1
    return c


def get_hea_ansatz(nqubits, nlayer, electrons, params, hea_type):
    if hea_type == 'ry_cascade':
        return ry_cascade(nqubits, nlayer, electrons, params)
