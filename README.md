## quafu-chemistry

**quafu-chemistry** is a simple VQE program based on PyQuafu (https://github.com/ScQ-Cloud/pyquafu) and Quafu quantum cloud computing cluster (http://quafu.baqis.ac.cn/) to compute the ground state of molecules.

## Libraries Dependency

* pyquafu

* openfermion

* pyscf

* openfermionpyscf

All the libraries can be install with:

```bash
pip install XXX
```

## Examples

A simple example for the hydrogen molecule (H2) can be found in the folder **example**. 

H2_simulate.ipynb uses simulator of quafu, and H2_quafu.ipynb use real quantum computer via cloud.
