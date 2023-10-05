from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper,ParityMapper,QubitConverter
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit_aer.primitives import Estimator
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
import numpy as np
import pylab
import qiskit.providers
from qiskit import Aer,pulse, QuantumCircuit
from qiskit.utils import QuantumInstance, algorithm_globals
import time
import pdb
from qiskit.algorithms.exceptions import AlgorithmError


seeds = 170
algorithm_globals.random_seed = seeds
seed_transpiler = seeds
iterations = 125
shot = 4000

ultra_simplified_ala_string = """
O 0.0 0.0 0.0
H 0.45 -0.1525 -0.8454
"""

driver = PySCFDriver(
    atom=ultra_simplified_ala_string.strip(),
    basis='sto3g',
    charge=1,
    spin=0,
    unit=DistanceUnit.ANGSTROM
)
qmolecule = driver.run()

hamiltonian = qmolecule.hamiltonian
coefficients = hamiltonian.electronic_integrals
second_q_op = hamiltonian.second_q_op()

mapper = JordanWignerMapper()
converter = QubitConverter(mapper=mapper, two_qubit_reduction=False)
qubit_op = converter.convert(second_q_op)

from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver

# solver = GroundStateEigensolver(
#     JordanWignerMapper(),
#     NumPyMinimumEigensolver(),
# )
# result = solver.solve(qmolecule)
# print(result.computed_energies)
# print(result.nuclear_repulsion_energy)
# ref_value = result.computed_energies + result.nuclear_repulsion_energy
# print(ref_value)


ansatz = UCCSD(
    qmolecule.num_spatial_orbitals,
    qmolecule.num_particles,
    mapper,
    initial_state=HartreeFock(
        qmolecule.num_spatial_orbitals,
        qmolecule.num_particles,
        mapper,
    ),
)
estimator = Estimator(
    backend_options = {
        'method': 'statevector',
        'device': 'CPU'
        # 'noise_model': noise_model
    },
    run_options = {
        'shots': shot,
        'seed': seeds,
    },
    transpile_options = {
        'seed_transpiler':seed_transpiler
    }
)
def callback(eval_count, params, value, meta):
    with open('log.txt', 'a') as f:
        f.write(str(eval_count) + '\n')
        f.write(str(params) + '\n')
        f.write(str(value) + '\n')
        f.write(str(meta) + '\n')
        print(eval_count)
        print('\n')
        print(params)
        print('\n')
        print(value)
        print('\n')
        print(meta)
        print('\n')
vqe_solver = VQE(estimator, ansatz, SLSQP(maxiter=1), callback=callback)
vqe_solver.initial_point = [0.0] * ansatz.num_parameters
start_time = time.time()
calc = GroundStateEigensolver(mapper, vqe_solver)

main_operator, aux_ops = calc.get_qubit_operators(qmolecule)
parameters = [0.0] * ansatz.num_parameters
num_parameters = ansatz.num_parameters
parameters = np.reshape(parameters, (-1, num_parameters)).tolist()
batch_size = len(parameters)


test_ansatz = QuantumCircuit(12)
test_ansatz.x(0)
test_ansatz.x(1)
test_ansatz.x(2)
test_ansatz.x(3)
test_ansatz.x(6)
test_ansatz.x(7)
test_ansatz.x(8)
test_ansatz.x(9)

try:
    job = estimator.run(batch_size * [test_ansatz], batch_size * [main_operator], [])
    estimator_result = job.result()
except Exception as exc:
    raise AlgorithmError("The primitive job to evaluate the energy failed!") from exc

values = estimator_result.values

# exit()

from utils.varsaw import parseHamiltonian, group_measurements, varsaw_expectation
import pickle
# run once!
h, first_term = parseHamiltonian('Hamiltonian/OHhamiltonian.txt')
# measurements, measurement_dict = group_measurements(h)
# filehandler = open(b"142observables.obj","wb")
# pickle.dump((measurements, measurement_dict),filehandler)

filehandler = open(b"142observables.obj","rb")
measurements, measurement_dict = pickle.load(filehandler)
test_ansatz = QuantumCircuit(12, 12)
test_ansatz.x(0)
test_ansatz.x(1)
test_ansatz.x(2)
test_ansatz.x(3)
test_ansatz.x(6)
test_ansatz.x(7)
test_ansatz.x(8)
test_ansatz.x(9)
print(varsaw_expectation(test_ansatz, measurements, measurement_dict, first_term, h))
test_ansatz = QuantumCircuit(12, 12)
varsaw_expectation(test_ansatz, measurements, measurement_dict, first_term, h)
