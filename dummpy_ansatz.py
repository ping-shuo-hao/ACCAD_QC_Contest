from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper,ParityMapper,QubitConverter
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit_aer.primitives import Estimator
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
import numpy as np
from qiskit import QuantumCircuit
from qiskit.utils import algorithm_globals
from qiskit.algorithms.exceptions import AlgorithmError
import pdb

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

from qiskit_nature.second_q.algorithms import GroundStateEigensolver


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

vqe_solver = VQE(estimator, ansatz, SLSQP(maxiter=1))
vqe_solver.initial_point = [0.0] * ansatz.num_parameters

calc = GroundStateEigensolver(mapper, vqe_solver)

main_operator, aux_ops = calc.get_qubit_operators(qmolecule)
parameters = [0.0] * ansatz.num_parameters
num_parameters = ansatz.num_parameters
parameters = np.reshape(parameters, (-1, num_parameters)).tolist()
batch_size = len(parameters)


dummy_ansatz = QuantumCircuit(12)
dummy_ansatz.x(0)
dummy_ansatz.x(1)
dummy_ansatz.x(2)
dummy_ansatz.x(3)
dummy_ansatz.x(6)
dummy_ansatz.x(7)
dummy_ansatz.x(8)
dummy_ansatz.x(9)

# pdb.set_trace()

try:
    job = estimator.run(batch_size * [dummy_ansatz], batch_size * [main_operator], [])
    estimator_result = job.result()
except Exception as exc:
    raise AlgorithmError("The primitive job to evaluate the energy failed!") from exc

values = estimator_result.values

print(values)