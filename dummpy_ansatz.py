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
from qiskit.providers.fake_provider import *
import pickle
from qiskit.providers.aer.noise import NoiseModel
import qiskit.providers.aer.noise as noise
from utils.varsaw import parseHamiltonian, group_measurements, varsaw_expectation
from qiskit_aer.primitives import Sampler

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

with open('NoiseModel/fakekolkata.pkl', 'rb') as file:
    noise_model_kolkata = pickle.load(file)
with open('NoiseModel/fakecairo.pkl', 'rb') as file:
    noise_model_cairo = pickle.load(file)
with open('NoiseModel/fakemontreal.pkl', 'rb') as file:
    noise_model_montreal = pickle.load(file)
noise_model1 = noise.NoiseModel()
noise_model2= noise.NoiseModel()
noise_model3 = noise.NoiseModel()
noise_model_kolkata = noise_model1.from_dict(noise_model_kolkata)
noise_model_cairo = noise_model2.from_dict(noise_model_cairo)
noise_model_montreal = noise_model3.from_dict(noise_model_montreal)


sampler = Sampler(
    backend_options = {
        'method': 'statevector',
        'device': 'CPU',
        'noise_model': noise_model_kolkata
    },
    run_options = {
        'shots': shot,
        'seed': seeds,
    },
    transpile_options = {
        'seed_transpiler':seed_transpiler
    }
)


h, first_term = parseHamiltonian('Hamiltonian/OHhamiltonian.txt')
# measurements, measurement_dict = group_measurements(h)
# filehandler = open(b"142observables.obj","wb")
# pickle.dump((measurements, measurement_dict),filehandler)

filehandler = open(b"142observables.obj","rb")
measurements, measurement_dict = pickle.load(filehandler)


dummy_ansatz = QuantumCircuit(12,12)
dummy_ansatz.x(0)
dummy_ansatz.x(1)
dummy_ansatz.x(2)
dummy_ansatz.x(3)
dummy_ansatz.x(6)
dummy_ansatz.x(7)
dummy_ansatz.x(8)
dummy_ansatz.x(9)

# pdb.set_trace()
print(varsaw_expectation(dummy_ansatz, measurements, measurement_dict, first_term, h, sampler))
# try:
#     job = estimator.run(batch_size * [dummy_ansatz], batch_size * [main_operator], [])
#     estimator_result = job.result()
# except Exception as exc:
#     raise AlgorithmError("The primitive job to evaluate the energy failed!") from exc

# values = estimator_result.values

# print(values)