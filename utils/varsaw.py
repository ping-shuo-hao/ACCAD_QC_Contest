# -*- coding: utf-8 -*-
"""
Given a particular qubit Hamiltonian, measuring the expected energy of any
given quantum state will depend only on the individual terms of that
Hamiltonian.

measureCircuit.py generates a circuit which will measure a quantum state in the
correct bases to allow the energy to be calculated. This may require generating
multiple circuits if the same qubit needs to be measured in two perpendicular
bases (i.e. Z and X).

To find the minimum number of circuits needed to measure an entire Hamiltonian,
we treat the terms of H as nodes in a graph, G, where there are edges between
nodes indicate those two terms commute with one another. Finding the circuits
now becomes a clique finding problem which can be solved by the
BronKerbosch algorithm.
"""
import pdb
import time
import sys
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer.primitives import Sampler
import numpy as np
import pprint
import copy
import networkx as nx
from networkx.algorithms import approximation
from qiskit.quantum_info import SparsePauliOp

class CommutativityType(object):
    def gen_comm_graph(term_array):
        raise NotImplementedError


class QWCCommutativity(CommutativityType):
    def gen_comm_graph(term_array):
        g = {}
        for i, term1 in enumerate(term_array):
            comm_array = []
            for j, term2 in enumerate(term_array):

                if i == j: continue
                commute = True
                for c1, c2 in zip(term1, term2):
                    if c1 == '*': continue
                    if (c1 != c2) and (c2 != '*'):
                        commute = False
                        break
                if commute:
                    comm_array += [''.join(term2)]
            g[''.join(term1)] = comm_array

        print('MEASURECIRCUIT: Generated graph for the Hamiltonian with {} nodes.'.format(len(g)))

        return g


class FullCommutativity(CommutativityType):
    def gen_comm_graph(term_array):
        g = {}
        for i, term1 in enumerate(term_array):
            comm_array = []
            for j, term2 in enumerate(term_array):

                if i == j: continue
                non_comm_indices = 0
                for c1, c2 in zip(term1, term2):
                    if c1 == '*': continue
                    if (c1 != c2) and (c2 != '*'):
                        non_comm_indices += 1
                if (non_comm_indices % 2) == 0:
                    comm_array += [''.join(term2)]
            g[''.join(term1)] = comm_array

        print('MEASURECIRCUIT: Generated graph for the Hamiltonian with {} nodes.'.format(len(g)))

        return g


def prune_graph(G, nodes):
    for n in nodes:
        neighbors = G.pop(n)
        for nn in neighbors:
            G[nn].remove(n)


def degeneracy_ordering(graph):
    """
    Produce a degeneracy ordering of the vertices in graph, as outlined in,
    Eppstein et. al. (arXiv:1006.5440)
    """

    # degen_order, will hold the vertex ordering
    degen_order = []

    while len(graph) > 0:
        # Populate D, an array containing a list of vertices of degree i at D[i]
        D = []
        for node in graph.keys():
            Dindex = len(graph[node])
            cur_len = len(D)
            if cur_len <= Dindex:
                while cur_len <= Dindex:
                    D.append([])
                    cur_len += 1
            D[Dindex].append(node)

        # Add the vertex with lowest degeneracy to degen_order
        for i in range(len(D)):
            if len(D[i]) != 0:
                v = D[i].pop(0)
                degen_order += [v]
                prune_graph(graph, [v])

    return degen_order


def degree_ordering(G):
    nodes = list(G.keys())
    return sorted(nodes, reverse=True, key=lambda n: len(G[n]))


def BronKerbosch_pivot(G, R, P, X, cliques):
    """
    For a given graph, G, find a maximal clique containing all of the vertices
    in R, some of the vertices in P, and none of the vertices in X.
    """
    if len(P) == 0 and len(X) == 0:
        # Termination case. If P and X are empty, R is a maximal clique
        cliques.append(R)
    else:
        # choose a pivot vertex
        pivot = next(iter(P.union(X)))
        # Recurse
        for v in P.difference(G[pivot]):
            # Recursion case.
            BronKerbosch_pivot(G, R.union({v}), P.intersection(G[v]),
                               X.intersection(G[v]), cliques)
            P.remove(v)
            X.add(v)


def NetworkX_approximate_clique_cover(graph_dict):
    """
    NetworkX poly-time heuristic is based on
    Boppana, R., & Halldórsson, M. M. (1992).
    Approximating maximum independent sets by excluding subgraphs.
    BIT Numerical Mathematics, 32(2), 180–196. Springer.
    """
    G = nx.Graph()
    for src in graph_dict:
        for dst in graph_dict[src]:
            G.add_edge(src, dst)
    return approximation.clique_removal(G)[1]


def BronKerbosch(G):
    """
    Implementation of Bron-Kerbosch algorithm (Bron, Coen; Kerbosch, Joep (1973),
    "Algorithm 457: finding all cliques of an undirected graph", Commun. ACM,
    ACM, 16 (9): 575–577, doi:10.1145/362342.362367.) using a degree ordering
    of the vertices in G instead of a degeneracy ordering.
    See: https://en.wikipedia.org/wiki/Bron-Kerbosch_algorithm
    """

    max_cliques = []

    while len(G) > 0:
        P = set(G.keys())
        R = set()
        X = set()
        v = degree_ordering(G)[0]
        cliques = []
        BronKerbosch_pivot(G, R.union({v}), P.intersection(G[v]),
                           X.intersection(G[v]), cliques)

        # print('i = {}, current v = {}'.format(i,v))
        # print('# cliques: ',len(cliques))

        sorted_cliques = sorted(cliques, key=len, reverse=True)
        max_cliques += [sorted_cliques[0]]
        # print(sorted_cliques[0])

        prune_graph(G, sorted_cliques[0])

    return max_cliques


def generate_circuit_matrix(Nq, max_cliques):
    circuitMatrix = np.empty((Nq, len(max_cliques)), dtype=str)
    for i, clique in enumerate(max_cliques):
        # each clique will get its own circuit
        clique_list = list(clique)

        # Take the first string to be the circuit template, i.e. '****Z**Z'
        circStr = list(clique_list[0])
        # print(circStr)

        # Loop through the characters of the template and replace the '*'s
        # with a X,Y,Z found in another string in the same clique
        for j, char in enumerate(circStr):
            if char == '*':
                # print('j = {}, c = {}'.format(j,char))
                for tstr in clique_list[1:]:
                    # Search through the remaining strings in the clique
                    # print('tstr = {}, {} != * = {}'.format(tstr, tstr[j], (tstr[j] != '*')))
                    if tstr[j] != '*':
                        circStr[j] = tstr[j]
                        break

                if circStr[j] == '*':
                    # After searching through all of the strings in the clique
                    # the current char is still '*', this means none of the
                    # terms in this clique depend on this qubit -> measure in
                    # the Z basis.
                    circStr[j] = 'Z'

        # Once the circuit string is filled in, add it to the circuit matrix
        for q in range(Nq):
            circuitMatrix[q, i] = circStr[q]

    return circuitMatrix


def genMeasureCircuit(H, Nq, commutativity_type, clique_cover_method=BronKerbosch):
    """
    Take in a given Hamiltonian, H, and produce the minimum number of
    necessary circuits to measure each term of H.

    Args:
    H: hamiltonian
    Nq: Number of qubits (?)
    commutativity_type: General commutativity or Qubit-wise commutativity

    Returns:
        List[QuantumCircuits]
    """

    start_time = time.time()

    term_reqs = np.full((len(H), Nq), '*', dtype=str)
    #    print(term_reqs)
    for i, term in enumerate(H):
        qubit_index = 0
        for op in term[1]:
            #            qubit_index = int(op[1:])
            term_reqs[i][qubit_index] = op
            #            basis = op[0]
            #            term_reqs[i][qubit_index] = basis
            qubit_index += 1

    # print(term_reqs)
    # Generate a graph representing the commutativity of the Hamiltonian terms
    comm_graph = commutativity_type.gen_comm_graph(term_reqs)

    # Find a set of cliques within the graph where the nodes in each clique
    # are disjoint from one another.
    max_cliques = clique_cover_method(comm_graph)

    end_time = time.time()

    print('MEASURECIRCUIT: {} found {} unique circuits'.format(
        clique_cover_method.__name__, len(max_cliques)))
    et = end_time - start_time
    print('MEASURECIRCUIT: Elapsed time: {:.6f}s'.format(et))
    return max_cliques


def parseHamiltonian(myPath):
    first=0
    H = []
    with open(myPath) as hFile:
        for i, line in enumerate(hFile):
            line = line.split("*")
            coef = line[0]
            if coef[0] == "-":
                coef = coef[1:].replace(" ", "")
                coef = -1 * float(coef)
            else:
                coef = coef[1:].replace(" ", "")
                coef = float(coef)

            ops = line[1]
            ops = ops.replace(" ", "")
            ops = ops.replace("\n", "")
            pauli_str = ""
            for op in ops:
                if op == "I":
                    pauli_str += "*"
                else:
                    pauli_str += op
            if i==0:
                first=coef
            else:
                H += [(coef, pauli_str)]

    return H,first


def merge_cliques(cliques, Nq):
    measurements = []

    for cliq in cliques:
        term = ""
        print("c", cliq)

        for i in range(Nq):
            temp = [op[i] for op in cliq]
            finall_op = "*"
            for op in temp:
                if op != "*":
                    finall_op = op
                    break
            term += finall_op

        measurements.append(term)

    return measurements


def group_measurements(hamiltonian):
    ops = [term[1] for term in hamiltonian]
    Nq = max([len(op) for op in ops])
    print('%s qubits' % Nq)
    cliques = genMeasureCircuit(hamiltonian, Nq, QWCCommutativity)
    measurements = merge_cliques(cliques, Nq)
    measure_dict = {}
    i = 0
    for cliq in cliques:
        for pauli_str in cliq:
            measure_dict[pauli_str] = i
        i += 1
    print("Total measurement terms: ", len(measurements))
    return measurements, measure_dict

def vqe_circuit(input_circuit, hamiltonian,Nq):
    '''
    Args:
    input_circuit: A vqe circuit with ansatz.
    hamiltonian: The hamiltonian string whose expectation would be measured
    using this circuit

    Returns:
    The VQE circuit for the given Pauli tensor hamiltonian
    '''
    final_circuit=input_circuit.copy()
    #add the measurement operations
    for i, el in enumerate(hamiltonian):
        if el == 'I':
            #no measurement for identity
            continue
        elif el == 'Z':
            final_circuit.measure(i, i)
        elif el == 'X':
            final_circuit.h(i)
            final_circuit.measure(i, i)
        elif el == 'Y':
            final_circuit.sdg(i)
            final_circuit.h(i)
            final_circuit.measure(i, i)

    return final_circuit

def get_term_value(pauli_term,measurement_term):
    value=1
    for i in range(len(pauli_term)):
        if pauli_term[i]!="*" and measurement_term[i]=="1":
            value*=-1

    return value

def get_expecatation_value(first_term,hamiltonian,results,measure_dict):
    expect_val=first_term
    for coef,term in hamiltonian:
        result=results[measure_dict[term]]
        temp_val=0
        for re in result.keys():
            temp_val+=result[re]*get_term_value(term,re)
        # print(term,temp_val)
        expect_val+=coef*temp_val

    return expect_val

def varsaw_expectation(circuit,measurements,measure_dict,first_term,hamiltonian, sampler, params=None):
    list_of_circuit=[]
    for term in measurements:
        cir=circuit.copy()
        list_of_circuit.append(vqe_circuit(cir,term,len(term)))

    # for cir in list_of_circuit:
    #     print(cir)
    
    if params == None:
        job = sampler.run(list_of_circuit)
    else:
        job = sampler.run(list_of_circuit, parameter_values=[params] * len(list_of_circuit))
    results=[]
    for result in job.result().quasi_dists:
        results.append(result.binary_probabilities())
    # print(results)

    return get_expecatation_value(first_term,hamiltonian,results,measure_dict)
    





if __name__ == "__main__":
    # change the number of qubits based on which hamiltonian is selected
    #hfile = 'OHhamiltonian.txt'
    #    H=[(0.1,"ZZ*Z"),(0.2,"Z*ZX"),(0.3,'ZZ**'),(0.1,"**ZX"),(0.1,"ZXXZ"),(0.1,"XZ*Z"),(0.1,"ZX*Z"),(0.1,"*XZZ"),(0.1,"X*ZZ"),(0.1,"XX*X")]
    #H,first = parseHamiltonian(hfile)
    #print(len(H))

    #measurements, measure_dict = varsaw_measurements(H)
    #h=[(2,"XXX"),(3,"*ZY"),(4,"*Z*")]
    #m_d={"XXX":0,"*ZY":1,"*Z*":1}
    #m_r=[{"000":0.2,"101":0.5,"111":0.3},{"010":0.3,"011":0.5,"110":0.2}]
    #print(get_expecatation_value(10,h,m_r,m_d))
    qc=QuantumCircuit(2,2)
    qc.h([0,1])
    h=[(1,"XY"),(2,"Z*"),(3,"ZZ")]
    
    measurements,measurement_dict=group_measurements(h)
    print(measurements)
    print(measurement_dict)
    print(varsaw_expectation(qc,measurements,measurement_dict,10,h))
    #varsaw_calculation(qc,["XY","IZ"],None,None,None,None)




    # Infer number of qubits from widest term in Hamiltonian
#    ops = [term[1] for term in H]
#    Nq = max([len(op) for op in ops])
#    print('%s qubits' % Nq)

#    cliques = genMeasureCircuit(H, Nq, QWCCommutativity)
#    measurements=merge_cliques(cliques,Nq)
#    measure_dict={}
#    i=0
#    for cliq in cliques:
#        for pauli_str in cliq:
#            measure_dict[pauli_str]=measurements[i]
#        i+=1
#    print("Total measurement terms: ",len(measurements))
#    print(measurements)
#    print(measure_dict)

