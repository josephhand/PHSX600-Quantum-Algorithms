import qml
import numpy as np

###################
# Codercise F.1.1 #
###################
def coefficients_to_values(coefficients):
    """Returns the value representation of a polynomial
    
    Args:
        coefficients (array[complex]): a 1-D array of complex 
            coefficients of a polynomial with 
            index i representing the i-th degree coefficient

    Returns: 
        array[complex]: the value representation of the 
            polynomial 
    """
    ##################
    # YOUR CODE HERE #
    ################## 
    return np.fft.fft(coefficients)

A = [4, 3, 2, 1]
print(coefficients_to_values(A))


###################
# Codercise F.1.2 #
###################
def values_to_coefficients(values):
    """Returns the coefficient representation of a polynomial
    
    Args:
        values (array[complex]): a 1-D complex array with 
            the value representation of a polynomial 

    Returns: 
        array[complex]: a 1-D complex array of coefficients
    """
    
    ##################
    # YOUR CODE HERE #
    ################## 
    return np.fft.ifft(values)


A = [10.+0.j,  2.-2.j,  2.+0.j,  2.+2.j]
print(values_to_coefficients(A))

###################
# Codercise F.1.3 #
###################
def nearest_power_of_2(x):
    """Given an integer, return the nearest power of 2. 
    
    Args:
        x (int): a positive integer

    Returns: 
        int: the nearest power of 2 of x
    """
    ##################
    # YOUR CODE HERE #
    ################## 
    
    return 1 if x == 0 else int(2**np.ceil(np.log2(x)))

###################
# Codercise F.1.4 #
###################

def fft_multiplication(poly_a, poly_b):
    """Returns the result of multiplying two polynomials
    
    Args:
        poly_a (array[complex]): 1-D array of coefficients 
        poly_b (array[complex]): 1-D array of coefficients 

    Returns: 
        array[complex]: complex coefficients of the product
            of the polynomials
    """
    ##################
    # YOUR CODE HERE #
    ################## 

    # Calculate the number of values required
    num_values = len(poly_a) + len(poly_b) - 1
    
    # Figure out the nearest power of 2
    num_values = nearest_power_of_2(num_values)

    # Pad zeros to the polynomial
    poly_a_padded = np.pad(poly_a, (0, num_values-len(poly_a)), 'constant')
    poly_b_padded = np.pad(poly_b, (0, num_values-len(poly_b)), 'constant')

    # Convert the polynomials to value representation 
    values_a = coefficients_to_values(poly_a_padded)
    values_b = coefficients_to_values(poly_b_padded)

    # Multiply
    values_product = np.multiply(values_a, values_b)

    # Convert back to coefficient representation
    poly_product = values_to_coefficients(values_product)
    
    return poly_product


###################
# Codercise F.2.1 #
###################

num_wires = 1
dev = qml.device("default.qubit", wires=num_wires)

@qml.qnode(dev)
def one_qubit_QFT(basis_id):
    """A circuit that computes the QFT on a single qubit. 
    
    Args:
        basis_id (int): An integer value identifying 
            the basis state to construct.
    
    Returns:
        array[complex]: The state of the qubit after applying QFT.
    """
    # Prepare the basis state |basis_id>
    bits = [int(x) for x in np.binary_repr(basis_id, width=num_wires)]
    qml.BasisState(bits, wires=[0])

    ##################
    # YOUR CODE HERE #
    ##################

    qml.Hadamard(wires=0)
    
    return qml.state()


###################
# Codercise F.2.2 #
###################

num_wires = 2
dev = qml.device("default.qubit", wires=num_wires)

@qml.qnode(dev)
def two_qubit_QFT(basis_id):
    """A circuit that computes the QFT on two qubits using qml.QubitUnitary. 
    
    Args:
        basis_id (int): An integer value identifying the basis state to construct.
    
    Returns:
        array[complex]: The state of the qubits after the QFT operation.
    """
    
    # Prepare the basis state |basis_id>
    bits = [int(x) for x in np.binary_repr(basis_id, width=num_wires)]
    qml.BasisState(bits, wires=[0, 1])
    
    ##################
    # YOUR CODE HERE #
    ##################
    mat = 0.5*np.array(
        [
            [1, 1, 1, 1],
            [1, 1j, -1, -1j],
            [1, -1, 1, -1],
            [1, -1j, -1, 1j]
        ]
    )

    qml.QubitUnitary(mat, wires=[0, 1])
    
    return qml.state()

###################
# Codercise F.2.3 #
###################

num_wires = 2
dev = qml.device("default.qubit", wires=num_wires)

@qml.qnode(dev)
def decompose_two_qubit_QFT(basis_id):
    """A circuit that computes the QFT on two qubits using elementary gates.
    
    Args:
        basis_id (int): An integer value identifying the basis state to construct.
    
    Returns:
        array[complex]: The state of the qubits after the QFT operation.
    """
    # Prepare the basis state |basis_id>
    bits = [int(x) for x in np.binary_repr(basis_id, width=num_wires)]
    qml.BasisState(bits, wires=[0, 1])
    
    ##################
    # YOUR CODE HERE #
    ##################
    qml.Hadamard(wires=0)
    qml.ControlledPhaseShift(np.pi/2, wires=[1, 0])
    qml.Hadamard(wires=1)
    qml.SWAP(wires=[0, 1])
    
    return qml.state()

###################
# Codercise F.3.1 #
###################

num_wires = 3
dev = qml.device("default.qubit", wires=num_wires)

@qml.qnode(dev)
def three_qubit_QFT(basis_id):
    """A circuit that computes the QFT on three qubits.
    
    Args:
        basis_id (int): An integer value identifying the basis state to construct.
        
    Returns:
        array[complex]: The state of the qubits after the QFT operation.
    """
    # Prepare the basis state |basis_id>
    bits = [int(x) for x in np.binary_repr(basis_id, width=num_wires)]
    qml.BasisState(bits, wires=[0, 1, 2])
    
    ##################
    # YOUR CODE HERE #
    ##################

    qml.Hadamard(wires=0)
    qml.ControlledPhaseShift(2*np.pi/2**2, wires=[1, 0])
    qml.ControlledPhaseShift(2*np.pi/2**3, wires=[2, 0])

    qml.Hadamard(wires=1)
    qml.ControlledPhaseShift(2*np.pi/2**2, wires=[2, 1])

    qml.Hadamard(wires=2)
    
    qml.SWAP(wires=[0, 2])
    
    return qml.state()


###################
# Codercise F.3.2 #
###################

dev = qml.device('default.qubit', wires=4)

            
def swap_bits(n_qubits):
    """A circuit that reverses the order of qubits, i.e.,
    performs a SWAP such that [q1, q2, ..., qn] -> [qn, ... q2, q1].
    
    Args:
        n_qubits (int): An integer value identifying the number of qubits.
    """
    ##################
    # YOUR CODE HERE #
    ##################

    for i in range(0, int(n_qubits/2)):
        qml.SWAP(wires=[i, n_qubits-1-i])

@qml.qnode(dev) 
def qft_node(basis_id, n_qubits):
    # Prepare the basis state |basis_id>
    bits = [int(x) for x in np.binary_repr(basis_id, width=n_qubits)]
    qml.BasisState(bits, wires=range(n_qubits))
    # qft_rotations(n_qubits)
    swap_bits(n_qubits)
    return qml.state()

###################
# Codercise F.3.3 #
###################

dev = qml.device('default.qubit', wires=4)

def qft_rotations(n_qubits):
    """A circuit performs the QFT rotations on the specified qubits.
    
    Args:
        n_qubits (int): An integer value identifying the number of qubits.
    """

    ##################
    # YOUR CODE HERE #
    ################## 

    for i in range(0, n_qubits):
        qml.Hadamard(wires=i)
        for j in range(0, n_qubits-i-1):
            qml.ControlledPhaseShift(2*np.pi/2**(j+2), wires=[i+j+1, i])


@qml.qnode(dev) 
def qft_node(basis_id, n_qubits):
    # Prepare the basis state |basis_id>
    bits = [int(x) for x in np.binary_repr(basis_id, width=n_qubits)]
    qml.BasisState(bits, wires=range(n_qubits))
    qft_rotations(n_qubits)
    swap_bits(n_qubits)
    return qml.state()


###################
# Codercise F.3.4 #
###################

dev = qml.device('default.qubit', wires=4)

def qft_recursive_rotations(n_qubits, wire=0):
    """A circuit that performs the QFT rotations on the specified qubits
        recursively.
        
    Args:
        n_qubits (int): An integer value identifying the number of qubits.
        wire (int): An integer identifying the wire 
                    (or the qubit) to apply rotations on.
    """

    ##################
    # YOUR CODE HERE #
    ##################

    if wire == n_qubits:
        return

    qml.Hadamard(wire)

    for i in range(wire+1, n_qubits):
        qml.ControlledPhaseShift(2*np.pi/2**(i-wire+1), wires=[i, wire])
    
    qft_recursive_rotations(n_qubits, wire+1)

@qml.qnode(dev) 
def qft_node(basis_id, n_qubits):
    # Prepare the basis state |basis_id>
    bits = [int(x) for x in np.binary_repr(basis_id, width=n_qubits)]
    qml.BasisState(bits, wires=range(n_qubits))
    qft_recursive_rotations(n_qubits)
    swap_bits(n_qubits)
    return qml.state()

###################
# Codercise F.3.5 #
###################

dev = qml.device('default.qubit', wires=4)

@qml.qnode(dev)
def pennylane_qft(basis_id, n_qubits):
    """A that circuit performs the QFT using PennyLane's QFT template.
    
    Args:
        basis_id (int): An integer value identifying 
            the basis state to construct.
        n_qubits (int): An integer identifying the 
            number of qubits.
            
    Returns:
        array[complex]: The state after applying the QFT to the qubits.
    """
    # Prepare the basis state |basis_id>
    bits = [int(x) for x in np.binary_repr(basis_id, width=n_qubits)]
    qml.BasisState(bits, wires=range(n_qubits))

    ##################
    # YOUR CODE HERE #
    ##################

    qml.QFT(wires=range(n_qubits))
    
    return qml.state()

