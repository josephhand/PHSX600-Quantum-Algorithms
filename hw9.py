import qml
import numpy as np

###################
# Codercise G.1.1 #
###################

n_bits = 4
dev = qml.device("default.qubit", wires=n_bits)


def oracle_matrix(combo):
    """Return the oracle matrix for a secret combination.

    Args:
        combo (list[int]): A list of bits representing a secret combination.

    Returns:
        array[float]: The matrix representation of the oracle.
    """
    index = np.ravel_multi_index(combo, [2] * len(combo))  # Index of solution
    my_array = np.identity(2 ** len(combo))  # Create the identity matrix
    my_array[index, index] = -1
    return my_array


@qml.qnode(dev)
def oracle_amp(combo):
    """Prepare the uniform superposition and apply the oracle.

    Args:
        combo (list[int]): A list of bits representing the secret combination.

    Returns:
        array[complex]: The quantum state (amplitudes) after applying the oracle.
    """
    ##################
    # YOUR CODE HERE #
    ##################

    # Prepare uniform superposition
    for i in range(n_bits):
        qml.Hadamard(wires=i)

    # Apply oracle
    qml.QubitUnitary(oracle_matrix(combo), wires=range(n_bits))
    
    return qml.state()


####################
# Codercise G.1.2a #
####################

n_bits = 4


def diffusion_matrix():
    """Return the diffusion matrix.

    Returns:
        array[float]: The matrix representation of the diffusion operator.
    """
    ##################
    # YOUR CODE HERE #
    ##################
    states = 2**n_bits
    return np.full((states, states), 2**(1-n_bits)) - np.identity(states)


@qml.qnode(dev)
def difforacle_amp(combo):
    """Apply the oracle and diffusion matrix to the uniform superposition.

    Args:
        combo (list[int]): A list of bits representing the secret combination.

    Returns:
        array[complex]: The quantum state (amplitudes) after applying the oracle
        and diffusion.
    """
    ##################
    # YOUR CODE HERE #
    ##################

    # Prepare uniform superposition
    for i in range(n_bits):
        qml.Hadamard(wires=i)

    # Apply Grover operator
    qml.QubitUnitary(oracle_matrix(combo), wires=range(n_bits))
    qml.QubitUnitary(diffusion_matrix(), wires=range(n_bits))
    
    return qml.state()


####################
# Codercise G.1.2b #
####################

@qml.qnode(dev)
def two_difforacle_amp(combo):
    """Apply the Grover operator twice to the uniform superposition.

    Args:
        combo (list[int]): A list of bits representing the secret combination.

    Returns:
        array[complex]: The resulting quantum state.
    """
    ##################
    # YOUR CODE HERE #
    ##################
    
    # Prepare uniform superposition
    for i in range(n_bits):
        qml.Hadamard(wires=i)

    # Apply Grover operator
    qml.QubitUnitary(oracle_matrix(combo), wires=range(n_bits))
    qml.QubitUnitary(diffusion_matrix(), wires=range(n_bits))
    
    # Apply Grover operator again
    qml.QubitUnitary(oracle_matrix(combo), wires=range(n_bits))
    qml.QubitUnitary(diffusion_matrix(), wires=range(n_bits))

    
    return qml.state()

###################
# Codercise G.2.1 #
###################

n_bits = 5
dev = qml.device("default.qubit", wires=n_bits)


def oracle_matrix(combo):
    """Return the oracle matrix for a secret combination.

    Args:
        combo (list[int]): A list of bits representing a secret combination.

    Returns:
        array[float]: The matrix representation of the oracle.
    """
    index = np.ravel_multi_index(combo, [2] * len(combo))  # Index of solution
    my_array = np.identity(2 ** len(combo))  # Create the identity matrix
    my_array[index, index] = -1
    return my_array


def diffusion_matrix():
    """Return the diffusion matrix.

    Returns:
        array[float]: The matrix representation of the diffusion operator.
    """
    psi_piece = (1 / 2**n_bits) * np.ones(2**n_bits)
    ident_piece = np.eye(2**n_bits)
    return 2 * psi_piece - ident_piece


@qml.qnode(dev)
def grover_circuit(combo, num_steps):
    """Apply the Grover operator num_steps times to the uniform superposition
       and return the state.

    Args:
        combo (list[int]): A list of bits representing the secret combination.
        num_steps (int): The number of iterations of the Grover operator
            our circuit is to perform.

    Returns:
        array[complex]: The quantum state (amplitudes) after repeated Grover
        iterations.
    """
    ##################
    # YOUR CODE HERE #
    ##################

    # Prepare uniform superposition
    for i in range(n_bits):
        qml.Hadamard(wires=i)

    # Apply Grover operator num_steps times
    for _ in range(num_steps):
        qml.QubitUnitary(oracle_matrix(combo), wires=range(n_bits))
        qml.QubitUnitary(diffusion_matrix(), wires=range(n_bits))
    
    return qml.state()


my_steps = 4  # YOUR STEP NUMBER HERE

###################
# Codercise G.3.1 #
###################

n_bits = 5
query_register = list(range(n_bits))
aux = [n_bits]
all_wires = query_register + aux
dev = qml.device("default.qubit", wires=all_wires)


def oracle(combo):
    """Implement an oracle using a multi-controlled X gate.

    Args:
        combo (list): A list of bits representing the secret combination.
    """
    combo_str = "".join(str(j) for j in combo)
    ##################
    # YOUR CODE HERE #
    ##################

    qml.MultiControlledX(query_register, aux, combo_str)
    


###################
# Codercise G.3.2 #
###################

def hadamard_transform(my_wires):
    """Apply the Hadamard transform on a given set of wires.

    Args:
        my_wires (list[int]): A list of wires on which the Hadamard transform will act.
    """
    for wire in my_wires:
        qml.Hadamard(wires=wire)


def diffusion():
    """Implement the diffusion operator using the Hadamard transform and
    multi-controlled X."""

    ##################
    # YOUR CODE HERE #
    ##################

    hadamard_transform(query_register)
    qml.MultiControlledX(query_register, aux, "0"*len(query_register))
    hadamard_transform(query_register)


###################
# Codercise G.3.3 #
###################

@qml.qnode(dev)
def grover_circuit(combo):
    """Apply the MultiControlledX Grover operator and return probabilities on
    query register.

    Args:
        combo (list[int]): A list of bits representing the secret combination.

    Returns:
        array[float]: Measurement outcome probabilities.
    """
    ##################
    # YOUR CODE HERE #
    ##################
    # PREPARE QUERY AND AUXILIARY SYSTEM
    qml.PauliX(wires=aux)
    hadamard_transform(all_wires)
    
    # APPLY GROVER ITERATION
    oracle(combo)
    diffusion()
    
    return qml.probs(wires=query_register)

