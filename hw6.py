import qml
import numpy as np

###################
# Codercise A.1.1 #
###################

n_bits = 4
dev = qml.device("default.qubit", wires=n_bits)

@qml.qnode(dev)
def naive_circuit():
    """Create a uniform superposition and return the probabilities.

    Returns: 
        array[float]: Probabilities for observing different outcomes.
    """
    for wire in range(n_bits):

        ##################
        # YOUR CODE HERE #
        ##################

        qml.Hadamard(wires=wire)
        

    return qml.probs(wires=range(n_bits))

###################
# Codercise A.2.1 #
###################

def oracle_matrix(combo):
    """Return the oracle matrix for a secret combination.
    
    Args:
        combo (list[int]): A list of bits representing a secret combination.
         
    Returns: 
        array[float]: The matrix representation of the oracle.
    """
    index = np.ravel_multi_index(combo, [2]*len(combo)) # Index of solution
    my_array = np.identity(2**len(combo)) # Create the identity matrix

    ##################
    # YOUR CODE HERE #
    ##################

    # MODIFY DIAGONAL ENTRY CORRESPONDING TO SOLUTION INDEX
    my_array[index,index] = -1

    return my_array


###################
# Codercise A.2.2 #
###################

n_bits = 4
dev = qml.device("default.qubit", wires=n_bits)

@qml.qnode(dev)
def oracle_circuit(combo):
    """Create a uniform superposition, apply the oracle, and return probabilities.
    
    Args:
        combo (list[int]): A list of bits representing a secret combination.

    Returns:
        list[float]: The output probabilities.
    """

    ##################
    # YOUR CODE HERE #
    ##################

    for w in range(n_bits):
        qml.Hadamard(wires=w)

    qml.QubitUnitary(oracle_matrix(combo), wires=range(n_bits))

    return qml.probs(wires=range(n_bits))


###################
# Codercise A.3.1 #
###################

n_bits = 4
dev = qml.device("default.qubit", wires=n_bits)

@qml.qnode(dev)
def pair_circuit(x_tilde, combo):
    """Test a pair labelled by x_tilde for the presence of a solution.
    
    Args:
        x_tilde (list[int]): An (n_bits - 1)-string labelling the pair to test.
        combo (list[int]): A secret combination of n_bits 0s and 1s.
        
    Returns:
        array[float]: Probabilities on the last qubit.
    """
    for i in range(n_bits-1): # Initialize x_tilde part of state
        if x_tilde[i] == 1:
            qml.PauliX(wires=i)

    ##################
    # YOUR CODE HERE #
    ##################
    qml.Hadamard(wires=n_bits-1)
    qml.QubitUnitary(oracle_matrix(combo), wires=range(n_bits))
    qml.Hadamard(wires=n_bits-1)
    
    return qml.probs(wires=n_bits-1)


###################
# Codercise A.3.2 #
###################

def pair_lock_picker(trials):
    """Create a combo, run pair_circuit until it succeeds, and tally success rate.
    
    Args:
        trials (int): Number of times to test the lock picker.

    Returns:
        float: The average number of times the lock picker uses pair_circuit.
    """
    x_tilde_strs = [np.binary_repr(n, n_bits-1) for n in range(2**(n_bits-1))]
    x_tildes = [[int(s) for s in x_tilde_str] for x_tilde_str in x_tilde_strs] 

    test_numbers = []

    for trial in range(trials):
        combo = secret_combo(n_bits) # Random list of bits
        counter = 0
        for x_tilde in x_tildes:
            counter += 1

            ##################
            # YOUR CODE HERE #
            ##################
            probs = pair_circuit(x_tilde, combo)
            if np.isclose(probs[1], 1):
                break
        
        test_numbers.append(counter)
    return sum(test_numbers)/trials

trials = 500
output = pair_lock_picker(trials)

print(f"For {n_bits} bits, it takes", output, "pair tests on average.")


###################
# Codercise A.4.1 #
###################

n_bits = 4
dev = qml.device("default.qubit", wires=n_bits)

def multisol_oracle_matrix(combos):
    """Return the oracle matrix for a set of solutions.

    Args:
        combos (list[list[int]]): A list of secret bit strings.

    Returns:
        array[float]: The matrix representation of the oracle.
    """
    indices = [np.ravel_multi_index(combo, [2]*len(combo)) for combo in combos]
    ##################
    # YOUR CODE HERE #
    ##################
    
    oracle_matrix = np.identity(2**len(combos[0]))

    for i in indices:
        oracle_matrix[i,i] = -1

    return oracle_matrix
    

@qml.qnode(dev)
def multisol_pair_circuit(x_tilde, combos):
    """Implements the circuit for testing a pair of combinations labelled by x_tilde.
    
    Args:
        x_tilde (list[int]): An (n_bits - 1)-bit string labelling the pair to test.
        combos (list[list[int]]): A list of secret bit strings.

    Returns:
        array[float]: Probabilities on the last qubit.
    """
    for i in range(n_bits-1): # Initialize x_tilde part of state
        if x_tilde[i] == 1:
            qml.PauliX(wires=i)

    ##################
    # YOUR CODE HERE #
    ##################
    qml.Hadamard(wires=n_bits-1)
    qml.QubitUnitary(multisol_oracle_matrix(combos), wires=range(n_bits))
    qml.Hadamard(wires=n_bits-1)

    return qml.probs(wires=n_bits-1)


###################
# Codercise A.4.2 #
###################

def parity_checker(combos):
    """Use multisol_pair_circuit to determine the parity of a solution set.

    Args:
        combos (list[list[int]]): A list of secret combinations.

    Returns: 
        int: The parity of the solution set.
    """
    parity = 0
    x_tilde_strs = [np.binary_repr(n, n_bits-1) for n in range(2**(n_bits-1))]
    x_tildes = [[int(s) for s in x_tilde_str] for x_tilde_str in x_tilde_strs]
    for x_tilde in x_tildes:

        ##################
        # YOUR CODE HERE #
        ##################

        # IMPLEMENT PARITY COUNTING ALGORITHM
        probs = multisol_pair_circuit(x_tilde, combos)

        parity = (parity + np.argmax(probs)) % 2
        

    return parity


###################
# Codercise A.5.1 #
###################

n_bits = 4
dev = qml.device("default.qubit", wires=n_bits)

@qml.qnode(dev)
def hoh_circuit(combo):
    """A circuit which applies Hadamard-oracle-Hadamard and returns probabilities.
    
    Args:
        combo (list[int]): A list of bits representing a secret combination.

    Returns:
        list[float]: Measurement outcome probabilities.
    """

    ##################
    # YOUR CODE HERE #
    ##################

    for w in range(n_bits):
        qml.Hadamard(wires=w)
        
    qml.QubitUnitary(oracle_matrix(combo), wires=range(n_bits))
    
    for w in range(n_bits):
        qml.Hadamard(wires=w)

    return qml.probs(wires=range(n_bits))

###################
# Codercise A.6.1 #
###################

n_bits = 4
dev = qml.device("default.qubit", wires=n_bits)

@qml.qnode(dev)
def multisol_hoh_circuit(combos):
    """A circuit which applies Hadamard, multi-solution oracle, then Hadamard.
    
    Args:
        combos (list[list[int]]): A list of secret bit strings.

    Returns: 
        array[float]: Probabilities for observing different outcomes.
    """

    ##################
    # YOUR CODE HERE #
    ##################
    for w in range(n_bits):
        qml.Hadamard(wires=w)
        
    qml.QubitUnitary(multisol_oracle_matrix(combos), wires=range(n_bits))
    
    for w in range(n_bits):
        qml.Hadamard(wires=w)
    

    return qml.probs(wires=range(n_bits))


###################
# Codercise A.6.2 #
###################

def deutsch_jozsa(promise_var):
    """Implement the Deutsch–Jozsa algorithm and guess the promise variable.
    
    Args:
        promise_var (int): Indicates whether the function is balanced (0) or constant (1).
        
    Returns: 
        int: A guess at the promise variable.
    """
    if promise_var == 0:
        how_many = 2**(n_bits - 1)
    else:
        how_many = np.random.choice([0, 2**n_bits]) # Choose all or nothing randomly
    combos = multisol_combo(n_bits, how_many) # Generate random combinations

    ##################
    # YOUR CODE HERE #
    ##################-
    
    result = multisol_hoh_circuit(combos)[0]

    if np.isclose(result, 0):
        return 0
    else:
        return 1


