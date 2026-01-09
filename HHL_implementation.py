#IMPORTS-------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit.library import PhaseEstimation
from qiskit_aer import AerSimulator, Aer
from qiskit_aer.noise import (NoiseModel,ReadoutError,depolarizing_error,)

import warnings
# This ignores the specific Qiskit deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#HHL ALGORITHM --------------------------------------------------------------------------------
class HHL:
    """
    This class solves the linear system Ax = b.
    
    MATHEMATICAL GOAL:
    We want to prepare a state |x> such that A|x> = |b>.
    This is equivalent to applying the inverse of A to state |b>: |x> = A^{-1}|b>.
    Since A is Hermitian, it has spectral decomposition A = sum(lambda_j |u_j><u_j|).
    The inverse is just A^{-1} = sum(1/lambda_j |u_j><u_j|).
    
    ALGORITHM:
    1. Express |b> in the eigenbasis of A: |b> = sum(c_j |u_j>)
    2. Use QPE to entangle eigenvalues: sum(c_j |u_j> |lambda_j>)
    3. Rotate ancilla by theta ~ 1/lambda: sum(c_j |u_j> |lambda_j> |1>) * (1/lambda_j)
    4. Uncompute QPE: sum( (c_j / lambda_j) |u_j> ) |0> |1>
    """
    
    def __init__(self, matrix_A, vector_b, clock_qubits=4):
        """
        This function gives the initial state to an object when it is created.
        Moreover, it prepares the data in ways needed to run the algorithm without 
        problems (normalization, avoiding overflows...)

        Args:
            matrix_A (np.array): The matrix to invert. Must be Hermitian.
            vector_b (np.array): The target vector.
            clock_qubits (int): The 'Resolution' of our eigenvalue measurement.
                                4 qubits = 16 'bins' to categorize eigenvalues.
        """
        self.matrix_A = matrix_A
        self.vector_b_raw = vector_b
        
        # --------------- DATA LOADING ----------------------------------------------------
        # Normalize input vector |b>
        self.vector_b_norm = np.linalg.norm(vector_b)
        self.vector_b = vector_b / self.vector_b_norm
        self.k = clock_qubits
        
        # --------- EIGENVALUE SCALING --------------------------------------------------------------
        # The quantum evolution operator U = e^{iAt} has phases in [0, 2pi].
        # Real eigenvalues (like 100 or 0.01) usually don't fit this range nicely.
        # So we calculate a time scalar 't' to stretch or shrink our eigenvalues 
        # so they fill about 85% of the quantum register's capacity.
        
        eigs = np.linalg.eigvalsh(self.matrix_A)
        l_min = np.min(np.abs(eigs))
        l_max = np.max(np.abs(eigs))
        
        # The largest eigenvalue maps to the integer ~85% of 2^k (0.85*2^k)
        target_int = int(2**self.k * 0.85) 
        #since the phase will be lambda*t, this ensures it will never be bigger than 0.85*2*pi, avoiding overlaps
        self.t = (2 * np.pi * target_int) / (l_max * 2**self.k)
        
        # Constant C: Ensures the rotation 2*arcsin(C/lambda) is never > 1.
        min_eig_projected = (l_min * self.t / (2*np.pi)) * (2**self.k)
        #this ensures that C is never larger than lambda
        self.C = min_eig_projected * 0.95 

    def get_fidelity(self, probs_1, probs_2):
        """
        Calculates the fidelity between two probabilty distributions in order to compare how close they are
        F = (Sum(sqrt(p1 * p2)))^2
        """

        sqrt_prod = np.sqrt(probs_1 * probs_2)
        
        fidelity = np.sum(sqrt_prod) ** 2
        
        return fidelity

    def run_and_plot(self, title="HHL Result", p_error=0 , sweep_steps=0, max_error=0.01):
        """
        This function runs the algorithm and plots the results
        """

        # --------- CLASSICAL VERIFICATION --------------------------------------------
        # We compare the quantum result against standard NumPy linear algebra.
        x_class = np.linalg.solve(self.matrix_A, self.vector_b_raw)
        x_norm = x_class / np.linalg.norm(x_class)
        c_probs = np.abs(x_norm)**2

        # --- POST-SELECTION ----------------------------------------------------------
        # HHL is not deterministic. It only succeeds when the Ancilla measures '1'.
        #we must filter the results to keep only the successful shots and transform them into probabilities
        # This function is definded here for using it later
        def probs_counts(counts_raw):
            success_counts = {}
            total_success = 0
        
            #'counts' is a dictionary containing:
            # 1. bitstring of the state (key) (examples: '00', '01', '10', '11')
            #   important: actually, the bitstring contains the state + ancilla (example, 00 state and ancilla 1: '001')

            # 2.  amount of times the state has been measured (counts) (examples: 121, 1, 23...)
            # .items() allows us to access both at the same time
            for bs, count in counts_raw.items():
                # if the ancilla is one
                if bs[-1] == '1': 
                    #add the amount of successful counts
                    total_success += count
                    # this converts from binary to decimal int
                    idx = int(bs[:-1], 2)
                    #update the amount of sucess counts for each state safely
                    success_counts[idx] = success_counts.get(idx, 0) + count
            
            if total_success == 0:
                print(f"  [Failed] 0 successful shots. This happens if C is too small or Matrix is ill-conditioned.")
                return
            
            #start a vector of 0s
            probs = np.zeros(len(self.vector_b))
            for idx in success_counts:
                #update with probabilities
                probs[idx] = success_counts[idx] / total_success

            return probs

        print(f"\n############## Running: {title} ##############")
        print("Matrix A:\n", np.round(self.matrix_A, 2))
        
        # Calculate number of qubits needed to store vector b (log2 of vector size)
        nb = int(np.log2(len(self.vector_b)))
        
        # ----- REGISTER ALLOCATION ---------------------------------------------
        #Ancilla: if ancilla=1, the inversion of the matrix was successful 
        #Clock: stores the eigenvalues temporarily
        #b: data register for |b> (input) and |x> (output)
        #the register q_b starts storing the vector |b>, and it is reshaped as the 
        #algorithm progresses, until it "becomes" the output vector |x>

        q_ancilla = QuantumRegister(1, 'ancilla') 
        q_clock = QuantumRegister(self.k, 'clock')
        q_b = QuantumRegister(nb, 'b')
        #this ones stores the measurements permanently
        c_meas = ClassicalRegister(nb + 1, 'meas')         
        #the full circuit is composed of the classical + quantum bits
        qc = QuantumCircuit(q_ancilla, q_clock, q_b, c_meas)
        # Load the problem vector |b> into the quantum computer.
        qc.initialize(self.vector_b, q_b)

        # ------------------- QUANTUM PHASE ESTIMATION (QPE) ------------------------------
        # This block performs the transformation: |u_j>|0> -> |u_j>|lambda_j>
        # It uses Hamiltonian Simulation U = e^{iAt}.
        # If A is not Hermitian, this unitary gate cannot be constructed physically.

        #create e^{iAt}
        u_mat = expm(1j * self.matrix_A * self.t)
        #make it a unitary gate
        u_gate = UnitaryGate(u_mat, label='e^iAt')
        #perform phase estimation to find eigenvalues
        qpe = PhaseEstimation(self.k, u_gate)
        qc.append(qpe, q_clock[:] + q_b[:])

        # -------------------------- EIGENVALUE INVERSION-------------------------------------
        # apply a function f(x) = 1/x to the coefficients.
        # We do this using a Controlled-Rotation on the ancilla.
        # The rotation angle is determined by the integer 'i' stored in the Clock register.
        
        #This loops over every possible "bin" (integer value) the clock register can hold (1,2,3,4...2^k)
        for i in range(1, 2**self.k):
            # We assume the integer 'i' represents a scaled eigenvalue.
            # it should be bigger than C by construction, but this may change if noise is present
            #that is why it is safer to include this check
            if i >= self.C:
                #compute C/lambda
                ratio = self.C / i
                #another safety check
                if ratio <= 1.0:
                    # Calculate rotation angle theta
                    #theta=2*arcsin(C/lambda)
                    theta = 2 * np.arcsin(ratio)
                    # Construct the control pattern for integer 'i'
                    #the following line turns an integer 'i' into its k-bit binary version
                    #example: if  i=3 and k=4 -> its 4-bit binary is 0011
                    pattern = format(i, f'0{self.k}b')
                    #Since we want the controlled mcry gate to fire for all numbers (regardless if they contain bits in 0 state)
                    #we are going to find where the bits = 0 are located and apply X gates on them to make them 1

                    # moreover, reversed(pattern) is needed since qiskit reads the pattern the other way around
                    #this line creates a list with 'True' in the position where a bit is 0 and 'False' if it's 1
                    flip_indices = [idx for idx, bit in enumerate(reversed(pattern)) if bit == '0']
                    #then we apply X gates to turn 0 into 1 
                    if flip_indices: qc.x([q_clock[j] for j in flip_indices])
                    # Apply the rotation (Controlled-RY)
                    # This shifts amplitude from |0> to |1> proportional to 1/lambda.
                    qc.mcry(theta, q_clock[:], q_ancilla[0])
                    #the following line 'restores' the initial number (it sets the bits we flipped back to 0)
                    if flip_indices: qc.x([q_clock[j] for j in flip_indices])

        # --------- INVERSE QPE --------------------------------------------------------------------
        #We must "undo" the Phase Estimation to disentangle the Clock register from our Data register
        qc.append(qpe.inverse(), q_clock[:] + q_b[:])

        # ------------ MEASUREMENT ---------------------------------------------------------------
        # We measure both the success flag (Ancilla) and the solution (b)
        #and assign the results to bit 0 and bit row 1, respectively
        qc.measure(q_ancilla, c_meas[0])
        qc.measure(q_b, c_meas[1:]) 

                # --------- NNOISE MODEL -----------------------------------------
        # In order to add the errors we apply them to the correspondig transpilation in quantum gates of the Phase estimation, the multicontrolled Y gates and so on
        # The most common and possible errors are applied in order to keep fidelity with the real case
        def noise_model(p):
            nm = NoiseModel()
            # CNOT error (critical for HHL)
            nm.add_all_qubit_quantum_error(depolarizing_error(p, 2), ['cx'])
            # 1-Qubit gate error
            nm.add_all_qubit_quantum_error(depolarizing_error(p, 1), ['sx', 'rz', 'x', 'h'])
            # Measurement error
            readout_err = ReadoutError([[0.98, 0.02],[0.02, 0.98]])
            nm.add_all_qubit_readout_error(readout_err, ['measure'])
            return nm

        # --- ------QUANTUM CIRCUIT IDEAL SIMULATION -----------------------------------------
        print("  -> Simulating Quantum Circuit (without Noise)...")
        #will use Aer in case we want to include noise
        simulator = AerSimulator()
        # Transpile the circuit
        qc_transpiled = transpile(qc, simulator)
        # run simulator many times (shots)
        result = simulator.run(qc_transpiled, shots=20000).result()
        #historgram (amount of times we got a specific result)
        counts = result.get_counts()
        q_probs = probs_counts(counts)
        

                # --- ------QUANTUM CIRCUIT NOISY SIMULATION -----------------------------------------
        if sweep_steps > 1:
            print(f"\n  -> Simulating Quantum Circuit (with Noise)...")

            print(f" -> Starting sweep with {sweep_steps} steps up to error {max_error}...")
            
            error_range = np.linspace(0, max_error, sweep_steps)
            fidelities = []
            
            for p in error_range:
                # Run with variable noise 'p'
                res = simulator.run(qc_transpiled, shots=10000, noise_model=noise_model(p)).result()
                probs_noisy = probs_counts(res.get_counts())
                
                # Calculate fidelity vs Analytical
                # (We use the analytical solution as 'ground truth' to see when noise breaks the algorithm)
                fid = self.get_fidelity(c_probs, probs_noisy)
                fidelities.append(fid)
            
            # Plot Curve
            plt.figure(figsize=(10, 6))
            plt.plot(error_range, fidelities, 'o-', color='royalblue', linewidth=2, label='HHL Fidelity between Analytical and Noisy simulation')
            plt.title(f"{title} - Robustness Analysis")
            plt.xlabel("Error Probability (p)")
            plt.ylabel("Fidelity")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()


        print(f"\n  -> Simulating Quantum Circuit (with Noise pr0bability {p_error})...")

        # run simulator many times (shots)
        result_noise = simulator.run(qc_transpiled, shots=20000, noise_model=noise_model(p_error)).result()
        #histogram (amount of times we got a specific result)
        counts_noise = result_noise.get_counts()
        q_probs_noise = probs_counts(counts_noise)

        # --------- Fidelity --------------------------------------------
    
        # Ideal vs Analytical
        # Measures how well the HHL (noiseless) approximates the mathematical solution
        fid_algo = self.get_fidelity(c_probs, q_probs)
        
        # Ideal vs Noisy 
        # Measures how much damage the noise caused to the quantum solution
        fid_noise = self.get_fidelity(q_probs, q_probs_noise)

        # Noisy vs Analytical 
        # Measures how much damage the noise caused to the analytical solution, which is what will happen in a real device
        fid_noise = self.get_fidelity(c_probs, q_probs_noise)

        print(f"\n--- FIDELITY RESULTS ---")
        print(f" Algorithm (Ideal vs Analytical): {fid_algo:.4f}")
        print(f" Robustness (Ideal vs Noisy): {fid_noise:.4f}")
        print(f" Possible real experiment (Noisy vs Analytical): {fid_noise:.4f}")


        #plot all results 
        self._plot(c_probs, q_probs, q_probs_noise, title)

    def _plot(self, c_probs, q_probs, q_probs_noise, title):
        x = np.arange(len(c_probs))
        width = 0.35
        
        #create figure
        fig, ax = plt.subplots(figsize=(8, 4))
        
        ax.bar(x - width/2, c_probs, width, label='Analytical (True)', color='navy')
        ax.bar(x, q_probs, width, label='HHL (Quantum)', color='cornflowerblue')
        ax.bar(x + width/2, q_probs_noise, width, label='HHL with Noise (Quantum)', color='green')
        
        ax.set_title(title)
        ax.set_ylabel('Probability |x|^2')
        ax.set_xlabel('Basis State')
        ax.legend()
        ax.grid(linestyle='--', alpha=0.7)
        
        #SAVE FIGURES-----------------------------------------------------------------------------------
        #create a 'results' folder
        import os
        if not os.path.exists('results'):
            os.makedirs('results')

        #format name properly
        clean_filename = title.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")
        filename = f"results/{clean_filename}.pdf"
        
        #save figure as PDF
        fig.savefig(filename, format='pdf', bbox_inches='tight')
        
        print(f"  [Saved] Plot saved as: {filename}")
        plt.show()

# ==========================================
#              END OF ALGORITHM
