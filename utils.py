import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
import os
import json
import pandas as pd

def generate_problem(size=6, quadratic_normal=(0, 1), linear_normal=(0, 1), constant_normal=(0, 1), sparsity=0, positive=False, seed=None):
    """
    Generate a random QUBO h(s)=s^T A s + a^T s + alpha.

    Parameters
    ----------
    size : int
        Number of variables n.
    quadratic_normal : tuple[float, float]
        (mean, std) for entries of A.
    linear_normal : tuple[float, float]
        (mean, std) for entries of a.
    constant_normal : tuple[float, float]
        (mean, std) for alpha.
    sparsity : float
        Fraction in [0,1] of A-entries dropped (kept with prob 1-sparsity).
    positive : bool
        If True, |A_ij| is used (nonnegative A).
    seed : int | None
        Seed for numpy Generator.

    Returns
    -------
    A : np.ndarray shape (n,n)
    a : np.ndarray shape (n,)
    alpha : float
    """
    if seed is not None:
        random = np.random.default_rng(seed)
    else:
        random = np.random.default_rng()

    A = np.zeros((size, size))
    a = np.zeros(size)
    alpha = random.normal(constant_normal[0], constant_normal[1])

    for i in range(size):
        a[i] = random.normal(linear_normal[0], linear_normal[1])
        for j in range(i, size):
            if np.random.rand() > sparsity:
                weight = random.normal(quadratic_normal[0], quadratic_normal[1])
                if positive:
                    weight = abs(weight)
                A[i, j] = weight
                A[j, i] = weight
    
    return A, a, alpha

def index_to_s(index, bits):
    """
    Convert integer index to ±1 spin vector of length `bits` (0→+1, 1→-1).

    Parameters
    ----------
    index : int
        Integer in [0, 2^bits-1].
    bits : int
        Number of spins.

    Returns
    -------
    tuple[int, ...]
        Spin configuration in {+1,-1}^bits.
    """
    index_binary = format(index, f'0{bits}b')
    s = []
    for bit in index_binary:
        if bit =='0':
            s.append(1)
        else:
            s.append(-1)
            
    return tuple(s)

def create_z_array(size, reverse=False):
    """
    Build Z-eigenvalue patterns (±1) for each qubit over 2^size states.

    Parameters
    ----------
    size : int
        Number of qubits.
    reverse : bool
        If True, reverse qubit order.

    Returns
    -------
    list[np.ndarray]
        List of length `size`, each array shape (2^size,).
    """
    z = np.array([1, -1])
    z_array = [np.array(1)]*size

    for i in range(size):
        z_array[i] = np.ones(2**i)
        z_array[i] = np.kron(z_array[i], z)
        z_array[i] = np.kron(z_array[i], np.ones(2**(size-i-1)))

    if reverse:
        z_array = z_array[::-1]
        
    return z_array

class QUBO:
    def __init__(self, A, a, alpha):
        """
        Hold a QUBO instance with matrix A, vector a, and constant alpha.

        Parameters
        ----------
        A : np.ndarray shape (n,n)
            Quadratic coefficients.
        a : np.ndarray shape (n,)
            Linear coefficients.
        alpha : float
            Constant term.
        """
        self.A = A
        self.a = a
        self.alpha = alpha
        self.size = len(a)
        
    def calculate_cost(self, s):
        """
        Evaluate h(s)=s^T A s + a^T s + alpha for s∈{±1}^n.

        Parameters
        ----------
        s : array-like shape (n,)
            Spin vector with entries ±1.

        Returns
        -------
        float
            h(s).
        """
        s = np.array(s)
        return s.T @ self.A @ s + self.a.T @ s + self.alpha
    
    def qubo_to_ising_brute(self):
        """
        Enumerate all 2^n spin states and store the energy vector H.

        Effects
        -------
        Sets attributes: H (np.ndarray), best_index (int),
        best_s (tuple), min_cost (float).
        """
        possible_s = list(product([1, -1], repeat=self.size))
        H = [self.calculate_cost(s) for s in possible_s]

        self.H = np.array(H)
        self.best_index = np.argmin(H)
        self.best_s = index_to_s(self.best_index, self.size)
        self.min_cost = np.min(H)

    def qubo_to_ising(self):
        """
        Compute H without enumeration using Kronecker Z-patterns.

        Effects
        ------------
        Sets attributes: H (np.ndarray), best_index (int),
        best_s (tuple), min_cost (float).
        """
        size = self.size
        dim = 2**size
        
        z_array = create_z_array(size)

        H = np.ones(dim)*self.alpha
        for i in range(size):
            H += self.a[i]*z_array[i]
            for j in range(size):
                H += self.A[i,j]*(z_array[i]*z_array[j])

        self.H = H
        self.best_index = np.argmin(H)
        self.best_s = index_to_s(self.best_index, size)
        self.min_cost = np.min(H)

def get_shift(qubo_list, version=4):
    """
    Compute an additive shift ≥ max_b max_m h_m(b) to make energies nonnegative.

    Parameters
    ----------
    qubo_list : list[QUBO]
        QUBOs to bound/shift.
    version : {1,2,3,4}
        Shift heuristic:
          1) ||A||_1 + ||a||_1 - alpha
          2) -n·λ_min(A) + ||a||_1 - alpha
          3) -(n+1)·λ_min(Â)        with Â=[A ½a; ½a^T alpha]
          4) -(n+1)·λ_min(Ã) - alpha    with Ã=[A ½a; ½a^T 0]

    Returns
    -------
    float
        Shift value.
    """
    shift = -np.inf
    for qubo in qubo_list:
        if version == 1:
            temp = np.sum(np.abs(qubo.A)) + np.sum(np.abs(qubo.a)) - qubo.alpha

        elif version == 2:
            lam_min = np.linalg.eigvalsh(qubo.A).min()
            temp = - len(qubo.a)*lam_min + np.abs(qubo.a).sum() - qubo.alpha

        elif version == 3:
            A_tilde = np.vstack((qubo.A, 0.5*qubo.a))
            col = np.hstack((0.5*qubo.a, qubo.alpha)).reshape(-1, 1)
            A_tilde = np.hstack((A_tilde, col))
            temp = - (len(qubo.a)+1)*np.linalg.eigvalsh(A_tilde).min()
        elif version == 4:
            A_tilde = np.vstack((qubo.A, 0.5*qubo.a))
            col = np.hstack((0.5*qubo.a, 0)).reshape(-1, 1)
            A_tilde = np.hstack((A_tilde, col))
            temp = - (len(qubo.a)+1)*np.linalg.eigvalsh(A_tilde).min()-qubo.alpha
        else:
            raise ValueError("Invalid version for shift calculation. Use 1, 2, 3, or 4.")

        if temp > shift:
            shift = temp
    return shift

class M_QUBOs:
    def __init__(self, qubos_list=[]):
        """
        Collect multiple QUBOs, build shifted energy arrays, and precompute h_max via the infinity norm across QUBOs.

        Parameters
        ----------
        qubos_list : list[QUBO]
            Input problems; each will have H computed if missing.

        Attributes
        ----------
        qubos_list : list[QUBO]
        shift : float
        ising_list : np.ndarray shape (M, 2^n)
        size : int
        M : int
        ising_size : int
        h_max : tuple
            Output of _calculate_one_h_p(∞).
        h_p_list : list
            Results for requested p-values.
        """
        
        self.qubos_list = qubos_list

        self.shift = get_shift(self.qubos_list)

        self.ising_list = []
        for qubo in qubos_list:
            if not hasattr(qubo, 'H'):
                qubo.qubo_to_ising()
            self.ising_list.append(qubo.H + self.shift)
        self.ising_list = np.array(self.ising_list)

        self.size = qubos_list[0].size
        self.M = len(qubos_list)
        self.ising_size = len(self.ising_list[0])

        self.h_max = self._calculate_one_h_p(p = np.inf)

        self.h_p_list = []

    def _calculate_one_h_p(self, p):
        """
        Compute entrywise p norms over M energies: h_p[b]=(∑_m |h_m[b]|^p)^{1/p}.

        Parameters
        ----------
        p : float | np.inf
            Norm order.

        Returns
        -------
        tuple
            (h_p, h_p_min, argmin, s_opt, (min, second_min, max))
        """
        
        h_p = np.linalg.norm(self.ising_list, axis=0, ord=p)
        h_p_sort = np.sort(h_p)
        arg_min = np.argmin(h_p)

        return (h_p, h_p_sort[0], arg_min, index_to_s(arg_min, self.size), (h_p_sort[0], h_p_sort[1], h_p_sort[-1]))

    def calculate_h_p(self, p_max, p_min=1):
        """
        Compute and store h_p for p∈{p_min,…,p_max}.

        Parameters
        ----------
        p_max : int
        p_min : int, default 1

        Side Effects
        ------------
        Populates self.h_p_list.
        """
        p_list = []
        for i in range(p_min, p_max+1):
                p_list.append(i)
        self.h_p_list = [self._calculate_one_h_p(p) for p in p_list]


###### implementation of Algorithm 1 to recieve C_p ######
def factorials_up_to_p(p):
    """
    Return factorials (0! … p!) as a tuple.

    Parameters
    ----------
    p : int

    Returns
    -------
    tuple[int, ...]
    """
    factorials = [1]
    for i in range(1, p + 1):
        factorials.append(factorials[-1] * i)
    return tuple(factorials)

def create_Cp(q_list, p):
    """
    Compute C_p coefficients (order p) for the shifted multi-QUBO expansion.

    Parameters
    ----------
    q_list : list[QUBO]
        Source QUBOs.
    p : int
        Expansion order.

    Returns
    -------
    np.ndarray
        C_p array of length 2^n indexed by spin bitmasks.
    """
    M = len(q_list)
    factorials = factorials_up_to_p(2*p)
    shift = get_shift(q_list, version=4)

    A_list = []
    a_list = []
    const_list = []


    for q in q_list:
        A_list.append(2*q.A)
        a_list.append(q.a)
        const_list.append(np.trace(q.A) + q.alpha + shift)


    n = len(a_list[0])
    C_p = np.zeros(2**n)

    upper_indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_summands = len(upper_indices) + n + 1

    def compositions(n, p):
        if n == 1:
            yield (p,)
        else:
            for i in range(p + 1):
                for tail in compositions(n - 1, p - i):
                    yield (i,) + tail
    
    for v in compositions(num_summands, p):
    
        s = 0
        for bit in range(n):
            t_2 = v[bit+1] + sum(v[index+n+1] for index, (i,j) in enumerate(upper_indices) if i == bit or j == bit)
            s += (t_2 % 2) * (2 ** bit)
        
        c = 0
        coeff = factorials[p] / np.prod([factorials[i] for i in v])

        for m in range(M):
            t_1 = coeff*const_list[m]**v[0]

            for i in range(n):
                t_1 *= a_list[m][i]**v[i+1]

            for index, (i,j) in enumerate(upper_indices):
                t_1 *= A_list[m][i, j]**v[index+n+1]
            c += t_1

        C_p[s] += c

    return C_p

def convert_Cp_to_Hp(C_p):
    """
    Reconstruct H_p over all bitstrings from C_p via Z-monomials.

    Parameters
    ----------
    C_p : np.ndarray
        Length 2^n coefficient array.

    Returns
    -------
    np.ndarray
        H_p values over all 2^n states.
    """
    size = int(np.log2(len(C_p)))
    H = np.ones(2**size)*C_p[0]
    z_array = create_z_array(size, reverse=True)
    for i, c in enumerate(C_p):
        if i == 0:
            continue
        z = np.ones(2**size)
        for index, qubit in enumerate(format(i, f"0{size}b")):
            z *= z_array[index]**int(qubit)
        H += c*z
    return H

###### Functions for applications ######

def create_contrained_qubo(qubo, contrains: list):
    """
    Build an M_QUBOs with base QUBO plus linear-penalty constraints (t, τ, gamma).

    Parameters
    ----------
    qubo : QUBO
        Base problem.
    contrains : list[tuple[np.ndarray, float, float]]
        Each (t, tau, gamma) yields Q' with (A, a-gamma t, alpha+gamma τ).

    Returns
    -------
    M_QUBOs
    """
    q_list = [qubo]
    A, a, alpha = qubo.A, qubo.a, qubo.alpha
    for (t, tau, gamma) in contrains:
        q_list.append(QUBO(A, a-gamma*t, alpha+gamma*tau))
    return M_QUBOs(q_list)

def generate_constraint(size=6, gamma=1e2, tau_scale=0, seed=None):
    """
    Sample a random constraint triple (t, τ, gamma).

    Parameters
    ----------
    size : int
        Length of t.
    gamma : float
        Penalty weight gamma.
    tau_scale : float
        multiplicative factor for τ's stddev.
    seed : int | None
        RNG seed.

    Returns
    -------
    tuple[np.ndarray, float, float]
        (t, tau, gamma).
    """
    if seed is not None:
        random = np.random.default_rng(seed)
    else:
        random = np.random.default_rng()
    t = random.normal(0, 1, size=size)
    tau =  random.normal(0, 1) * tau_scale
    constraint = (t, tau, gamma)
    return constraint

def create_t_pm(W, v):
    """
    Map (W, v) to QUBO parameters (A, a, alpha) for T_±-style objectives.

    Parameters
    ----------
    W : np.ndarray shape (n,n)
        Edge weights (adjacency-like).
    v : np.ndarray shape (n,)
        Vertex weights.

    Returns
    -------
    A : np.ndarray
    a : np.ndarray
    alpha : float
    """
    A = W/4
    a = (W@np.ones(len(v))+v)/2
    alpha = (np.sum(W) + 2*np.sum(v))/4

    return A, a, alpha

def generate_random_graph(n, mode='gaussian', seed=None):
    """
    Sample n points in 2D and per-vertex weights.

    Parameters
    ----------
    n : int
        Number of vertices.
    mode : {'gaussian','uniform'}
        Point distribution (uniform inside unit disk or 2D Gaussian).
    seed : int | None
        RNG seed.

    Returns
    -------
    points : dict[int, tuple[float,float]]
    vertices : dict[int, float]
    """
    if seed is not None:
        np.random.seed(seed)
    
    if mode not in ['gaussian', 'uniform']:
        raise ValueError("Mode must be 'gaussian' or 'uniform'")

    points = {}
    vertices = {}
    while len(points) < n:
        if mode == 'uniform':
            x, y = np.random.uniform(-1, 1, 2)
            if x**2 + y**2 <= 1:
                i = len(points)
                points[i] = (x, y)
                vertices[i] = np.random.rand()

        else:  # gaussian
            x, y = np.random.normal(0, 1, size = 2)
            i = len(points)
            points[i] = (x, y)
            vertices[i] = np.random.normal(0, 1)

    return points, vertices

def generate_distance_matrix(points, vertices=None):
    """
    Build an upper-triangular Euclidean distance matrix from 2D points.

    Parameters
    ----------
    points : dict[int, tuple[float,float]]
        Node → (x,y).
    vertices : dict[int, float] | None
        Optional per-node weights.

    Returns
    -------
    W : np.ndarray shape (n,n)
        Upper-triangular distances (zeros elsewhere).
    v : np.ndarray, optional
        If vertices is given, stacked weights.
    """
    n = len(points)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
            W[i, j] = dist
    if vertices is not None:
        v = np.array([vertices[i] for i in range(n)])
        return W, v
    else:
        return W

###### Visualization functions ######

def visualize_ising(qubo: QUBO, save_as: str = None, fontsize: int = 16):
    """
    Plot the energy landscape H over all bitstring indices.

    Parameters
    ----------
    qubo : QUBO
        Must have H computed.
    save_as : str | None
        Basename to save .svg/.pdf under figures/.
    fontsize : int
        Label and tick font size.
    """
    if not hasattr(qubo, 'H'):
        raise ValueError("H is not calculated. Run qubo_to_ising() first.")
    
    H = qubo.H
    plt.figure(figsize=(10, 6))
    plt.rcParams['text.usetex'] = True #This can cause problems, so only enable if you have LaTeX installed
    plt.plot(range(len(H)), H, '-')
    plt.plot(qubo.best_index, qubo.min_cost, 'o', label = rf'$b={qubo.best_s} \; h(b)={round(qubo.min_cost,3)}$')
    plt.xlabel(r'index of $\mathbf{b}$', fontsize=fontsize)
    plt.ylabel(r'$h(\mathbf{b})$', fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    if save_as is not None:
        plt.savefig("figures/"+save_as+".svg", bbox_inches='tight')
        plt.savefig("figures/"+save_as+".pdf", bbox_inches='tight')
    plt.show()

def visualize_m_ising(qubos: M_QUBOs, save_as: str = None, show_part: bool = False, tpm: bool = False, show_original: bool = True, fontsize: int = 16, legend_anchor: tuple = (1,1)):
    """
    Plot h_max, optional source H_m, and stored h_(p) curves for an M_QUBOs.

    Parameters
    ----------
    qubos : M_QUBOs
    save_as : str | None
    show_part : bool
        If True, highlight region where H_1-H_2>0.
    tpm : bool
        If True, label first two QUBOs as T_+, T_-.
    show_original : bool
        If True, overlay individual shifted H_m.
    fontsize : int
    legend_anchor : tuple[float,float]
    """
    plt.figure(figsize=(10, 6))
    plt.rcParams['text.usetex'] = True
    n_p = len(qubos.h_p_list)
    n_q = qubos.M
    blues = cm.Blues(np.linspace(0.4, 1, n_p))
    greens = cm.Greens(np.linspace(0.4, 1, n_q))
    blue_index = 0
    green_index = 0
    x = np.linspace(0, qubos.ising_size-1, qubos.ising_size, dtype=int)
    shift = qubos.shift

    if show_part:
        def f(x):
            return qubos.qubos_list[0].H[x]- qubos.qubos_list[1].H[x]
        mask = f(x) > 0
        marker = '.-'
    else:
        mask = np.ones_like(x, dtype=bool)
        marker = '-'

    plt.plot(x, np.where(mask, qubos.h_max[0], np.nan), marker, zorder=2, color='red')
    plt.plot(qubos.h_max[2], qubos.h_max[1], 'o', color='red', zorder=5, label=r'$h_{\max}$')

    
    custom_titles = []
    if show_original:
        if not tpm:
            title1 = r'$h_{m}$'
            plt.plot([], [], color='none', label=title1)
            custom_titles.append(title1)
        for index, q in enumerate(qubos.qubos_list[:1 if show_part else None]):
            if tpm:
                label = r'$T_+$' if index == 0 else r'$T_-$'
            else:
                label = rf'$m={index+1}$'
            plt.plot(x, q.H+shift, ':', zorder=0, color=greens[green_index])
            plt.plot(q.best_index, q.min_cost+shift, 'o', color=greens[green_index], zorder=3, label=label)
            green_index += 1

    if len(qubos.h_p_list) > 0:
        title2 = r'$h_{(p)}$'
        plt.plot([], [], color='none', label=title2)
        custom_titles.append(title2)

    for index, h_p in enumerate(qubos.h_p_list):
        plt.plot(x, h_p[0], '--', zorder=1, color=blues[blue_index])
        plt.plot(h_p[2], h_p[1], 'o', color=blues[blue_index], zorder=4, label = rf'$p={index+1}$')
        blue_index += 1


    if show_part:
        n = np.log2(len(x))
        y_low, y_high = qubos.h_max[1]-shift - n**2/4, qubos.h_max[1]-shift + n**2
        ax = plt.gca()
        ax.set_ylim(y_low, y_high)
    else:
        ax = plt.gca()

    plt.xlabel(r'$\mathbf{b}$ in base 10', fontsize=fontsize)
    plt.ylabel(r'$h(\mathbf{b})$', fontsize=fontsize)

    leg = plt.legend(bbox_to_anchor=legend_anchor, loc='upper left', fontsize=fontsize)
    for txt in leg.get_texts():
        if txt.get_text() in custom_titles:
            txt.set_x(txt.get_position()[0] - 10)


    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    if save_as is not None:
        plt.savefig("figures/"+save_as+".svg", bbox_inches='tight')
        plt.savefig("figures/"+save_as+".pdf", bbox_inches='tight')
    plt.show()

def plot_performance_from_csv(csv_path: str, output_path: str = None, ylim: tuple = None, p_max: int = None, fontsize: int = 16, legend_1_anchor: tuple = (0.5, 0.95), legend_2_anchor: tuple = (0.5, 1.025)):
    """
    Plot mean absolute/relative errors vs p for each problem size from a CSV.

    Parameters
    ----------
    csv_path : str
        File with columns: size, wrong_idx(JSON), energy_diff(JSON).
    output_path : str | None
        Basename to save .pdf/.svg.
    ylim : tuple | None
        y-limits (log scale).
    p_max : int | None
        Max p to display (infer if None).
    fontsize : int
    legend_1_anchor : tuple
    legend_2_anchor : tuple
    """

    df = pd.read_csv(
        csv_path,
        converters={
            "wrong_idx": json.loads,
            "energy_diff": json.loads,
        }
    )

    size_list = sorted(df["size"].unique())
    n_sizes   = len(size_list)
    if p_max is None:
        p_max = len(df["wrong_idx"].iat[0])

    fig, ax = plt.subplots(figsize=(10, 6))
    blues   = cm.Blues(np.linspace(0.4, 0.8, n_sizes))
    oranges = cm.Oranges(np.linspace(0.4, 0.8, n_sizes))

    blue_handles   = []
    orange_handles = []
    markers = ["d", "s", "^", "o", "v", "x", "*"][:n_sizes]

    for size, cb, co, marker in zip(size_list, blues, oranges, markers):
        sub = df[df["size"] == size]
        wrong_arr  = np.vstack(sub["wrong_idx"].values)
        energy_arr = np.vstack(sub["energy_diff"].values)

        mean_wrong  = wrong_arr.mean(axis=0)[:p_max]
        mean_energy = energy_arr.mean(axis=0)[:p_max]

        # absolute Errors
        bl, = ax.plot(
            np.arange(1, p_max + 1),
            mean_wrong,
            marker=marker,
            color=cb,
            label=f"Size {size}"
        )
        blue_handles.append(bl)

        # relative Errors
        ol, = ax.plot(
            np.arange(1, p_max + 1),
            mean_energy,
            marker=marker,
            color=co,
            label=f"Size {size}"
        )
        orange_handles.append(ol)


    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel(r"$p$", fontsize=fontsize)
    ax.set_ylabel("Error", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xticks(np.arange(1, p_max+1, 1))

    abs_title = r"Absolute Error $\epsilon$:"
    handles2  = [plt.Line2D([], [], linestyle="none")] + blue_handles
    labels2   = [abs_title] + [f"Size {s}" for s in size_list]
    leg = fig.legend(
        handles2, labels2,
        loc="upper center",
        bbox_to_anchor=legend_1_anchor,
        ncol=n_sizes + 1,
        fontsize=fontsize,
        handlelength=0,
        handletextpad=0.5
    )

    rel_title = r"Relative Error $\delta$:"
    handles2  = [plt.Line2D([], [], linestyle="none")] + orange_handles
    labels2   = [rel_title] + [f"Size {s}" for s in size_list]
    fig.legend(
        handles2, labels2,
        loc="upper center",
        bbox_to_anchor=legend_2_anchor,
        ncol=n_sizes + 1,
        fontsize=fontsize,
        handlelength=0,
        handletextpad=0.5
    )
    fig.add_artist(leg)

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path+".pdf", bbox_inches="tight")
        plt.savefig(output_path+".svg", bbox_inches='tight')
    plt.show()

def plot_performance_from_csv_multi(csv_pattern: str, M_list=(2, 8), output_path: str = None, ylim: tuple = (1e-6, 2), p_max: int = None, fontsize: int = 16):
    """
    Stack the performance plot across multiple M using per-M CSVs.

    Parameters
    ----------
    csv_pattern : str
        e.g. "data/performance_random_M{M}.csv".
    M_list : iterable[int]
        Values of M to include (rows).
    output_path : str | None
        Basename to save .pdf/.svg.
    ylim : tuple
        y-limits (log scale).
    p_max : int | None
        Max p to display (infer if None).
    fontsize : int
    """

    first_df = pd.read_csv(
        csv_pattern.format(M=M_list[0]),
        converters={"wrong_idx": json.loads, "energy_diff": json.loads}
    )
    size_list_ref = sorted(first_df["size"].unique())
    n_sizes = len(size_list_ref)
    if p_max is None:
        p_max   = len(first_df["wrong_idx"].iat[0])

    blues   = cm.Blues(np.linspace(0.4, 0.8, n_sizes))
    oranges = cm.Oranges(np.linspace(0.4, 0.8, n_sizes))
    markers = ["d", "s", "^", "o", "v", "x", "*"][:n_sizes]

    n_rows = len(M_list)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 5 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    blue_handles, orange_handles = [], []

    for row_idx, (ax, M) in enumerate(zip(axes, M_list)):
        df = pd.read_csv(
            csv_pattern.format(M=M),
            converters={"wrong_idx": json.loads, "energy_diff": json.loads}
        )
        size_list = sorted(df["size"].unique())

        for s_idx, (size, cb, co, marker) in enumerate(zip(size_list_ref, blues, oranges, markers)):
            sub = df[df["size"] == size]
            if sub.empty:
                continue

            wrong_arr  = np.vstack(sub["wrong_idx"].values)
            energy_arr = np.vstack(sub["energy_diff"].values)

            mean_wrong  = wrong_arr.mean(axis=0)[:p_max]
            mean_energy = energy_arr.mean(axis=0)[:p_max]

            bl, = ax.plot(
                np.arange(1, p_max + 1),
                mean_wrong,
                marker=marker,
                color=cb,
                label=f"Size {size}"
            )

            ol, = ax.plot(
                np.arange(1, p_max + 1),
                mean_energy,
                marker=marker,
                color=co,
                label=f"Size {size}"
            )

            if row_idx == 0:
                blue_handles.append(bl)
                orange_handles.append(ol)

        ax.set_title(f"M = {M}", fontsize=fontsize)
        ax.axhline(0, color="k", linestyle="--", linewidth=1)
        ax.set_yscale("log")
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_ylabel("Error", fontsize=fontsize)
        ax.tick_params(axis="both", labelsize=fontsize)
        ax.set_xticks(np.arange(1, p_max+1, 1))

    axes[-1].set_xlabel(r"$p$", fontsize=fontsize)

    abs_title = r"Absolute Error $\epsilon$:"
    rel_title = r"Relative Error $\delta$:"

    handles_abs = [plt.Line2D([], [], linestyle="none")] + blue_handles
    labels_abs  = [abs_title] + [f"Size {s}" for s in size_list_ref]

    leg1 = fig.legend(
        handles_abs, labels_abs,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=len(labels_abs),
        fontsize=fontsize,
        handlelength=0,
        handletextpad=0.5
    )
    fig.add_artist(leg1)

    handles_rel = [plt.Line2D([], [], linestyle="none")] + orange_handles
    labels_rel  = [rel_title] + [f"Size {s}" for s in size_list_ref]
    fig.legend(
        handles_rel, labels_rel,
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        ncol=len(labels_rel),
        fontsize=fontsize,
        handlelength=0,
        handletextpad=0.5
    )

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path+".svg", bbox_inches="tight")
        plt.savefig(output_path+".pdf", bbox_inches="tight")
    plt.show()


def _load_constrained_csv(path: str) -> pd.DataFrame:
    """
    Load a constrained-performance CSV with JSON columns.

    Parameters
    ----------
    path : str

    Returns
    -------
    pd.DataFrame
        Columns include size, wrong_idx, energy_diff, allowed.
    """
    return pd.read_csv(
        path,
        converters={
            "wrong_idx":  json.loads, 
            "energy_diff": json.loads,
            "allowed":    json.loads,
        }
    )
def make_proxies(handles, markersize=8):
    """
    Create legend-only Line2D proxies with larger markers.

    Parameters
    ----------
    handles : list[matplotlib.lines.Line2D]
    markersize : int

    Returns
    -------
    list[matplotlib.lines.Line2D]
    """
    proxies = []
    for h in handles:
        proxies.append(Line2D([], [],
                              marker=h.get_marker(),
                              color=h.get_color(),
                              linestyle='',
                              markersize=markersize))
    return proxies

def plot_grid_performance_constrained(csv_path_template: str, output_path: str = None, p_max: int = None, size_list=None, gammas=(50, 100), num_constraints_list=(1, 3, 5), tau: int = 0, fontsize: int = 16, ylim=(1e-5, 2), use_tex: bool = False):
    """
    Plot a grid of constrained performance (epsilon, delta, nu) over gamma*(#constraints).

    Parameters
    ----------
    csv_path_template : str
        Template with {nc}, {tau}, {gamma}.
    output_path : str | None
        Basename to save .pdf/.svg.
    p_max : int | None
        Max p to display (infer if None).
    size_list : list[int] | None
        Subset of sizes to show.
    gammas : iterable[float]
        Row values.
    num_constraints_list : iterable[int]
        Column values.
    tau : int
    fontsize : int
    ylim : tuple
        y-limits (log scale).
    use_tex : bool
    """
    if use_tex:
        plt.rcParams['text.usetex'] = True

    probe_path = csv_path_template.format(nc=num_constraints_list[0], tau=tau, gamma=gammas[0])
    if not os.path.exists(probe_path):
        raise FileNotFoundError(f"File not found: {probe_path}\n"
                                f"Check that your template matches your filenames.")
    df0 = _load_constrained_csv(probe_path)
    sizes = size_list or sorted(df0["size"].unique())

    n_sizes = len(sizes)
    blues   = cm.Blues(np.linspace(0.4, 0.8, n_sizes))
    oranges = cm.Oranges(np.linspace(0.4, 0.8, n_sizes))
    greens  = cm.Greens(np.linspace(0.4, 0.8, n_sizes))
    markers = ["d", "s", "^", "o", "v", "x", "*"][:n_sizes]

    n_rows = len(gammas)
    n_cols = len(num_constraints_list)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols + 2.5, 3 * n_rows + 0.6),
        sharex=True, sharey=True
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])


    blue_handles, orange_handles, green_handles = [], [], []

    inferred_pmax = p_max
    for i, gamma in enumerate(gammas):
        for j, nc in enumerate(num_constraints_list):
            ax = axes[i, j]

            path = csv_path_template.format(nc=nc, tau=tau, gamma=gamma)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing file for panel (gamma={gamma}, nc={nc}): {path}")
            df = _load_constrained_csv(path)

            if inferred_pmax is None:
                inferred_pmax = len(df["wrong_idx"].iat[0])
            ps = np.arange(1, inferred_pmax + 1)

            for k, size in enumerate(sizes):
                sub = df[df["size"] == size]
                if sub.empty:
                    continue

                wrong_arr   = np.vstack(sub["wrong_idx"].values)
                energy_arr  = np.vstack(sub["energy_diff"].values)
                allowed_arr = np.vstack(sub["allowed"].values)

                mean_wrong   = wrong_arr.mean(axis=0)[:inferred_pmax]   # ε
                mean_energy  = energy_arr.mean(axis=0)[:inferred_pmax]  # Δ
                mean_allowed = allowed_arr.mean(axis=0)[:inferred_pmax] # ν

                h_b, = ax.plot(ps, mean_wrong,
                               marker=markers[k], markersize=4, color=blues[k], linestyle='-')

                h_o, = ax.plot(ps, mean_energy,
                               marker=markers[k], markersize=4, color=oranges[k], linestyle='-')

                h_g, = ax.plot(ps, mean_allowed,
                               marker=markers[k], markersize=4, color=greens[k], linestyle='-')

                if i == 0 and j == 0:
                    blue_handles.append(h_b)
                    orange_handles.append(h_o)
                    green_handles.append(h_g)
                    blue_proxies   = make_proxies(blue_handles,   markersize=8)
                    orange_proxies = make_proxies(orange_handles, markersize=8)
                    green_proxies  = make_proxies(green_handles,  markersize=8)

            ax.axhline(0, color="k", linestyle="--", linewidth=1)
            ax.set_yscale("log")
            if i == n_rows - 1:
                ax.set_xlabel(r"$p$", fontsize=fontsize)
            if j == 0:
                ax.set_ylabel("Error", fontsize=fontsize)

            title_nc = "constraint" if nc == 1 else "constraints"
            ax.set_title(rf"$\gamma={gamma},\quad{nc} $ {title_nc}", fontsize=fontsize)
            ax.set_ylim(ylim)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.set_xticks(np.arange(0, p_max+1, 2))

    title_abs = r"Absolute Error $\epsilon$:"
    title_rel = r"Relative Error $\delta$:"
    title_vio = r"Violations $\nu$:"

    leg1 = fig.legend([Line2D([], [], linestyle="none")] + blue_proxies,
                      [title_abs] + [f"Size {s}" for s in sizes],
                      loc="upper center", bbox_to_anchor=(0.5, 1.12),
                      ncol=n_sizes + 1, fontsize=fontsize,
                      handlelength=0, handletextpad=0.5)
    fig.add_artist(leg1)

    leg2 = fig.legend([Line2D([], [], linestyle="none")] + orange_proxies,
                      [title_rel] + [f"Size {s}" for s in sizes],
                      loc="upper center", bbox_to_anchor=(0.5, 1.06),
                      ncol=n_sizes + 1, fontsize=fontsize,
                      handlelength=0, handletextpad=0.5)
    fig.add_artist(leg2)

    fig.legend([Line2D([], [], linestyle="none")] + green_proxies,
               [title_vio] + [f"Size {s}" for s in sizes],
               loc="upper center", bbox_to_anchor=(0.5, 1),
               ncol=n_sizes + 1, fontsize=fontsize,
               handlelength=0, handletextpad=0.5)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path+".svg", bbox_inches="tight")
        plt.savefig(output_path+".pdf", bbox_inches="tight")
    plt.show()

def plot_thm1(csv_path, p, M, size=16, bins=100, metric='wrong_idx', gap='ratio', save_as=None):
    """
    Bin a chosen error at fixed p against a spectral-gap x and mark x=exp(log M / p)-1.

    Parameters
    ----------
    csv_path : str
        Must contain columns: size, ratio_values(JSON), metric(JSON).
    p : int
        Index into metric lists.
    M : int
    size : int
    bins : int
    metric : {'wrong_idx','energy_diff', ...}
    gap : {'ratio','relative'} | str
        Construct x from ratio_values or use an existing numeric column.
    save_as : str | None
        Basename to save .pdf/.svg.
    """

    df = pd.read_csv(
        csv_path,
        converters={
            "wrong_idx": json.loads,
            "energy_diff": json.loads,
            "ratio_values": json.loads,
        }
    )
    df = df[df["size"] == size].copy()

    def _ratio_from_tuple(t):
        lam1, lam2, _ = t
        return (lam2 - lam1) / lam1 if lam1 != 0 else np.nan

    def _relative_from_tuple(t):
        lam1, lam2, lammax = t
        denom = (lammax - lam1)
        return (lam2 - lam1) / denom if denom != 0 else np.nan

    if gap == 'ratio':
        x_series = df["ratio_values"].apply(_ratio_from_tuple)
    elif gap == 'relative':
        x_series = df["ratio_values"].apply(_relative_from_tuple)
    else:
        if gap in df.columns:
            x_series = pd.to_numeric(df[gap], errors='coerce')
        else:
            raise ValueError(
                f"gap='{gap}' not recognized and column '{gap}' not found. "
                "Use 'ratio', 'relative', or an existing numeric column name."
            )

    df["_x"] = x_series
    df = df[np.isfinite(df["_x"])].copy()
    if df.empty:
        raise ValueError("No valid x-values after constructing from ratio_values (check data).")

    df[f"metric_{p}"] = df[f"{metric}"].apply(lambda lst: lst[p])

    x_min, x_max = df["_x"].min(), df["_x"].max()
    if np.isclose(x_min, x_max):
        eps = 1e-12 if x_min == 0 else 1e-6 * abs(x_min)
        x_min, x_max = x_min - eps, x_max + eps
    bin_edges = np.linspace(x_min, x_max, bins + 1)

    df["ratio_bin"] = pd.cut(df["_x"], bin_edges)
    grouped = df.groupby("ratio_bin", observed=True)[f"metric_{p}"].mean()

    bin_centers = [iv.mid for iv in grouped.index]

    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, grouped.values, marker="o")

    x_val = np.exp(np.log(M) / p) - 1
    plt.axvline(x=x_val, color="red", linestyle="--", label=f"x = {x_val:.4f}")

    plt.xlabel("Spectral Gap Ratio", fontsize=16)
    plt.ylabel(r"Absolute Error $\epsilon$", fontsize=16)
    plt.xlim(0, 1.6 * x_val)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.grid(True)

    if save_as is not None:
        plt.savefig(save_as + ".svg", bbox_inches='tight')
        plt.savefig(save_as + ".pdf", bbox_inches='tight')
    plt.show()


def plot_thm1_multi(csv_path, p_list, M, size=16, bins=100, metric="wrong_idx", gap='ratio', save_as=None, x_lim=(0, 0.16)):
    """
    Plot the binned error vs spectral-gap x for multiple p and mark each x=exp(log M / p)-1.

    Parameters
    ----------
    csv_path : str
        Must contain columns: size, ratio_values(JSON), metric(JSON).
    p_list : iterable[int]
        p-indices to plot.
    M : int
    size : int
    bins : int
    metric : {'wrong_idx','energy_diff', ...}
    gap : {'ratio','relative'} | str
        Construct x from ratio_values or use an existing numeric column.
    save_as : str | None
        Basename to save .pdf/.svg.
    x_lim : tuple | None
        x-axis limits.
    """
    df = pd.read_csv(
        csv_path,
        converters={
            "wrong_idx": json.loads,
            "energy_diff": json.loads,
            "ratio_values": json.loads,
        }
    )
    df = df[df["size"] == size].copy()

    def _ratio_from_tuple(t):
        lam1, lam2, _ = t
        return (lam2 - lam1) / lam1 if lam1 != 0 else np.nan

    def _relative_from_tuple(t):
        lam1, lam2, lammax = t
        denom = (lammax - lam1)
        return (lam2 - lam1) / denom if denom != 0 else np.nan

    if gap == 'ratio':
        x_series = df["ratio_values"].apply(_ratio_from_tuple)
    elif gap == 'relative':
        x_series = df["ratio_values"].apply(_relative_from_tuple)
    else:
        if gap in df.columns:
            x_series = pd.to_numeric(df[gap], errors='coerce')
        else:
            raise ValueError(
                f"gap='{gap}' not recognized and column '{gap}' not found. "
                "Use 'ratio', 'relative', or an existing numeric column name."
            )

    df["_x"] = x_series
    df = df[np.isfinite(df["_x"])].copy()
    if df.empty:
        raise ValueError("No data left after filtering/constructing the x-axis (check ratio_values and size).")

    x_min, x_max = df["_x"].min(), df["_x"].max()
    if np.isclose(x_min, x_max):
        eps = 1e-12 if x_min == 0 else 1e-6 * abs(x_min)
        x_min, x_max = x_min - eps, x_max + eps
    ratio_bins = np.linspace(x_min, x_max, bins + 1)
    df["ratio_bin"] = pd.cut(df["_x"], ratio_bins)

    import numpy as _np
    colors = cm.Blues(_np.linspace(0.4, 0.9, len(p_list)))

    plt.figure(figsize=(10, 6))

    for p, color in zip(p_list, colors):
        df[f"metric_{p}"] = df[metric].apply(lambda lst: lst[p])

        grouped = df.groupby("ratio_bin", observed=True)[f"metric_{p}"].mean()

        bin_centers = [interval.mid for interval in grouped.index]
        plt.plot(
            bin_centers,
            grouped.values,
            marker="o",
            label=rf"${p}$",
            color=color
        )

        x_val = np.exp(np.log(M) / p) - 1
        plt.axvline(x=x_val, color=color, linestyle="--")

    plt.xlabel(r"Spectral Gap Ratio", fontsize=16)
    plt.ylabel(r"Absolute Error $\epsilon$", fontsize=16)

    if x_lim is None:
        plt.xlim(0, df["_x"].max())
    else:
        plt.xlim(x_lim)

    plt.grid(True)
    ax = plt.gca()
    ax.tick_params(axis="both", which="major", labelsize=16)

    plt.legend(title=r"$p$", loc="upper right", fontsize=14, title_fontsize=14)

    if save_as:
        plt.savefig(save_as + '.svg', bbox_inches="tight")
        plt.savefig(save_as + '.pdf', bbox_inches="tight")
    plt.show()
