import torch as t
import numpy as np
import json
import os
from numpy.random import uniform as urng


pauli_X = t.tensor([[0, 1],[1, 0]], dtype=t.double)
pauli_Y = t.tensor([[0, -1],[1,  0]], dtype=t.double) * 1j
pauli_Z = t.tensor([[1, 0],[0,-1]], dtype=t.double)

def generate_sequence(seed: int, gate_count: int, n_sequences: int, min_depth: int, max_depth: int):
    # generates n_sequences with random depth in range of min/max_depth
    # each sequence is like [0100100100] for gate set of 2
    # each number in gate_seq corresponds with particular gate, e.g. [0, 1, 2, 3] - four gates
    # cirquit model is the set of sequences of numbers, where numbers define particular gate
    np.random.seed(seed)
    sequence = []
    for _ in range(n_sequences):
        depth = np.random.randint(low = min_depth, high= max_depth, size=[1])
        gate_set = np.random.randint(low = 0, high = gate_count, size = depth)
        sequence.append(gate_set)
    return sequence

def load_sequences_from_json(file_path: str):
    base_dir = os.path.dirname(__file__)
    full_path = os.path.join(base_dir, file_path)

    with open(full_path, "r") as f:
        data = json.load(f)
    return data["sequences"]

# ---------- Базовые утилиты для 2-кубитного случая ----------

def kron_n(*ops: t.Tensor) -> t.Tensor:
    """Кронекер-произведение произвольного числа матриц."""
    M = ops[0]
    for A in ops[1:]:
        M = t.kron(M.contiguous(), A.contiguous())
    return M

def embed_1q(U_2x2: t.Tensor, target: int, n_qubits: int = 2) -> t.Tensor:
    I2 = t.eye(2, dtype=U_2x2.dtype)
    ops = []
    for q in range(n_qubits):
        ops.append(U_2x2 if q == target else I2)
    return kron_n(*ops)

def U_CNOT_01() -> t.Tensor:
    """Стандартный CNOT с управляющим 0-м кубитом и управляемым 1-м."""
    return t.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=t.cdouble)

# ---------- Однокубитные функции ----------

def U_rotation(theta, phi, delta):
    return t.tensor(
        [
            [np.cos(delta/2)-1j*np.cos(theta)*np.sin(delta/2),
             -1j*np.sin(theta)*np.sin(delta/2)*np.exp(-1j*phi)],
            [-1j*np.sin(theta)*np.sin(delta/2)*np.exp(1j*phi),
             np.cos(delta/2)+1j*np.cos(theta)*np.sin(delta/2)],
        ],
        dtype=t.cdouble,
    )

def u_to_g(u):
    """
    Transforms unitary operator U into superoperator G
    """
    u_conj = u.conj()
    g = t.einsum("ij,kl -> ikjl", u, u_conj).reshape(
        u.size(0) * u_conj.size(0),
        u.size(1) * u_conj.size(1),
    )
    return g

def tensor_u_to_g(n_tensors: int, u: t.Tensor, r, d_model):       
    u_tensor = t.reshape(u, [n_tensors, r, d_model, d_model])
    g = t.einsum("ajmn,ajkl -> amknl", u_tensor, u_tensor.conj()).reshape([n_tensors, d_model**2, d_model**2])
    return g

def G_depol(p_depol = 0, d = 2):
    kraus_depol = t.stack([
        np.sqrt(1 - p_depol) * t.eye(d),
        np.sqrt(p_depol/3) * pauli_X,
        np.sqrt(p_depol/3) * pauli_Y,
        np.sqrt(p_depol/3) * pauli_Z,
    ])
    return t.tensor(
        t.einsum("amn,akl -> mknl", kraus_depol, kraus_depol.conj()).reshape([d**2, d**2]),
        dtype=t.cdouble,
    )

def G_ampl_damp(p_ampl_damp = 0, d = 2):
    kraus_ampl_damp = t.stack([
        t.tensor([[1, 0], [0, np.sqrt(1 - p_ampl_damp)]], dtype=t.cdouble),
        t.tensor([[0, np.sqrt(p_ampl_damp)], [0, 0]], dtype=t.cdouble),
    ])
    return t.tensor(
        t.einsum("amn,akl -> mknl", kraus_ampl_damp, kraus_ampl_damp.conj()).reshape([d**2, d**2]),
        dtype=t.cdouble,
    )

def G_phase_damp(p_phase_damp = 0, d = 2):
    kraus_phase_damp = t.stack([
        np.sqrt(1 - p_phase_damp)*t.eye(d),
        np.sqrt(p_phase_damp)*pauli_Z,
    ])
    return t.tensor(
        t.einsum("amn,akl -> mknl", kraus_phase_damp, kraus_phase_damp.conj()).reshape([d**2, d**2]),
        dtype=t.cdouble,
    )

def G_rotation_gate(theta, phi, delta,
        p_depol = 0, p_ampl_damp = 0, p_phase_damp = 0, d_theta = 0, d_phi = 0, d_delta = 0):
    theta = theta + urng(-d_theta, d_theta)
    phi = phi + urng(-d_phi, d_phi)
    delta = delta + urng(-d_delta, d_delta)

    g_sqrt = u_to_g(U_rotation(theta, phi, delta))
    g_sqrt = t.einsum("ij, jk -> ik", G_ampl_damp(p_ampl_damp), g_sqrt)
    g_sqrt = t.einsum("ij, jk -> ik", G_phase_damp(p_phase_damp), g_sqrt)
    g_sqrt = t.einsum("ij, jk -> ik", G_depol(p_depol), g_sqrt)

    return g_sqrt

def build_gate_set_1q(cfg):
    # [sqrt(X), sqrt(Y)]
    gate_set = t.stack([
        G_rotation_gate(np.pi/2, 0, np.pi/2,
            cfg.noise.p_depol, cfg.noise.p_ampl_damp, cfg.noise.p_phase_damp, cfg.noise.d_theta, cfg.noise.d_phi, cfg.noise.d_delta),
        G_rotation_gate(np.pi/2, np.pi/2, np.pi/2,
            cfg.noise.p_depol, cfg.noise.p_ampl_damp, cfg.noise.p_phase_damp, cfg.noise.d_theta, cfg.noise.d_phi, cfg.noise.d_delta),
    ])
    return gate_set

# ---------- 2-кубитные гейты ----------

def G_from_kraus(kraus: t.Tensor) -> t.Tensor:
    # Универсальный конструктор супероператора из набора Краусов
    # kraus.shape = (K, d, d) -> G.shape = (d^2, d^2)
    
    d = kraus.size(-1)
    return t.einsum("amn,akl -> mknl", kraus, kraus.conj()).reshape(d**2, d**2).to(t.cdouble)

def G_depol_on_qubit(p_depol: float, target: int, n_qubits: int = 2) -> t.Tensor:
    
    # Деполяризация на целевом кубите
    # Краусы: sqrt(1-p) I, sqrt(p/3){X, Y, Z}
    #   target=0: {X I, Y I, Z I}
    #   target=1: {I X, I Y, I Z}
    
    I2 = t.eye(2, dtype=t.cdouble)
    X = pauli_X.to(t.cdouble)
    Y = pauli_Y.to(t.cdouble)
    Z = pauli_Z.to(t.cdouble)

    if target == 0:
        ops = [X, Y, Z]
        kron_fun = lambda A: kron_n(A, I2)
    else:
        ops = [X, Y, Z]
        kron_fun = lambda A: kron_n(I2, A)

    K0 = np.sqrt(1 - p_depol) * kron_fun(I2)
    K_list = [K0]
    for A in ops:
        K_list.append(np.sqrt(p_depol/3) * kron_fun(A))
    kraus = t.stack(K_list)  # (4, 4, 4) для двух кубит
    return G_from_kraus(kraus)

def G_ampl_damp_on_qubit(p_ampl: float, target: int, n_qubits: int = 2) -> t.Tensor:
    # Амплитудная релаксация
    
    K0_1q = t.tensor([[1, 0], [0, np.sqrt(1 - p_ampl)]], dtype=t.cdouble)
    K1_1q = t.tensor([[0, np.sqrt(p_ampl)], [0, 0]], dtype=t.cdouble)

    if target == 0:
        K0 = kron_n(K0_1q, t.eye(2, dtype=t.cdouble))
        K1 = kron_n(K1_1q, t.eye(2, dtype=t.cdouble))
    else:
        K0 = kron_n(t.eye(2, dtype=t.cdouble), K0_1q)
        K1 = kron_n(t.eye(2, dtype=t.cdouble), K1_1q)

    kraus = t.stack([K0, K1])
    return G_from_kraus(kraus)

def G_phase_damp_on_qubit(p_phase: float, target: int, n_qubits: int = 2) -> t.Tensor:
    # Фазовая релаксация

    I2c = t.eye(2, dtype=t.cdouble)
    Zc = pauli_Z.to(t.cdouble)

    K0_1q = np.sqrt(1 - p_phase) * I2c
    K1_1q = np.sqrt(p_phase) * Zc

    if target == 0:
        K0 = kron_n(K0_1q, I2c)
        K1 = kron_n(K1_1q, I2c)
    else:
        K0 = kron_n(I2c, K0_1q)
        K1 = kron_n(I2c, K1_1q)

    kraus = t.stack([K0, K1])
    return G_from_kraus(kraus)

def G_rotation_gate_2q_on_qubit(
    target: int,
    theta: float,
    phi: float,
    delta: float,
    p_depol: float = 0.0,
    p_ampl_damp: float = 0.0,
    p_phase_damp: float = 0.0,
    d_theta: float = 0.0,
    d_phi: float = 0.0,
    d_delta: float = 0.0,
) -> t.Tensor:
    # однокубитный поворот на выбранном кубите в двухкубитном пространстве + шум
    
    theta = theta + urng(-d_theta, d_theta)
    phi   = phi   + urng(-d_phi, d_phi)
    delta = delta + urng(-d_delta, d_delta)

    # Однокубитный U
    U_1q = U_rotation(theta, phi, delta)  # 2x2

    # Действуем на целевой кубит
    U_2q = embed_1q(U_1q, target, n_qubits=2)  # 4x4
    
    # Получаем супероператор поворота на целевой кубит
    g = u_to_g(U_2q)  # 16x16

    # Умножаем супероператор шума на супероператор поворота для целевого кубита
    if p_ampl_damp > 0:
        g = t.einsum("ij,jk -> ik", G_ampl_damp_on_qubit(p_ampl_damp, target), g)
    if p_phase_damp > 0:
        g = t.einsum("ij,jk -> ik", G_phase_damp_on_qubit(p_phase_damp, target), g)
    if p_depol > 0:
        g = t.einsum("ij,jk -> ik", G_depol_on_qubit(p_depol, target), g)

    return g

def G_depol_2q(p_depol: float) -> t.Tensor:
    """
    Двухкубитный деполяризующий канал
    Краусы: (1-p)I, p/15 {X I, X X, X Y, X Z,
                          Y I, Y X, Y Y, Y Z,
                          Z I, Z X, Z Y, Z Z
                          I X, I Y, I Z}
    """
    I2 = t.eye(2, dtype=t.cdouble)
    X = pauli_X.to(t.cdouble)
    Y = pauli_Y.to(t.cdouble)
    Z = pauli_Z.to(t.cdouble)

    twoq_paulis = []
    for P1 in (I2, X, Y, Z):
        for P2 in (I2, X, Y, Z):
            P = kron_n(P1, P2)  # 4x4
            twoq_paulis.append(P)

    I4 = twoq_paulis[0]  # I⊗I

    # Kraus-операторы
    K_list = []
    K_list.append(t.sqrt(t.tensor(1 - p_depol, dtype=t.double)) * I4)

    non_identity = twoq_paulis[1:]
    scale = t.sqrt(t.tensor(p_depol / 15.0, dtype=t.double))
    for P in non_identity:
        K_list.append(scale * P)

    kraus = t.stack(K_list)  # (16, 4, 4)
    return G_from_kraus(kraus)

def G_ampl_corr_2q(p_corr: float) -> t.Tensor:
    """
    Коррелированная двухкубитная амплитудная релаксация
    """
    dtype = t.cdouble
    p_t = t.tensor(p_corr, dtype=t.double)

    K0 = t.zeros((4, 4), dtype=dtype)
    K1 = t.zeros((4, 4), dtype=dtype)

    # Базис: |00>, |01>, |10>, |11>
    K0[0, 0] = 1.0
    K0[1, 1] = 1.0
    K0[2, 2] = 1.0
    K0[3, 3] = t.sqrt(1.0 - p_t)

    # |00><11|
    K1[0, 3] = t.sqrt(p_t)

    kraus = t.stack([K0, K1])   # (2, 4, 4)
    return G_from_kraus(kraus)  # 16x16

def G_ampl_full_2q(p_local: float, p_corr: float) -> t.Tensor:
    """
    Полная двухкубитная амплитудная релаксация
    """
    # Сначала локальная релаксация на каждом кубите
    G_loc_0 = G_ampl_damp_on_qubit(p_local, target=0)
    G_loc_1 = G_ampl_damp_on_qubit(p_local, target=1)
    G_local = G_loc_1 @ G_loc_0   # сначала qubit 0, потом qubit 1

    # Затем коррелированная компонента
    if p_corr > 0.0:
        G_corr = G_ampl_corr_2q(p_corr)
        G_full = G_corr @ G_local
    else:
        G_full = G_local

    return G_full

def G_phase_corr_2q(p: float) -> t.Tensor:
    """
    Коррелированный двухкубитный фазовый шум: ZxZ.
    """
    p_t = t.tensor(p, dtype=t.double)

    I2 = t.eye(2, dtype=t.cdouble)
    Z = pauli_Z.to(t.cdouble)
    I4 = kron_n(I2, I2)
    Z2 = kron_n(Z, Z)

    K0 = t.sqrt(1.0 - p_t) * I4
    K1 = t.sqrt(p_t) * Z2

    kraus = t.stack([K0, K1])
    return G_from_kraus(kraus)

def G_phase_full_2q(p_local: float, p_corr: float) -> t.Tensor:
    """
    Полная двухкубитная фазовая релаксация:
      - локальная фаза на каждом кубите
      - коррелированный ZxZ шум
    """
    # локальная фаза
    G_loc_0 = G_phase_damp_on_qubit(p_local, 0)
    G_loc_1 = G_phase_damp_on_qubit(p_local, 1)
    G_local = G_loc_1 @ G_loc_0

    if p_corr > 0.0:
        G_corr = G_phase_corr_2q(p_corr)
        return G_corr @ G_local
    else:
        return G_local

def G_CNOT_2q(
    p_depol_2q: float = 0.0,
    p_ampl_local: float = 0.0,
    p_ampl_corr: float = 0.0,
    p_phase_local: float = 0.0,
    p_phase_corr: float = 0.0,
) -> t.Tensor:

    # Идеальный CNOT
    U = U_CNOT_01()
    g = u_to_g(U)

    # Амплитудная релаксация (локальная + коррелированная)
    if p_ampl_local > 0.0 or p_ampl_corr > 0.0:
        G_ampl = G_ampl_full_2q(p_ampl_local, p_ampl_corr)   # 16x16
        g = G_ampl @ g

    # Фазовая релаксация (локальная + коррелированная)
    if p_phase_local > 0.0 or p_phase_corr > 0.0:
        G_phase = G_phase_full_2q(p_phase_local, p_phase_corr)  # 16x16
        g = G_phase @ g

    # Двухкубитный деполяризующий канал
    if p_depol_2q > 0.0:
        G_dep = G_depol_2q(p_depol_2q)   # 16x16
        g = G_dep @ g

    return g

def build_gate_set_2q(cfg):
    """
    Пример набора гейтов для двух кубитов:
    [X_on_0q, Y_on_0q, X_on_1q, Y_on_1q, CNOT]
    """
    gates = [
        G_rotation_gate_2q_on_qubit(0, np.pi/2, 0,       np.pi/2,
            cfg.noise.p_depol, cfg.noise.p_ampl_damp, cfg.noise.p_phase_damp, cfg.noise.d_theta, cfg.noise.d_phi, cfg.noise.d_delta),  # sqrt(X) on qubit 0
        G_rotation_gate_2q_on_qubit(0, np.pi/2, np.pi/2, np.pi/2,
            cfg.noise.p_depol, cfg.noise.p_ampl_damp, cfg.noise.p_phase_damp, cfg.noise.d_theta, cfg.noise.d_phi, cfg.noise.d_delta),  # sqrt(Y) on qubit 0
        G_rotation_gate_2q_on_qubit(1, np.pi/2, 0,       np.pi/2,
            cfg.noise.p_depol, cfg.noise.p_ampl_damp, cfg.noise.p_phase_damp, cfg.noise.d_theta, cfg.noise.d_phi, cfg.noise.d_delta),  # sqrt(X) on qubit 1
        G_rotation_gate_2q_on_qubit(1, np.pi/2, np.pi/2, np.pi/2,
            cfg.noise.p_depol, cfg.noise.p_ampl_damp, cfg.noise.p_phase_damp, cfg.noise.d_theta, cfg.noise.d_phi, cfg.noise.d_delta),  # sqrt(Y) on qubit 1
        G_CNOT_2q(cfg.noise.p_depol, cfg.noise.p_ampl_damp, cfg.noise.p_phase_damp),
    ]
    return t.stack(gates)

def rho_prep(dim: int, spam_cfg):
    # 1-кубитное состояние с ошибкой приготовления
    rho_1q = t.tensor(
        [[1 - spam_cfg.prep_err, 0],
         [0, spam_cfg.prep_err]],
        dtype=t.cdouble,
        )
    if dim == 2:
        return rho_1q.flatten()
    
    rho_2q = t.kron(rho_1q, rho_1q)
    return rho_2q.flatten()

def meas_proj(dim: int, spam_cfg):
    
    P_meas_1q = t.tensor(
            [[1 - spam_cfg.meas_01_err, 0],
             [0, spam_cfg.meas_10_err]],
            dtype=t.cdouble,
    )
    
    I2 = t.eye(2, dtype=t.cdouble)
    E0 = P_meas_1q
    E1 = I2 - E0
    
    if dim == 2:
        return t.stack([E0.flatten(), E1.flatten()], dim=0)

    E00 = t.kron(E0, E0)
    E01 = t.kron(E0, E1)
    E10 = t.kron(E1, E0)
    I_sys = t.eye(4, dtype=t.cdouble)
    E11 = I_sys - (E00 + E01 + E10)

    return t.stack([E00.flatten(), E01.flatten(), E10.flatten(), E11.flatten()], dim=0)

def generate_g(gate_sequence, gate_set: t.Tensor) -> t.Tensor:
    """
    gate_set: (n_gates, d2, d2) complex
    returns: (d2, d2) complex
    """
    d2 = gate_set.size(1)
    device = gate_set.device
    dtype = gate_set.dtype  # expected complex

    if gate_sequence is None or len(gate_sequence) == 0:
        return t.eye(d2, dtype=dtype, device=device)

    if isinstance(gate_sequence, np.ndarray):
        seq = gate_sequence.tolist()
    else:
        seq = list(gate_sequence)

    G_acc = gate_set[seq[0]]
    for idx in seq[1:]:
        G_acc = gate_set[idx] @ G_acc
    return G_acc