from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Gate:
    name: str
    qubits: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", self.name.upper())
        object.__setattr__(self, "qubits", tuple(self.qubits))


@dataclass(frozen=True)
class PathSummary:
    median_c2: float
    mean_c2: float
    max_c2: float
    median_cancellation: float
    mean_cancellation: float
    max_cancellation: float
    num_boundaries: int
    paths_per_input: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def gate(name: str, *qubits: int) -> Gate:
    return Gate(name, qubits)


def h_layer(n_qubits: int) -> list[Gate]:
    return [gate("H", q) for q in range(n_qubits)]


def s_layer(n_qubits: int) -> list[Gate]:
    return [gate("S", q) for q in range(n_qubits)]


def cz_chain(n_qubits: int) -> list[Gate]:
    return [gate("CZ", q, q + 1) for q in range(n_qubits - 1)]


def simple_clifford_path_family(n_qubits: int, depth: int) -> list[Gate]:
    gates: list[Gate] = []
    for _ in range(depth):
        gates.extend(h_layer(n_qubits))
        gates.extend(s_layer(n_qubits))
        gates.extend(h_layer(n_qubits))
        gates.extend(s_layer(n_qubits))
    return gates


def brickwork_clifford_family(n_qubits: int, depth: int) -> list[Gate]:
    gates: list[Gate] = []
    for _ in range(depth):
        gates.extend(h_layer(n_qubits))
        gates.extend(cz_chain(n_qubits))
        gates.extend(h_layer(n_qubits))
        gates.extend(s_layer(n_qubits))
        gates.extend(cz_chain(n_qubits))
    return gates


def count_hadamards(gates: Iterable[Gate]) -> int:
    return sum(1 for item in gates if item.name == "H")


def unitary_from_gates(n_qubits: int, gates: Iterable[Gate]) -> np.ndarray:
    gate_list = list(gates)
    dim = 1 << n_qubits
    unitary = np.zeros((dim, dim), dtype=np.complex128)
    for initial in range(dim):
        state: dict[int, complex] = {initial: 1.0 + 0.0j}
        for item in gate_list:
            next_state: dict[int, complex] = {}
            for basis, amp in state.items():
                for new_basis, new_amp in _apply_gate_to_basis(n_qubits, basis, amp, item):
                    next_state[new_basis] = next_state.get(new_basis, 0.0j) + new_amp
            state = next_state
        for output, amp in state.items():
            unitary[output, initial] = amp
    return unitary


def operator_magic_from_gates(n_qubits: int, gates: Iterable[Gate]) -> float:
    unitary = unitary_from_gates(n_qubits, gates)
    return operator_magic(unitary)


def pauli_path_coherence_from_gates(n_qubits: int, gates: Iterable[Gate]) -> float:
    unitary = unitary_from_gates(n_qubits, gates)
    return pauli_path_coherence(unitary)


def pauli_path_coherence(unitary: np.ndarray) -> float:
    transfer = pauli_transfer_matrix(unitary)
    dim_pauli = transfer.shape[0]
    fourth_moment = float(np.sum(transfer**4))
    value = -math.log(max(fourth_moment / dim_pauli, 1e-300))
    return 0.0 if abs(value) < 1e-12 else value


def pauli_transfer_matrix(unitary: np.ndarray) -> np.ndarray:
    dim = unitary.shape[0]
    n_qubits = int(round(math.log2(dim)))
    if unitary.shape != (dim, dim) or (1 << n_qubits) != dim:
        raise ValueError("unitary must be a square 2^n by 2^n matrix")

    paulis = list(product(range(4), repeat=n_qubits))
    matrices = [_pauli_matrix(pauli) for pauli in paulis]
    transfer = np.zeros((len(paulis), len(paulis)), dtype=float)
    for p_index, p_matrix in enumerate(matrices):
        evolved = unitary.conj().T @ p_matrix @ unitary
        for q_index, q_matrix in enumerate(matrices):
            value = np.trace(q_matrix @ evolved) / dim
            if abs(value.imag) > 1e-10:
                raise ValueError("Pauli transfer matrix has unexpected imaginary entry")
            transfer[q_index, p_index] = float(value.real)
    return transfer


def operator_magic(unitary: np.ndarray) -> float:
    dim = unitary.shape[0]
    n_qubits = int(round(math.log2(dim)))
    if unitary.shape != (dim, dim) or (1 << n_qubits) != dim:
        raise ValueError("unitary must be a square 2^n by 2^n matrix")
    return stabilizer_renyi_entropy(choi_state(unitary), 2 * n_qubits)


def choi_state(unitary: np.ndarray) -> np.ndarray:
    dim = unitary.shape[0]
    n_qubits = int(round(math.log2(dim)))
    state = np.zeros(dim * dim, dtype=np.complex128)
    scale = 1.0 / math.sqrt(dim)
    for x in range(dim):
        for y in range(dim):
            state[x | (y << n_qubits)] += scale * unitary[y, x]
    return state


def stabilizer_renyi_entropy(state: np.ndarray, n_qubits: int) -> float:
    expected_dim = 1 << n_qubits
    if len(state) != expected_dim:
        raise ValueError("state length does not match n_qubits")
    norm = np.linalg.norm(state)
    if norm == 0:
        raise ValueError("state must be nonzero")
    psi = state / norm
    fourth_moment = 0.0
    for pauli in product(range(4), repeat=n_qubits):
        value = _pauli_expectation(psi, pauli)
        fourth_moment += float(abs(value) ** 4)
    normalized = fourth_moment / expected_dim
    entropy = -math.log(max(normalized, 1e-300))
    return 0.0 if abs(entropy) < 1e-12 else entropy


def boundary_path_amplitudes(
    n_qubits: int,
    gates: Iterable[Gate],
    initial: int,
) -> dict[int, list[complex]]:
    paths: list[tuple[int, complex]] = [(initial, 1.0 + 0.0j)]
    for item in gates:
        next_paths: list[tuple[int, complex]] = []
        for basis, amp in paths:
            next_paths.extend(_apply_gate_to_basis(n_qubits, basis, amp, item))
        paths = next_paths
    grouped: dict[int, list[complex]] = {}
    for basis, amp in paths:
        grouped.setdefault(basis, []).append(amp)
    return grouped


def boundary_measures(
    amplitudes: Iterable[complex],
    epsilon: float = 1e-12,
) -> tuple[float, float]:
    amps = np.asarray(list(amplitudes), dtype=np.complex128)
    if amps.size == 0:
        raise ValueError("at least one path amplitude is required")
    weights = np.abs(amps) ** 2
    total_weight = float(np.sum(weights))
    if total_weight <= 0:
        raise ValueError("path amplitudes must have positive total weight")
    probabilities = weights / total_weight
    c2 = -math.log(float(np.sum(probabilities**2)))
    cancellation = math.log(float(np.sum(np.abs(amps))) / max(abs(np.sum(amps)), epsilon))
    if abs(c2) < 1e-12:
        c2 = 0.0
    if abs(cancellation) < 1e-12:
        cancellation = 0.0
    return c2, cancellation


def path_summary(
    n_qubits: int,
    gates: Iterable[Gate],
    epsilon: float = 1e-12,
    max_total_paths: int | None = None,
) -> PathSummary:
    gate_list = list(gates)
    dim = 1 << n_qubits
    paths_per_input = 1 << count_hadamards(gate_list)
    total_paths = dim * paths_per_input
    if max_total_paths is not None and total_paths > max_total_paths:
        raise ValueError(f"path enumeration would create {total_paths} paths")

    c2_values: list[float] = []
    cancellation_values: list[float] = []
    for initial in range(dim):
        grouped = boundary_path_amplitudes(n_qubits, gate_list, initial)
        for amplitudes in grouped.values():
            c2, cancellation = boundary_measures(amplitudes, epsilon=epsilon)
            c2_values.append(c2)
            cancellation_values.append(cancellation)

    c2_array = np.asarray(c2_values, dtype=float)
    cancellation_array = np.asarray(cancellation_values, dtype=float)
    return PathSummary(
        median_c2=float(np.median(c2_array)),
        mean_c2=float(np.mean(c2_array)),
        max_c2=float(np.max(c2_array)),
        median_cancellation=float(np.median(cancellation_array)),
        mean_cancellation=float(np.mean(cancellation_array)),
        max_cancellation=float(np.max(cancellation_array)),
        num_boundaries=len(c2_values),
        paths_per_input=paths_per_input,
    )


def linear_fit(xs: Iterable[float], ys: Iterable[float]) -> dict[str, float]:
    x = np.asarray(list(xs), dtype=float)
    y = np.asarray(list(ys), dtype=float)
    if x.size != y.size or x.size < 2:
        raise ValueError("linear_fit needs at least two paired points")
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2)}


def _apply_gate_to_basis(
    n_qubits: int,
    basis: int,
    amp: complex,
    item: Gate,
) -> list[tuple[int, complex]]:
    name = item.name
    if name in {"H", "S", "SDG", "T", "TDG", "X", "Z"}:
        if len(item.qubits) != 1:
            raise ValueError(f"{name} expects one qubit")
        q = item.qubits[0]
        _check_qubit(n_qubits, q)
        bit = (basis >> q) & 1
        if name == "H":
            cleared = basis & ~(1 << q)
            scale = amp / math.sqrt(2.0)
            return [
                (cleared, scale),
                (cleared | (1 << q), scale if bit == 0 else -scale),
            ]
        if name == "S":
            return [(basis, amp * (1j if bit else 1.0))]
        if name == "SDG":
            return [(basis, amp * (-1j if bit else 1.0))]
        if name == "T":
            return [(basis, amp * (np.exp(1j * math.pi / 4.0) if bit else 1.0))]
        if name == "TDG":
            return [(basis, amp * (np.exp(-1j * math.pi / 4.0) if bit else 1.0))]
        if name == "X":
            return [(basis ^ (1 << q), amp)]
        if name == "Z":
            return [(basis, -amp if bit else amp)]

    if name in {"CNOT", "CX", "CZ"}:
        if len(item.qubits) != 2:
            raise ValueError(f"{name} expects two qubits")
        q0, q1 = item.qubits
        _check_qubit(n_qubits, q0)
        _check_qubit(n_qubits, q1)
        if q0 == q1:
            raise ValueError(f"{name} qubits must be distinct")
        b0 = (basis >> q0) & 1
        b1 = (basis >> q1) & 1
        if name in {"CNOT", "CX"}:
            return [(basis ^ (1 << q1), amp)] if b0 else [(basis, amp)]
        return [(basis, -amp if (b0 and b1) else amp)]

    raise ValueError(f"unsupported gate: {item.name}")


def _pauli_expectation(state: np.ndarray, pauli: tuple[int, ...]) -> complex:
    xmask = 0
    for q, code in enumerate(pauli):
        if code in {1, 2}:
            xmask |= 1 << q

    total = 0.0 + 0.0j
    for basis, amp in enumerate(state):
        phase = 1.0 + 0.0j
        for q, code in enumerate(pauli):
            bit = (basis >> q) & 1
            if code == 2:
                phase *= 1j if bit == 0 else -1j
            elif code == 3 and bit:
                phase *= -1.0
        total += np.conjugate(state[basis ^ xmask]) * amp * phase
    return total


def _pauli_matrix(pauli: tuple[int, ...]) -> np.ndarray:
    single = [
        np.array([[1, 0], [0, 1]], dtype=np.complex128),
        np.array([[0, 1], [1, 0]], dtype=np.complex128),
        np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        np.array([[1, 0], [0, -1]], dtype=np.complex128),
    ]
    matrix = np.array([[1]], dtype=np.complex128)
    for code in reversed(pauli):
        matrix = np.kron(matrix, single[code])
    return matrix


def _check_qubit(n_qubits: int, qubit: int) -> None:
    if qubit < 0 or qubit >= n_qubits:
        raise ValueError(f"qubit index {qubit} is outside 0..{n_qubits - 1}")
