# Quantum_wdw_qtransform



import random
import numpy as np

random.seed(39)
np.random.seed(39)

from pathlib import Path
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector



FILES = (
    Path("/data/loto7_4718_k90.csv"),
    Path("/data/loto7_4718_k90_loto_2981.csv"),
    Path("/data/loto7_4718_k90_loto_plus_1737.csv"),
)



NUM_BALLS = 39
DRAW_SIZE = 7
N_QUBITS = 6
N_STATES = 2 ** N_QUBITS


def load_draws(csv_path: Path) -> np.ndarray:
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.replace(";", ",").split(",") if p.strip()]
            nums = [int(x) for x in parts[:DRAW_SIZE]]
            if len(nums) != DRAW_SIZE:
                raise ValueError(f"Neispravan red u {csv_path}: {line}")
            if any(n < 1 or n > NUM_BALLS for n in nums) or len(set(nums)) != DRAW_SIZE:
                raise ValueError(f"Nelegalna kombinacija u {csv_path}: {nums}")
            rows.append(sorted(nums))
    if len(rows) < 100:
        raise ValueError(f"Premalo kola u {csv_path}: {len(rows)}")
    return np.array(rows, dtype=np.int32)


def draws_to_binary_matrix(draws: np.ndarray) -> np.ndarray:
    t_len = len(draws)
    mat = np.zeros((t_len, NUM_BALLS), dtype=np.float64)
    rows_idx = np.arange(t_len)[:, None]
    mat[rows_idx, draws - 1] = 1.0
    return mat


def compute_causal_q_whitened_features(
    binary_mat: np.ndarray, q_values=(2.0, 4.0, 8.0), max_win: int = 48
) -> np.ndarray:
    t_len, n_balls = binary_mat.shape
    raw_energy = np.zeros((t_len, n_balls), dtype=np.float64)

    wavelets = []
    for q in q_values:
        w_len = min(max_win, max(8, int(round(q * 5))))
        tau = np.arange(w_len, dtype=np.float64)
        f0 = q / (2.0 * np.pi * max(4.0, w_len / 2.5))
        sigma = w_len / (2.5 * np.sqrt(2.0))
        env = np.exp(-0.5 * ((tau - (w_len - 1)) / sigma) ** 2)
        env /= np.sum(env)
        phase = 2.0 * np.pi * f0 * (tau - (w_len - 1))
        kernel = env * np.exp(-1j * phase)
        wavelets.append((w_len, kernel))

    for t in range(t_len):
        acc = np.zeros(n_balls, dtype=np.float64)
        for w_len, kernel in wavelets:
            if t + 1 < w_len:
                sub = binary_mat[: t + 1]
                k_sub = kernel[w_len - (t + 1) :]
                k_sub = k_sub / (np.sum(np.abs(k_sub)) + 1e-12)
            else:
                sub = binary_mat[t + 1 - w_len : t + 1]
                k_sub = kernel
            proj = np.abs(k_sub @ sub)
            acc += proj ** 2
        raw_energy[t] = acc / len(wavelets)

    whitened = np.zeros_like(raw_energy)
    cum_psd = np.zeros(n_balls, dtype=np.float64)
    eps = 1e-8
    for t in range(t_len):
        row_t = raw_energy[t]
        if t == 0:
            cum_psd = row_t.copy() + eps
            whitened[t] = row_t / cum_psd
        else:
            psd_est = (cum_psd / float(t)) + eps
            whitened[t] = row_t / psd_est
            cum_psd += row_t

    return whitened


def encode_context_to_3qubits(feature_vec: np.ndarray) -> np.ndarray:
    z0 = float(np.mean(feature_vec[0:13]))
    z1 = float(np.mean(feature_vec[13:26]))
    z2 = float(np.mean(feature_vec[26:39]))
    total = z0 + z1 + z2 + 1e-12
    angles = np.pi * np.array([z0 / total, z1 / total, z2 / total], dtype=np.float64)
    return angles


def build_wdw_circuit(ctx_angles: np.ndarray, params: np.ndarray) -> QuantumCircuit:
    qc = QuantumCircuit(N_QUBITS)

    for i in range(3):
        qc.ry(float(ctx_angles[i]), i)

    idx = 0
    for _ in range(2):
        for q in range(N_QUBITS):
            qc.ry(float(params[idx]), q)
            idx += 1
            qc.rz(float(params[idx]), q)
            idx += 1
        for q in range(N_QUBITS - 1):
            qc.cx(q, q + 1)
        qc.cx(N_QUBITS - 1, 0)

    return qc


def wdw_conditional_distribution(ctx_angles: np.ndarray, params: np.ndarray) -> np.ndarray:
    qc = build_wdw_circuit(ctx_angles, params)
    sv = Statevector.from_instruction(qc)
    probs_64 = np.abs(sv.data) ** 2

    basis_idx = np.arange(N_STATES, dtype=np.float64)[:, None]
    ball_idx = np.arange(NUM_BALLS, dtype=np.float64)[None, :]
    proj_weights = 0.5 * (
        1.0 + np.cos(2.0 * np.pi * (basis_idx + 0.5) * (ball_idx + 0.5) / N_STATES)
    )
    proj_weights /= np.sum(proj_weights, axis=1, keepdims=True)

    dist_39 = probs_64 @ proj_weights
    dist_39 = np.maximum(dist_39, 1e-12)
    dist_39 /= np.sum(dist_39)
    return dist_39


def train_wdw_model(
    ctx_angles_seq: np.ndarray,
    target_binary_seq: np.ndarray,
    reg_lambda: float = 1e-3,
    max_iter: int = 45,
) -> np.ndarray:
    n_samples = len(ctx_angles_seq)
    target_dist = target_binary_seq / float(DRAW_SIZE)

    step = max(1, n_samples // 180)
    sub_ctx = ctx_angles_seq[::step]
    sub_tgt = target_dist[::step]
    sub_len = len(sub_ctx)

    n_params = 2 * N_QUBITS * 2
    rng = np.random.RandomState(39)
    init_params = rng.uniform(-0.35, 0.35, size=n_params)

    def loss_fn(params: np.ndarray) -> float:
        total_loss = 0.0
        for i in range(sub_len):
            pred_dist = wdw_conditional_distribution(sub_ctx[i], params)
            total_loss -= float(np.sum(sub_tgt[i] * np.log(pred_dist)))
        total_loss /= float(sub_len)
        total_loss += reg_lambda * float(np.sum(params ** 2))
        return float(total_loss)

    res = minimize(
        loss_fn,
        init_params,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": 1e-5},
    )
    return res.x


def evaluate_distribution(
    ctx_angles_seq: np.ndarray, target_binary_seq: np.ndarray, params: np.ndarray
) -> tuple[float, float]:
    n_samples = len(ctx_angles_seq)
    target_dist = target_binary_seq / float(DRAW_SIZE)
    nll = 0.0
    hits = 0.0
    for i in range(n_samples):
        pred_dist = wdw_conditional_distribution(ctx_angles_seq[i], params)
        nll -= float(np.sum(target_dist[i] * np.log(pred_dist)))
        top7 = np.argsort(pred_dist)[-DRAW_SIZE:]
        actual = np.flatnonzero(target_binary_seq[i] > 0.5)
        hits += float(len(set(top7.tolist()).intersection(actual.tolist())))
    return nll / float(n_samples), hits / float(n_samples)


def select_next_combination(dist_39: np.ndarray) -> list[int]:
    if dist_39.shape != (NUM_BALLS,) or not np.all(np.isfinite(dist_39)):
        raise ValueError("Neispravna izlazna raspodela modela.")
    if abs(float(np.sum(dist_39)) - 1.0) > 1e-5:
        raise ValueError("Raspodela nije normalizovana.")
    top7_idx = np.argsort(dist_39)[-DRAW_SIZE:]
    combo = sorted(int(i + 1) for i in top7_idx)
    if len(combo) != DRAW_SIZE or len(set(combo)) != DRAW_SIZE:
        raise ValueError("Nelegalna izlazna kombinacija.")
    return combo


def run_pipeline_for_csv(csv_path: Path) -> None:
    draws = load_draws(csv_path)
    binary_mat = draws_to_binary_matrix(draws)
    t_len = len(binary_mat)

    n_trans = t_len - 1
    train_end = int(n_trans * 0.70)
    val_end = int(n_trans * 0.85)

    candidate_q_sets = (
        (2.0, 4.0, 8.0),
        (3.0, 6.0, 12.0),
    )

    best_q = next(iter(candidate_q_sets))
    best_val_nll = float("inf")
    best_val_hits = 0.0
    best_test_nll = float("inf")
    best_test_hits = 0.0

    for q_set in candidate_q_sets:
        whitened = compute_causal_q_whitened_features(binary_mat, q_values=q_set)
        ctx_all = np.array([encode_context_to_3qubits(whitened[t]) for t in range(t_len)])

        x_train = ctx_all[:train_end]
        y_train = binary_mat[1 : train_end + 1]

        x_val = ctx_all[train_end:val_end]
        y_val = binary_mat[train_end + 1 : val_end + 1]

        params_cv = train_wdw_model(x_train, y_train, max_iter=35)
        val_nll, val_hits = evaluate_distribution(x_val, y_val, params_cv)

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_val_hits = val_hits
            best_q = q_set

            x_test = ctx_all[val_end:n_trans]
            y_test = binary_mat[val_end + 1 : n_trans + 1]
            best_test_nll, best_test_hits = evaluate_distribution(x_test, y_test, params_cv)

    whitened_final = compute_causal_q_whitened_features(binary_mat, q_values=best_q)
    ctx_final = np.array([encode_context_to_3qubits(whitened_final[t]) for t in range(t_len)])

    x_full = ctx_final[:n_trans]
    y_full = binary_mat[1 : n_trans + 1]
    final_params = train_wdw_model(x_full, y_full, max_iter=50)

    next_dist = wdw_conditional_distribution(ctx_final[-1], final_params)
    next_combo = select_next_combination(next_dist)

    print(f"CSV: {csv_path.name} (kola: {t_len})")
    print(f"Validacija -> NLL: {best_val_nll:.4f} | Prosek pogodaka: {best_val_hits:.3f}")
    print(f"Test       -> NLL: {best_test_nll:.4f} | Prosek pogodaka: {best_test_hits:.3f}")
    print(f"NEXT       -> {next_combo}")
    print("-" * 52)


def main() -> None:
    print("Metoda: Q-Transform PSD Whitening + Wheeler-DeWitt Quantum State (6 qubita)")
    print("Posebnost: Kauzalna vreme-frekvencija analiza sa izbeljivanjem i kvantna projekcija amplituda")
    print("=" * 52)
    for csv_path in FILES:
        run_pipeline_for_csv(csv_path)


if __name__ == "__main__":
    main()



"""
Metoda: Q-Transform PSD Whitening + Wheeler-DeWitt Quantum State (6 qubita)
Posebnost: Kauzalna vreme-frekvencija analiza sa izbeljivanjem i kvantna projekcija amplituda
====================================================

CSV: loto7_4718_k90.csv (kola: 4718)
Validacija -> NLL: 3.6667 | Prosek pogodaka: 1.258
Test       -> NLL: 3.6650 | Prosek pogodaka: 1.308
NEXT       -> [11, x, 14, y, 21, z, 27]
----------------------------------------------------

CSV: loto7_4718_k90_loto_2981.csv (kola: 2981)
Validacija -> NLL: 3.6676 | Prosek pogodaka: 1.268
Test       -> NLL: 3.6687 | Prosek pogodaka: 1.345
NEXT       -> [6, x, 15, y, 21, z, 36]
----------------------------------------------------

CSV: loto7_4718_k90_loto_plus_1737.csv (kola: 1737)
Validacija -> NLL: 3.6676 | Prosek pogodaka: 1.185
Test       -> NLL: 3.6751 | Prosek pogodaka: 1.218
NEXT       -> [1, x, 4, y, 9, z, 15]
----------------------------------------------------
"""



"""
Generating Gravitational Waveforms 
detect them in real data 
a Python library widely used in gravitational wave astronomy 
Q-Transform: A Time-Frequency Analysis for Gravitational Waves 
Cleaning Noise in Gravitational Wave Data: Understanding and Mitigating Interference 
When physicists tried to write a quantum theory of gravity, they arrived at the Wheeler-DeWitt equation, whose solution is a wavefunction of the cosmos that never changes.


Pet koncepata iz astronomije gravitacionih talasa (npr. PyCBC / GWpy metodologija) i jedan iz kvantne gravitacije:
1. Generating Gravitational Waveforms (Generisanje talasnih oblika): Modelovanje kontinuiranog signala čija se frekvencija i amplituda menjaju kroz vreme (često kao chirp signal).
2. Detect them in real data (Detekcija u stvarnim podacima): Usklađeno filtriranje (Matched Filtering), odnosno računanje korelacije između banke šablona i bučnog niza podataka u frekventnom domenu.
3. Q-Transform (Time-Frequency Analysis): Višerezolucijska analiza vremenske serije sa konstantnim faktorom kvaliteta \(Q\), koja mapira evoluciju signala u vreme-frekvencija ravan.
4. Cleaning Noise in Gravitational Wave Data: Procena spektralne gustine snage (PSD) pozadine i izbeljivanje (whitening) podataka da bi se izdvojile strukturne anomalije iznad šuma.
5. Wheeler-DeWitt jednačina: Kvantno-gravitaciona jednačina \(\hat{H}\Psi = 0\), gde je talasna funkcija \(\Psi \) stacionarna (bez eksplicitnog spoljašnjeg vremena), a dinamika se opisuje kroz korelacije među unutrašnjim stepenima slobode (uslovna verovatnoća po Page-Wootters mehanizmu).


• Q-Transform + izbeljivanje (PSD whitening) kao ulazna reprezentacija: Istorija od ~5.000 kola Loto 7/39 može se posmatrati kao 39 paralelnih vremenskih serija (ili spektralni niz stanja). Q-transformacija sa različitim \(Q\) faktorima hvata istovremeno i brze lokalne promene (kratki prozori) i spore dugoročne cikluse (dugi prozori) bez pretpostavke normalne raspodele. Izbeljivanje u odnosu na istorijski PSD uklanja stacionarni pozadinski šum i ističe strukturu prelaza.
• Wheeler-DeWitt / Page-Wootters princip u kvantnom kolu (Qiskit): 6 qubita - sedam registara / 64 bazna stanja za mapiranje 39 brojeva i amplitude. Umesto klasičnog vremenskog koraka gde se predviđa samo sledeći tren, konstruišem stacionarno kvantno stanje (talasnu funkciju \(\Psi \) sistema) koje kodira zajedničke korelacije između stanja prethodnih kola i stanja narednog kola. Merenjem uslovnih amplituda dobijam direktnu regresiju raspodele nad brojevima 1-39, bez klasifikatora i bez svođenja na proste istorijske frekvencije.


Kratak pregled arhitekture (v1): 
• Naziv i posebnost metode: Q-Transform PSD Whitening + Wheeler-DeWitt (WDW) Quantum Conditional State (6 qubita). Svaki od 39 brojeva posmatra se kao signal kroz istoriju izvlačenja; lokalna frekvencijska evolucija izvlači se konstantnim \(Q\)-faktorom (Q-Transform) i izbeljuje u odnosu na kauzalni PSD (spektralnu gustinu snage), čime se uklanja stacionarni pozadinski šum.
• Kvantni deo (Qiskit Statevector simulator): 6 qubita (\(2^6 = 64\) bazna stanja). Prva 3 qubita predstavljaju uslovni registar trenutnog spektralnog konteksta, a preostala 3 qubita predstavljaju registar narednog stanja. U duhu Wheeler-DeWitt jednačine (\(\hat{H}\Psi = 0\)), parametrizovano kvantno kolo uči stacionarnu zajedničku talasnu funkciju \(\Psi \) nad prelazima, dok se predikcija raspodele za naredno kolo dobija projekcijom (uslovljavanjem) amplitude na opaženi kontekst.
• Klasični deo: Hronološko računanje Q-transformacije i kauzalnog PSD-a (bez curenja budućih podataka), hronološki odabir hiperparametara (\(Q\), opseg učenja), L-BFGS-B optimizacija parametara kvantnog kola na celoj dostupnoj istoriji prelaza i deterministički izbor 7 brojeva sa najvišom uslovnom amplitudom/verovatnoćom.
"""
