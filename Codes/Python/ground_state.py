
# ============================================================
# CODE 1: GROUND-STATE DFT ENERGIES FOR AT / GC
# PySCF | RKS-B3LYP | def2-SVP
# No TDDFT or TDA
# ============================================================

import os

# Set thread limits before importing NumPy or PySCF.
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import csv
import time
import multiprocessing as mp

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from pyscf import gto, dft, lib


# ------------------------------------------------------------
# 1. USER SETTINGS
# ------------------------------------------------------------

# CHANGE this to the folder containing your randomized XYZ files.
INPUT_ROOT = Path("/mnt/e/Multimode/MM-Variation/GC-MM/output1")

# Keep results separate from the input files.
RESULT_ROOT = Path("/mnt/e/Multimode/Results/GC/Ground State")

BASIS = "def2-svp"
XC_FUNCTIONAL = "b3lyp"

CHARGE = 0
SPIN = 0

N_WORKERS = 8
THREADS_PER_WORKER = 4

MAX_MEMORY_MB = 10000
MAX_SCF_CYCLES = 150
SCF_TOLERANCE = 1e-8
GRID_LEVEL = 3


# ------------------------------------------------------------
# 2. READ XYZ FILE
# ------------------------------------------------------------

def read_xyz(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    if len(lines) < 3:
        raise ValueError("XYZ file is too short.")

    n_atoms = int(lines[0].strip())
    atom_lines = lines[2:2 + n_atoms]

    if len(atom_lines) != n_atoms:
        raise ValueError("XYZ atom count does not match its header.")

    atoms = []

    for line in atom_lines:
        fields = line.split()

        if len(fields) < 4:
            raise ValueError(f"Invalid XYZ atom line: {line}")

        symbol = fields[0]
        x, y, z = map(float, fields[1:4])

        atoms.append((symbol, (x, y, z)))

    return atoms


# ------------------------------------------------------------
# 3. CALCULATE ONE GROUND-STATE ENERGY
# ------------------------------------------------------------

def calculate_one(task):
    xyz_path, result_path = task

    lib.num_threads(THREADS_PER_WORKER)

    atoms = read_xyz(xyz_path)

    mol = gto.M(
        atom=atoms,
        basis=BASIS,
        charge=CHARGE,
        spin=SPIN,
        unit="Angstrom",
        max_memory=MAX_MEMORY_MB,
        verbose=0
    )

    mf = dft.RKS(mol)
    mf.xc = XC_FUNCTIONAL
    mf.max_cycle = MAX_SCF_CYCLES
    mf.conv_tol = SCF_TOLERANCE
    mf.grids.level = GRID_LEVEL

    start = time.time()
    mf.kernel()
    elapsed = time.time() - start

    if not mf.converged:
        raise RuntimeError("SCF did not converge.")

    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the individual ground-state result.
    with open(result_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "geometry",
            "n_atoms",
            "basis_functions",
            "ground_state_energy_hartree",
            "ground_state_energy_eV",
            "SCF_converged",
            "SCF_time_seconds"
        ])

        writer.writerow([
            str(xyz_path),
            mol.natm,
            mol.nao_nr(),
            f"{mf.e_tot:.12f}",
            f"{mf.e_tot * 27.211386245988:.12f}",
            True,
            f"{elapsed:.2f}"
        ])

    return {
        "geometry": str(xyz_path),
        "n_atoms": mol.natm,
        "basis_functions": mol.nao_nr(),
        "ground_state_energy_hartree": mf.e_tot,
        "ground_state_energy_eV": (
            mf.e_tot * 27.211386245988
        ),
        "SCF_converged": True,
        "SCF_time_seconds": round(elapsed, 2),
        "status": "SUCCESS",
        "result_csv": str(result_path),
        "error": ""
    }


# ------------------------------------------------------------
# 4. MAIN PROGRAM
# ------------------------------------------------------------

def main():

    if not INPUT_ROOT.exists():
        raise FileNotFoundError(
            f"Input directory not found: {INPUT_ROOT}"
        )

    xyz_files = sorted(
        p for p in INPUT_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() == ".xyz"
    )

    if not xyz_files:
        raise FileNotFoundError(
            f"No XYZ files found under {INPUT_ROOT}"
        )

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    tasks = []

    for xyz in xyz_files:
        relative = xyz.relative_to(INPUT_ROOT)
        output = RESULT_ROOT / relative.with_suffix(".csv")
        tasks.append((str(xyz), str(output)))

    total = len(tasks)
    rows = []
    summary_file = RESULT_ROOT / "ground_state_summary.csv"
    start = time.time()

    print("=" * 65)
    print("GROUND-STATE DFT CALCULATIONS")
    print(f"Geometries: {total}")
    print(f"Workers: {N_WORKERS}")
    print(f"Threads per worker: {THREADS_PER_WORKER}")
    print(f"Functional: {XC_FUNCTIONAL}")
    print(f"Basis: {BASIS}")
    print(f"Output: {RESULT_ROOT}")
    print("=" * 65, flush=True)

    with ProcessPoolExecutor(
        max_workers=N_WORKERS,
        mp_context=mp.get_context("spawn")
    ) as executor:

        futures = {
            executor.submit(calculate_one, task): task
            for task in tasks
        }

        for count, future in enumerate(
            as_completed(futures), start=1
        ):
            xyz_path, _ = futures[future]

            try:
                row = future.result()
                print(
                    f"[{count}/{total}] SUCCESS | "
                    f"{Path(xyz_path).name} | "
                    f"E = {row['ground_state_energy_hartree']:.8f} Ha | "
                    f"{row['SCF_time_seconds']} s",
                    flush=True
                )

            except Exception as exc:
                row = {
                    "geometry": xyz_path,
                    "n_atoms": "",
                    "basis_functions": "",
                    "ground_state_energy_hartree": "",
                    "ground_state_energy_eV": "",
                    "SCF_converged": False,
                    "SCF_time_seconds": "",
                    "status": "FAILED",
                    "result_csv": "",
                    "error": f"{type(exc).__name__}: {exc}"
                }

                print(
                    f"[{count}/{total}] FAILED | "
                    f"{Path(xyz_path).name} | {exc}",
                    flush=True
                )

            rows.append(row)

            # Update the summary after each completed geometry.
            with open(summary_file, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=list(rows[0].keys())
                )
                writer.writeheader()
                writer.writerows(rows)

            elapsed = time.time() - start
            remaining = elapsed / count * (total - count)

            print(
                f"Progress: {100*count/total:.1f}% | "
                f"Estimated remaining: {remaining/60:.1f} min",
                flush=True
            )

    print("=" * 65)
    print("GROUND-STATE CALCULATIONS FINISHED")
    print(f"Summary: {summary_file}")
    print(f"Total time: {(time.time()-start)/60:.1f} min")
    print("=" * 65)


if __name__ == "__main__":
    main()
