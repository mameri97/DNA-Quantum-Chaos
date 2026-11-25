
# ============================================================
# CODE 2: KOHN-SHAM ORBITAL ENERGIES FOR AT / GC
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
INPUT_ROOT = Path("/mnt/e/dna_frames")

# Use a different folder from Code 1.
RESULT_ROOT = Path("/mnt/e/dna_frames/Result of orbital energies")

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

HARTREE_TO_EV = 27.211386245988


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
# 3. CALCULATE ORBITAL ENERGIES FOR ONE GEOMETRY
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

    # Convert all orbital energies from Hartree to eV.
    orbital_energies = mf.mo_energy * HARTREE_TO_EV
    occupations = mf.mo_occ

    occupied_indices = [
        i for i, occ in enumerate(occupations) if occ > 1e-8
    ]
    virtual_indices = [
        i for i, occ in enumerate(occupations) if occ <= 1e-8
    ]

    if not occupied_indices or not virtual_indices:
        raise RuntimeError(
            "Could not identify both occupied and virtual orbitals."
        )

    homo_index = occupied_indices[-1]
    lumo_index = virtual_indices[0]

    homo_ev = float(orbital_energies[homo_index])
    lumo_ev = float(orbital_energies[lumo_index])
    gap_ev = lumo_ev - homo_ev

    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the complete orbital spectrum for this geometry.
    with open(result_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "orbital_index",
            "occupation",
            "orbital_type",
            "orbital_energy_hartree",
            "orbital_energy_eV"
        ])

        for i, (energy, occ) in enumerate(
            zip(mf.mo_energy, occupations), start=1
        ):
            if occ > 1e-8:
                orbital_type = "occupied"
            else:
                orbital_type = "virtual"

            writer.writerow([
                i,
                float(occ),
                orbital_type,
                f"{float(energy):.12f}",
                f"{float(energy * HARTREE_TO_EV):.12f}"
            ])

    return {
        "geometry": str(xyz_path),
        "n_atoms": mol.natm,
        "basis_functions": mol.nao_nr(),
        "n_occupied_orbitals": len(occupied_indices),
        "n_virtual_orbitals": len(virtual_indices),
        "HOMO_eV": homo_ev,
        "LUMO_eV": lumo_ev,
        "HOMO_LUMO_gap_eV": gap_ev,
        "SCF_energy_hartree": float(mf.e_tot),
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
    summary_file = RESULT_ROOT / "orbital_summary.csv"
    start = time.time()

    print("=" * 65)
    print("KOHN-SHAM ORBITAL CALCULATIONS")
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
                    f"HOMO={row['HOMO_eV']:.4f} eV | "
                    f"LUMO={row['LUMO_eV']:.4f} eV | "
                    f"Gap={row['HOMO_LUMO_gap_eV']:.4f} eV",
                    flush=True
                )

            except Exception as exc:
                row = {
                    "geometry": xyz_path,
                    "n_atoms": "",
                    "basis_functions": "",
                    "n_occupied_orbitals": "",
                    "n_virtual_orbitals": "",
                    "HOMO_eV": "",
                    "LUMO_eV": "",
                    "HOMO_LUMO_gap_eV": "",
                    "SCF_energy_hartree": "",
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

            # Update summary after each completed geometry.
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
    print("ORBITAL CALCULATIONS FINISHED")
    print(f"Summary: {summary_file}")
    print(f"Total time: {(time.time()-start)/60:.1f} min")
    print("=" * 65)


if __name__ == "__main__":
    main()
