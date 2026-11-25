
from pathlib import Path
import math
import os
import csv
from multiprocessing import Pool, cpu_count

# ==========================================
# CPU THREAD CONTROL
# Set BEFORE importing PySCF
# ==========================================

THREADS_PER_WORKER = 1
MAX_WORKERS = 24

os.environ["OMP_NUM_THREADS"] = str(THREADS_PER_WORKER)
os.environ["MKL_NUM_THREADS"] = str(THREADS_PER_WORKER)
os.environ["OPENBLAS_NUM_THREADS"] = str(THREADS_PER_WORKER)

from pyscf import gto, dft

# ==========================================
# DIRECTORIES
# ==========================================

BASE_DIR = Path("/mnt/e/Multimode/MM-Variation/GC-MM")

RESULTS_DIR = Path("/mnt/e/Multimode/Results/GC/Interaction Energies")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# CONSTANTS
# ==========================================

COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "P": 1.07,
    "S": 1.05
}

HARTREE_TO_KJMOL = 2625.4996

FIELDNAMES = [
    "Filename",
    "Status",
    "E_Complex_kJ",
    "Raw_Interaction_kJ",
    "BSSE_Penalty_kJ",
    "CP_Corrected_kJ"
]

# ==========================================
# IDENTIFY MOLECULAR FRAGMENTS
# ==========================================

def get_covalent_radius(symbol):
    clean_sym = symbol.split("-")[-1].capitalize()
    return COVALENT_RADII.get(clean_sym, 0.75)


def find_molecules(atoms_data):
    """Split atoms into fragments using covalent connectivity."""

    n = len(atoms_data)
    adj = {i: [] for i in range(n)}

    for i in range(n):
        sym_i, xi, yi, zi = atoms_data[i]
        r_i = get_covalent_radius(sym_i)

        for j in range(i + 1, n):
            sym_j, xj, yj, zj = atoms_data[j]
            r_j = get_covalent_radius(sym_j)

            dist = math.sqrt(
                (xi - xj)**2
                + (yi - yj)**2
                + (zi - zj)**2
            )

            if dist < 1.3 * (r_i + r_j):
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * n
    components = []

    for i in range(n):
        if not visited[i]:
            component = []
            queue = [i]
            visited[i] = True

            while queue:
                node = queue.pop(0)
                component.append(atoms_data[node])

                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)

            components.append(component)

    return components


# ==========================================
# READ XYZ FILE
# ==========================================

def read_xyz(xyz_file):
    """Read atomic symbols and coordinates from an XYZ file."""

    lines = xyz_file.read_text(
        encoding="utf-8"
    ).strip().splitlines()

    atoms_data = []

    for line in lines[2:]:
        parts = line.split()

        if len(parts) >= 4:
            atoms_data.append((
                parts[0],
                float(parts[1]),
                float(parts[2]),
                float(parts[3])
            ))

    if not atoms_data:
        raise ValueError("No atomic coordinates found.")

    return atoms_data


# ==========================================
# PySCF DFT CALCULATION
# ==========================================

def run_pyscf_calculation(atoms_data):
    """Run B3LYP-D3(BJ)/def2-SVP single-point calculation."""

    atom_list = [
        f"{sym} {x:.6f} {y:.6f} {z:.6f}"
        for sym, x, y, z in atoms_data
    ]

    mol = gto.M(
        atom="; ".join(atom_list),
        basis="def2-svp",
        charge=0,
        spin=0,
        unit="Angstrom",
        verbose=0
    )

    mf = dft.RKS(mol, xc="b3lyp-d3bj")
    mf.conv_tol = 1e-8
    mf.grids.level = 2

    energy_hartree = mf.kernel()

    if not mf.converged:
        raise RuntimeError("SCF calculation did not converge.")

    return energy_hartree


# ==========================================
# REFERENCE ENERGIES SHARED BY WORKERS
# ==========================================

GLOBAL_E_MONO1_ISO_KJ = None
GLOBAL_E_MONO2_ISO_KJ = None


def init_worker(e_m1_iso, e_m2_iso):
    """Initialize the isolated monomer energies in each worker."""

    global GLOBAL_E_MONO1_ISO_KJ
    global GLOBAL_E_MONO2_ISO_KJ

    GLOBAL_E_MONO1_ISO_KJ = e_m1_iso
    GLOBAL_E_MONO2_ISO_KJ = e_m2_iso


# ==========================================
# PROCESS ONE XYZ FILE
# ==========================================

def process_single_file(xyz_file):
    """Calculate raw and counterpoise-corrected interaction energies."""

    try:
        atoms_data = read_xyz(xyz_file)
        fragments = find_molecules(atoms_data)

        if len(fragments) != 2:
            raise ValueError(
                f"Expected 2 fragments, found {len(fragments)}"
            )

        frag1, frag2 = fragments

        # Ghost atoms for the counterpoise correction
        frag1_ghosts = [
            (f"X-{sym}", x, y, z)
            for sym, x, y, z in frag1
        ]

        frag2_ghosts = [
            (f"X-{sym}", x, y, z)
            for sym, x, y, z in frag2
        ]

        # Three calculations per geometry
        e_complex = run_pyscf_calculation(atoms_data)

        e_mono1_cp = run_pyscf_calculation(
            frag1 + frag2_ghosts
        )

        e_mono2_cp = run_pyscf_calculation(
            frag2 + frag1_ghosts
        )

        # Convert Hartree to kJ/mol
        e_comp_kj = e_complex * HARTREE_TO_KJMOL
        e_m1_cp_kj = e_mono1_cp * HARTREE_TO_KJMOL
        e_m2_cp_kj = e_mono2_cp * HARTREE_TO_KJMOL

        e_m1_iso_kj = GLOBAL_E_MONO1_ISO_KJ
        e_m2_iso_kj = GLOBAL_E_MONO2_ISO_KJ

        # Raw interaction energy
        delta_e_raw = (
            e_comp_kj - (e_m1_iso_kj + e_m2_iso_kj)
        )

        # BSSE penalty
        bsse_energy = (
            (e_m1_cp_kj - e_m1_iso_kj)
            + (e_m2_cp_kj - e_m2_iso_kj)
        )

        # Counterpoise-corrected interaction energy
        delta_e_corrected = (
            e_comp_kj - (e_m1_cp_kj + e_m2_cp_kj)
        )

        return {
            "Filename": xyz_file.name,
            "Status": "Success",
            "E_Complex_kJ": round(e_comp_kj, 4),
            "Raw_Interaction_kJ": round(delta_e_raw, 4),
            "BSSE_Penalty_kJ": round(bsse_energy, 4),
            "CP_Corrected_kJ": round(delta_e_corrected, 4)
        }

    except Exception as e:
        return {
            "Filename": xyz_file.name,
            "Status": f"Error: {str(e)}",
            "E_Complex_kJ": None,
            "Raw_Interaction_kJ": None,
            "BSSE_Penalty_kJ": None,
            "CP_Corrected_kJ": None
        }


# ==========================================
# PROCESS ONE OUTPUT FOLDER
# ==========================================

def process_output_folder(folder, folder_index, total_folders):

    output_csv = RESULTS_DIR / (
        f"{folder.name}_interaction_energies.csv"
    )

    xyz_files = sorted(
        f for f in folder.glob("*.xyz")
        if "_monomer" not in f.name
    )

    print("\n" + "=" * 75)
    print(
        f"OUTPUT FOLDER {folder_index}/{total_folders}: "
        f"{folder.name}"
    )
    print("=" * 75)

    if not xyz_files:
        print(f"No XYZ files found in {folder}. Skipping.")
        return

    print(f"XYZ files found: {len(xyz_files)}")
    print(f"CSV output: {output_csv}")

    # --------------------------------------
    # Resume: identify previously successful files
    # --------------------------------------

    processed_files = set()

    if output_csv.exists() and output_csv.stat().st_size > 0:
        with open(
            output_csv,
            "r",
            newline="",
            encoding="utf-8"
        ) as csvfile:

            reader = csv.DictReader(csvfile)

            if reader.fieldnames and "Filename" in reader.fieldnames:
                for row in reader:
                    if row.get("Status") == "Success":
                        processed_files.add(row["Filename"])

        print(
            f"Previously successful: {len(processed_files)}"
        )

    remaining_files = [
        f for f in xyz_files
        if f.name not in processed_files
    ]

    total_remaining = len(remaining_files)

    print(f"Files remaining: {total_remaining}")

    if total_remaining == 0:
        print("All files in this folder are already calculated.")
        return

    # --------------------------------------
    # Find a valid reference geometry
    # --------------------------------------

    reference_fragments = None

    for xyz_file in xyz_files:
        try:
            candidate_atoms = read_xyz(xyz_file)
            candidate_fragments = find_molecules(candidate_atoms)

            if len(candidate_fragments) == 2:
                reference_fragments = candidate_fragments
                break

        except Exception:
            continue

    if reference_fragments is None:
        print(
            f"ERROR: No valid two-fragment geometry "
            f"found in {folder.name}."
        )
        return

    ref_frag1, ref_frag2 = reference_fragments

    print("Calculating isolated reference monomer energies...")

    try:
        e_mono1_iso_ref = (
            run_pyscf_calculation(ref_frag1)
            * HARTREE_TO_KJMOL
        )

        e_mono2_iso_ref = (
            run_pyscf_calculation(ref_frag2)
            * HARTREE_TO_KJMOL
        )

    except Exception as e:
        print(f"Reference calculation failed: {e}")
        return

    print(
        f"Reference monomer 1: {e_mono1_iso_ref:.4f} kJ/mol"
    )
    print(
        f"Reference monomer 2: {e_mono2_iso_ref:.4f} kJ/mol"
    )

    # --------------------------------------
    # Determine CPU worker count
    # --------------------------------------

    total_cores = cpu_count()
    num_workers = min(MAX_WORKERS, total_cores)

    print(f"Available CPU threads: {total_cores}")
    print(f"Parallel workers: {num_workers}")
    print(f"Threads per worker: {THREADS_PER_WORKER}")
    print(
        f"Allocated worker threads: "
        f"{num_workers * THREADS_PER_WORKER}"
    )
    print("-" * 75)

    file_exists = (
        output_csv.exists()
        and output_csv.stat().st_size > 0
    )

    with open(
        output_csv,
        "a",
        newline="",
        encoding="utf-8"
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=FIELDNAMES
        )

        if not file_exists:
            writer.writeheader()
            csvfile.flush()

        # ----------------------------------
        # Parallel calculations and per-file progress
        # ----------------------------------

        with Pool(
            processes=num_workers,
            initializer=init_worker,
            initargs=(
                e_mono1_iso_ref,
                e_mono2_iso_ref
            )
        ) as pool:

            for count, result in enumerate(
                pool.imap_unordered(
                    process_single_file,
                    remaining_files,
                    chunksize=1
                ),
                start=1
            ):

                writer.writerow(result)
                csvfile.flush()

                percentage = count / total_remaining * 100

                status_label = (
                    "SUCCESS"
                    if result["Status"] == "Success"
                    else "FAILED"
                )

                print(
                    f"[{folder.name}] "
                    f"File {count}/{total_remaining} "
                    f"({percentage:.2f}%) | "
                    f"{result['Filename']} | "
                    f"{status_label}",
                    flush=True
                )

    print("-" * 75)
    print(f"Finished folder: {folder.name}")
    print(f"Results saved: {output_csv}")
    print(f"Total files: {len(xyz_files)}")
    print(f"Previously successful: {len(processed_files)}")
    print(f"Newly processed: {total_remaining}")


# ==========================================
# MAIN PROGRAM
# ==========================================

if __name__ == "__main__":

    if not BASE_DIR.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {BASE_DIR}"
        )

    # Match output1, output2, ..., outputN and sort numerically
    output_folders = sorted(
        (
            folder for folder in BASE_DIR.iterdir()
            if folder.is_dir()
            and folder.name.startswith("output")
            and folder.name[len("output"):].isdigit()
        ),
        key=lambda folder: int(
            folder.name[len("output"):]
        )
    )

    if not output_folders:
        raise FileNotFoundError(
            f"No output1, output2, ... folders found in {BASE_DIR}"
        )

    total_folders = len(output_folders)

    print("=" * 75)
    print("PySCF BASE-PAIR INTERACTION ENERGY CALCULATIONS")
    print("=" * 75)
    print(f"Input directory: {BASE_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Number of output folders: {total_folders}")
    print("Method: B3LYP-D3(BJ)/def2-SVP")
    print(f"CPU threads available: {cpu_count()}")
    print(f"Maximum parallel workers: {MAX_WORKERS}")
    print(f"Threads per worker: {THREADS_PER_WORKER}")
    print("=" * 75)

    for folder_index, folder in enumerate(
        output_folders,
        start=1
    ):
        process_output_folder(
            folder,
            folder_index,
            total_folders
        )

    print("\n" + "=" * 75)
    print("ALL OUTPUT FOLDERS HAVE BEEN PROCESSED.")
    print(f"Results directory: {RESULTS_DIR.resolve()}")
    print("=" * 75)
