import math
import random
from pathlib import Path

# ============================================================
# USER SETTINGS
# ============================================================

N = 100  # Number of repetitions / output folders
input_filename = "WC-GC.xyz"

# EXACT number of XYZ files in EACH output folder
xyz_files_per_repetition = 1000

# Random displacement range, in angstroms
offset_min = -1.000
offset_max = 1.000

# Hydrogen-bond geometric criteria
max_h_acceptor_distance = 3.0
min_dha_angle = 120.0
strong_max_distance = 2.5
strong_min_angle = 150.0
min_h_acceptor_limit = 1.5

# ============================================================
# APPROXIMATE COVALENT RADII (angstroms)
# ============================================================

radii = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
}

# ============================================================
# READ THE INPUT XYZ FILE
# ============================================================

with open(input_filename, "r", encoding="utf-8") as file:
    lines = file.readlines()

if not lines:
    raise ValueError(f"The input file {input_filename!r} is empty.")

atom_count = int(lines[0].strip())
atoms = []

for line in lines[2:2 + atom_count]:
    parts = line.split()

    if len(parts) < 4:
        continue

    element = parts[0].capitalize()
    x, y, z = map(float, parts[1:4])

    if element not in radii:
        raise ValueError(
            f"No covalent radius is defined for element {element!r}."
        )

    atoms.append((element, (x, y, z)))

if len(atoms) != atom_count:
    raise ValueError(
        f"The XYZ file declares {atom_count} atoms, but "
        f"{len(atoms)} valid atom coordinate lines were read."
    )

# ============================================================
# DETECT COVALENT BONDS AND NEIGHBORS
# ============================================================

bonds = []
neighbors = [[] for _ in atoms]

for i in range(len(atoms)):
    element1, xyz1 = atoms[i]

    for j in range(i + 1, len(atoms)):
        element2, xyz2 = atoms[j]

        dx = xyz1[0] - xyz2[0]
        dy = xyz1[1] - xyz2[1]
        dz = xyz1[2] - xyz2[2]

        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        bond_limit = 1.2 * (radii[element1] + radii[element2])

        if distance <= bond_limit:
            bonds.append((i, j, distance))
            neighbors[i].append(j)
            neighbors[j].append(i)

print("Atoms and neighbors:")

for i, (element, xyz) in enumerate(atoms):
    neighbor_list = [
        f"{j + 1} ({atoms[j][0]})"
        for j in neighbors[i]
    ]

    print(
        f"Atom {i + 1}: {element}, coordinates = {xyz}, "
        f"neighbors = "
        f"{', '.join(neighbor_list) if neighbor_list else 'none'}"
    )

print("\nEstimated bonds:")

for i, j, distance in bonds:
    print(
        f"Atom {i + 1} ({atoms[i][0]}) — "
        f"Atom {j + 1} ({atoms[j][0]}): {distance:.3f} Å"
    )

# ============================================================
# DETECT POSSIBLE HYDROGEN BONDS
# ============================================================

hydrogen_bonds = []

possible_donors = {"N", "O", "F"}
possible_acceptors = {"N", "O", "F"}

for h, (h_element, h_xyz) in enumerate(atoms):

    if h_element != "H":
        continue

    for donor in neighbors[h]:

        donor_element, donor_xyz = atoms[donor]

        if donor_element not in possible_donors:
            continue

        for acceptor, (
            acceptor_element, acceptor_xyz
        ) in enumerate(atoms):

            if acceptor_element not in possible_acceptors:
                continue

            if acceptor == donor or acceptor in neighbors[donor]:
                continue

            dx = h_xyz[0] - acceptor_xyz[0]
            dy = h_xyz[1] - acceptor_xyz[1]
            dz = h_xyz[2] - acceptor_xyz[2]

            distance = math.sqrt(dx**2 + dy**2 + dz**2)

            # D-H...A angle, measured at H
            v1 = [
                donor_xyz[k] - h_xyz[k]
                for k in range(3)
            ]

            v2 = [
                acceptor_xyz[k] - h_xyz[k]
                for k in range(3)
            ]

            dot = sum(v1[k] * v2[k] for k in range(3))

            len1 = math.sqrt(
                sum(value**2 for value in v1)
            )

            len2 = math.sqrt(
                sum(value**2 for value in v2)
            )

            if len1 == 0.0 or len2 == 0.0:
                continue

            cosine = max(
                -1.0,
                min(1.0, dot / (len1 * len2))
            )

            angle = math.degrees(math.acos(cosine))

            if (
                distance <= max_h_acceptor_distance
                and angle >= min_dha_angle
            ):
                hydrogen_bonds.append(
                    (donor, h, acceptor, distance, angle)
                )

# ============================================================
# CLASSIFY HYDROGEN-BOND CANDIDATES
# ============================================================

strong = [
    hb for hb in hydrogen_bonds
    if hb[3] <= strong_max_distance
    and hb[4] >= strong_min_angle
]

other = [
    hb for hb in hydrogen_bonds
    if hb not in strong
]

strong.sort(key=lambda hb: (hb[3], -hb[4]))


def print_bonds(title, bond_list):
    print(f"\n{title}:")

    if not bond_list:
        print("  None found")
        return

    for donor, h, acceptor, distance, angle in bond_list:
        print(
            f"  Donor {donor + 1} ({atoms[donor][0]})–"
            f"H {h + 1}···acceptor {acceptor + 1} "
            f"({atoms[acceptor][0]}): "
            f"{distance:.3f} Å, {angle:.1f}°"
        )


print_bonds("Stronger geometric candidates", strong)
print_bonds("Other possible candidates", other)

# ============================================================
# FIND CONNECTED MOLECULAR COMPONENTS
# ============================================================

def find_components(atom_neighbors):
    """Group atoms connected by inferred covalent bonds."""

    component_id = [-1] * len(atom_neighbors)
    components = []

    for start in range(len(atom_neighbors)):

        if component_id[start] != -1:
            continue

        group_number = len(components)
        component = []
        stack = [start]
        component_id[start] = group_number

        while stack:
            atom = stack.pop()
            component.append(atom)

            for neighbor in atom_neighbors[atom]:
                if component_id[neighbor] == -1:
                    component_id[neighbor] = group_number
                    stack.append(neighbor)

        components.append(component)

    return component_id, components


component_id, components = find_components(neighbors)

original_coords = [
    list(atom[1])
    for atom in atoms
]

# ============================================================
# PREPARE ELIGIBLE HYDROGEN-BOND MOVES
# ============================================================

eligible_bonds = []

for bond_number, (
    donor, h, acceptor, old_distance, angle
) in enumerate(strong, start=1):

    if component_id[donor] == component_id[acceptor]:
        print(
            f"Skipping bond {bond_number}: donor and acceptor are in the "
            "same connected molecule. It needs a flexible-group adjustment."
        )
        continue

    h_xyz = original_coords[h]
    a_xyz = original_coords[acceptor]

    vector = [
        a_xyz[k] - h_xyz[k]
        for k in range(3)
    ]

    length = math.sqrt(
        sum(value**2 for value in vector)
    )

    if length == 0.0:
        print(
            f"Skipping bond {bond_number}: zero H-acceptor distance."
        )
        continue

    # Only allow offsets that satisfy the minimum H...acceptor distance
    valid_offset_min = max(
        offset_min,
        min_h_acceptor_limit - old_distance
    )

    if valid_offset_min > offset_max:
        print(
            f"Skipping bond {bond_number}: no offset in the chosen "
            "range can satisfy the minimum H-acceptor distance."
        )
        continue

    eligible_bonds.append({
        "bond_number": bond_number,
        "donor": donor,
        "hydrogen": h,
        "acceptor": acceptor,
        "old_distance": old_distance,
        "angle": angle,
        "direction": [
            value / length for value in vector
        ],
        "acceptor_molecule": components[
            component_id[acceptor]
        ],
        "valid_offset_min": valid_offset_min,
    })

if not eligible_bonds:
    raise RuntimeError(
        "No eligible strong hydrogen-bond candidates were found. "
        "The script cannot create valid displaced structures. "
        "Check the input geometry and bond-detection criteria."
    )

# ============================================================
# GENERATE EXACTLY 100 XYZ FILES PER OUTPUT FOLDER
# ============================================================

for run_number in range(1, N + 1):

    print("\n" + "=" * 60)
    print(f"STARTING REPETITION {run_number} OF {N}")
    print("=" * 60)

    output_folder = Path(f"output{run_number}")
    output_folder.mkdir(parents=True, exist_ok=True)

    # Remove old XYZ files so previous runs cannot leave extra files
    for old_xyz in output_folder.glob("*.xyz"):
        old_xyz.unlink()

    min_hbond_length = float("inf")
    max_hbond_length = float("-inf")
    generated_count = 0

    # Generate exactly the requested number of frames
    for frame_number in range(
        1, xyz_files_per_repetition + 1
    ):

        # Randomly select one eligible hydrogen bond
        hb = random.choice(eligible_bonds)

        # Generate an offset that obeys the minimum distance
        offset = random.uniform(
            hb["valid_offset_min"],
            offset_max
        )

        target_distance = hb["old_distance"] + offset

        min_hbond_length = min(
            min_hbond_length,
            target_distance
        )

        max_hbond_length = max(
            max_hbond_length,
            target_distance
        )

        # Begin with the original molecular coordinates
        new_coords = [
            xyz[:]
            for xyz in original_coords
        ]

        # Translate the whole acceptor molecule
        shift = [
            offset * value
            for value in hb["direction"]
        ]

        for atom_index in hb["acceptor_molecule"]:
            for k in range(3):
                new_coords[atom_index][k] += shift[k]

        label = (
            f"Run {run_number}; frame {frame_number}; "
            f"HB {hb['bond_number']}; atoms "
            f"{hb['donor'] + 1}-H{hb['hydrogen'] + 1}..."
            f"{hb['acceptor'] + 1}; "
            f"target H-A={target_distance:.3f} A; "
            f"original angle={hb['angle']:.1f} deg; "
            f"offset={offset:+.4f} A"
        )

        # Unique filenames ensure no random-offset collisions
        xyz_filename = f"frame_{frame_number:03d}.xyz"
        filepath = output_folder / xyz_filename

        with open(filepath, "w", encoding="utf-8") as output:
            output.write(f"{len(atoms)}\n{label}\n")

            for (element, _), (x, y, z) in zip(
                atoms, new_coords
            ):
                output.write(
                    f"{element:2s} "
                    f"{x:.8f} {y:.8f} {z:.8f}\n"
                )

        generated_count += 1

    # ========================================================
    # SAVE SUMMARY STATISTICS
    # ========================================================

    summary_filepath = (
        output_folder / "hbond_summary.txt"
    )

    with open(
        summary_filepath, "w", encoding="utf-8"
    ) as summary_file:

        summary_file.write(
            "Hydrogen Bond Length Summary Statistics (GC Base Pair)\n"
        )

        summary_file.write(
            f"Repetition: {run_number} of {N}\n"
        )

        summary_file.write("=" * 50 + "\n")

        summary_file.write(
            f"Minimum target H-bond length: "
            f"{min_hbond_length:.3f} Å\n"
        )

        summary_file.write(
            f"Maximum target H-bond length: "
            f"{max_hbond_length:.3f} Å\n"
        )

        summary_file.write(
            f"XYZ files generated: {generated_count}\n"
        )

    # Verify the exact file count
    actual_xyz_count = len(
        list(output_folder.glob("*.xyz"))
    )

    if actual_xyz_count != xyz_files_per_repetition:
        raise RuntimeError(
            f"Expected {xyz_files_per_repetition} XYZ files "
            f"in {output_folder}, but found {actual_xyz_count}."
        )

    print(f"Repetition {run_number} completed.")
    print(f"XYZ files generated: {actual_xyz_count}")
    print(f"Output folder: {output_folder.resolve()}")
    print(f"Summary saved: {summary_filepath}")

# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 60)
print(
    f"ALL {N} REPETITIONS COMPLETED: "
    f"{xyz_files_per_repetition} XYZ FILES PER FOLDER"
)
print(
    f"Total XYZ files generated: "
    f"{N * xyz_files_per_repetition}"
)
print("=" * 60)