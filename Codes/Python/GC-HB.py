import math
import random
from pathlib import Path

# Approximate covalent radii, in ångströms (Å)
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

filename = "WC-GC.xyz"  # Updated to the GC filename

# Read the file
with open(filename, "r") as file:
    lines = file.readlines()

atom_count = int(lines[0])
atoms = []

# Each atom line contains its element and x, y, z coordinates
for line in lines[2:2 + atom_count]:
    parts = line.split()

    element = parts[0].capitalize()
    x, y, z = map(float, parts[1:4])

    atoms.append((element, (x, y, z)))

# These will store the bonds and each atom's neighbors
bonds = []
neighbors = [[] for _ in atoms]

# Compare each pair of atoms
for i in range(len(atoms)):
    element1, xyz1 = atoms[i]

    for j in range(i + 1, len(atoms)):
        element2, xyz2 = atoms[j]

        # Calculate the distance between the two atoms
        dx = xyz1[0] - xyz2[0]
        dy = xyz1[1] - xyz2[1]
        dz = xyz1[2] - xyz2[2]
        distance = math.sqrt(dx**2 + dy**2 + dz**2)

        # Estimate whether they are bonded based on their radii
        bond_limit = 1.2 * (radii[element1] + radii[element2])

        if distance <= bond_limit:
            bonds.append((i, j, distance))
            neighbors[i].append(j)
            neighbors[j].append(i)

# Print each atom and its neighbors
print("Atoms and neighbors:")
for i, (element, xyz) in enumerate(atoms):
    neighbor_list = [
        f"{j + 1} ({atoms[j][0]})" for j in neighbors[i]
    ]

    print(
        f"Atom {i + 1}: {element}, "
        f"coordinates = {xyz}, "
        f"neighbors = {', '.join(neighbor_list) if neighbor_list else 'none'}"
    )

# Print the estimated bonds and their lengths
print("\nEstimated bonds:")
for i, j, distance in bonds:
    print(
        f"Atom {i + 1} ({atoms[i][0]}) — "
        f"Atom {j + 1} ({atoms[j][0]}): {distance:.3f} Å"
    )

max_h_acceptor_distance = 3.0
min_dha_angle = 120.0

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

        for acceptor, (acceptor_element, acceptor_xyz) in enumerate(atoms):
            if acceptor_element not in possible_acceptors:
                continue
            if acceptor == donor or acceptor in neighbors[donor]:
                continue

            dx = h_xyz[0] - acceptor_xyz[0]
            dy = h_xyz[1] - acceptor_xyz[1]
            dz = h_xyz[2] - acceptor_xyz[2]
            distance = math.sqrt(dx**2 + dy**2 + dz**2)

            # Angle donor–H···acceptor, measured at H
            v1 = [donor_xyz[k] - h_xyz[k] for k in range(3)]
            v2 = [acceptor_xyz[k] - h_xyz[k] for k in range(3)]

            dot = sum(v1[k] * v2[k] for k in range(3))
            len1 = math.sqrt(sum(x**2 for x in v1))
            len2 = math.sqrt(sum(x**2 for x in v2))

            cosine = max(-1.0, min(1.0, dot / (len1 * len2)))
            angle = math.degrees(math.acos(cosine))

            if distance <= max_h_acceptor_distance and angle >= min_dha_angle:
                hydrogen_bonds.append(
                    (donor, h, acceptor, distance, angle)
                )

# Use stricter geometry criteria to separate the stronger candidates
strong = [
    hb for hb in hydrogen_bonds
    if hb[3] <= 2.5 and hb[4] >= 150.0
]
other = [hb for hb in hydrogen_bonds if hb not in strong]

# Rank the strong candidates: shorter distance first, then straighter angle
strong.sort(key=lambda hb: (hb[3], -hb[4]))

def print_bonds(title, bond_list):
    print(f"\n{title}:")
    if not bond_list:
        print("  None found")
        return

    for donor, h, acceptor, distance, angle in bond_list:
        print(
            f"  Donor {donor + 1} ({atoms[donor][0]})–"
            f"H {h + 1}···"
            f"acceptor {acceptor + 1} ({atoms[acceptor][0]}): "
            f"{distance:.3f} Å, {angle:.1f}°"
        )

print_bonds("Stronger geometric candidates", strong)
print_bonds("Other possible candidates", other)

# Generate 100 random offsets between -1.00 and 1.00
offsets = [random.uniform(-1.00, 1.00) for _ in range(100)]

def find_components(neighbors):
    """Group atoms connected by the inferred covalent bonds."""
    component_id = [-1] * len(neighbors)
    components = []

    for start in range(len(neighbors)):
        if component_id[start] != -1:
            continue

        group_number = len(components)
        component = []
        stack = [start]
        component_id[start] = group_number

        while stack:
            atom = stack.pop()
            component.append(atom)

            for neighbor in neighbors[atom]:
                if component_id[neighbor] == -1:
                    component_id[neighbor] = group_number
                    stack.append(neighbor)

        components.append(component)

    return component_id, components

component_id, components = find_components(neighbors)

# Define a physical minimum limit for the hydrogen···acceptor distance (in Å)
min_h_acceptor_limit = 1.5

original_coords = [list(atom[1]) for atom in atoms]
frames = []

# Track min and max H-bond lengths across generated frames
min_hbond_length = float('inf')
max_hbond_length = float('-inf')

for bond_number, (donor, h, acceptor, old_distance, angle) in enumerate(strong, start=1):

    if component_id[donor] == component_id[acceptor]:
        print(
            f"Skipping bond {bond_number}: donor and acceptor are in the "
            "same connected molecule. It needs a flexible-group adjustment."
        )
        continue

    h_xyz = original_coords[h]
    a_xyz = original_coords[acceptor]

    # Unit vector pointing from H toward the acceptor
    vector = [a_xyz[k] - h_xyz[k] for k in range(3)]
    length = math.sqrt(sum(value**2 for value in vector))
    direction = [value / length for value in vector]

    acceptor_molecule = components[component_id[acceptor]]

    for offset in offsets:
        target_distance = old_distance + offset
        
        # Enforce physical minimum distance limit
        if target_distance < min_h_acceptor_limit:
            continue

        # Update min and max trackers
        if target_distance < min_hbond_length:
            min_hbond_length = target_distance
        if target_distance > max_hbond_length:
            max_hbond_length = target_distance

        # Start each version from the original structure
        new_coords = [xyz[:] for xyz in original_coords]

        # Translate the whole acceptor molecule along the H···acceptor line
        shift = [offset * direction[k] for k in range(3)]
        for atom_index in acceptor_molecule:
            for k in range(3):
                new_coords[atom_index][k] += shift[k]

        label = (
            f"HB {bond_number}; atoms {donor + 1}-H{h + 1}...{acceptor + 1}; "
            f"target H-A={target_distance:.3f} A; angle={angle:.1f} deg"
        )
        frames.append((bond_number, offset, target_distance, label, new_coords))

output_folder = Path("output")
output_folder.mkdir(exist_ok=True)

# Save individual coordinate frames
for number, (bond_number, offset, target_distance, label, coords) in enumerate(
    frames, start=1
):
    offset_name = f"{offset:+.2f}".replace("+", "plus").replace("-", "minus")
    filename = f"hb{bond_number}_change_{offset_name}A.xyz"
    filepath = output_folder / filename

    with open(filepath, "w") as output:
        output.write(f"{len(atoms)}\n{label}\n")

        for (element, _), (x, y, z) in zip(atoms, coords):
            output.write(f"{element:2s} {x:.8f} {y:.8f} {z:.8f}\n")

    print(f"Saved: {filepath}")

# Save the min and max summary statistics to a separate text file
summary_filepath = output_folder / "hbond_summary.txt"
with open(summary_filepath, "w") as summary_file:
    summary_file.write("Hydrogen Bond Length Summary Statistics (GC Base Pair)\n")
    summary_file.write("=" * 50 + "\n")
    if frames:
        summary_file.write(f"Minimum H-Bond Length: {min_hbond_length:.3f} Å\n")
        summary_file.write(f"Maximum H-Bond Length: {max_hbond_length:.3f} Å\n")
        summary_file.write(f"Total Frames Generated: {len(frames)}\n")
    else:
        summary_file.write("No valid frames generated.\n")

print(f"\nSummary saved successfully: {summary_filepath}")