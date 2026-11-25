import os
import math
import random
from pathlib import Path

def load_xyz(filepath):
    """Loads an atomic structure from an XYZ file."""
    with open(filepath, "r") as f:
        lines = f.readlines()
    atom_count = int(lines[0].strip())
    comment = lines[1].strip()
    atoms = []
    for line in lines[2:2 + atom_count]:
        parts = line.split()
        element = parts[0].capitalize()
        x, y, z = map(float, parts[1:4])
        atoms.append((element, [x, y, z]))
    return atoms, comment

def get_bonds(atoms):
    """Infers covalent bonds based on atomic radii."""
    radii = {
        "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, 
        "F": 0.57, "P": 1.07, "S": 1.05, "Cl": 1.02
    }
    neighbors = [[] for _ in atoms]
    for i in range(len(atoms)):
        elem1, xyz1 = atoms[i]
        for j in range(i + 1, len(atoms)):
            elem2, xyz2 = atoms[j]
            dx = xyz1[0] - xyz2[0]
            dy = xyz1[1] - xyz2[1]
            dz = xyz1[2] - xyz2[2]
            dist = math.sqrt(dx**2 + dy**2 + dz**2)
            limit = 1.2 * (radii.get(elem1, 0.7) + radii.get(elem2, 0.7))
            if dist <= limit:
                neighbors[i].append(j)
                neighbors[j].append(i)
    return neighbors

def find_components(neighbors):
    """Groups atoms into connected molecular components (Monomer 1 and Monomer 2)."""
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

def generate_multimode_ensemble(input_xyz, output_dir, num_samples=100, max_dispatch=0.03):
    """
    Generates N randomized geometry files with multi-mode atomic displacements 
    while preserving base-pair connectivity.
    """
    atoms, comment = load_xyz(input_xyz)
    neighbors = get_bonds(atoms)
    component_id, components = find_components(neighbors)
    
    if len(components) != 2:
        print(f"Warning: Found {len(components)} connected components. Expected exactly 2 monomers.")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    original_coords = [list(atom[1]) for atom in atoms]
    
    print(f"Starting multi-mode random displacement generation...")
    print(f"Target samples: {num_samples} | Max displacement amplitude: ±{max_dispatch} Å")
    
    success_count = 0
    
    while success_count < num_samples:
        success_count += 1
        new_coords = [xyz[:] for xyz in original_coords]
        
        # Apply multi-mode displacements grouped by monomer to maintain pairing integrity
        for comp in components:
            shift_x = random.uniform(-max_dispatch, max_dispatch)
            shift_y = random.uniform(-max_dispatch, max_dispatch)
            shift_z = random.uniform(-max_dispatch, max_dispatch)
            
            for idx in comp:
                dx = random.uniform(-max_dispatch, max_dispatch)
                dy = random.uniform(-max_dispatch, max_dispatch)
                dz = random.uniform(-max_dispatch, max_dispatch)
                
                new_coords[idx][0] += shift_x + dx
                new_coords[idx][1] += shift_y + dy
                new_coords[idx][2] += shift_z + dz
                
        # Save generated geometry
        filename = f"multimode_sample_{success_count:04d}.xyz"
        filepath = output_path / filename
        
        with open(filepath, "w") as f:
            f.write(f"{len(atoms)}\nMulti-mode displaced sample {success_count} (disp={max_dispatch})\n")
            for (elem, _), (x, y, z) in zip(atoms, new_coords):
                f.write(f"{elem:2s} {x:.8f} {y:.8f} {z:.8f}\n")
                
        if success_count % max(1, num_samples // 10) == 0 or success_count == num_samples:
            pct = (success_count / num_samples) * 100
            print(f"Progress: {pct:.0f}% completed ({success_count}/{num_samples} files saved)")

    print(f"\nSuccessfully saved all {num_samples} files to folder: '{output_path}'")

if __name__ == "__main__":
    # Configuration using raw strings (r"...") to avoid path escape issues
    input_file = "WC-GC.xyz"         
# Update with your actual input file path
    target_folder = "GC-MM"
# Destination folder for generated .xyz files
    
    generate_multimode_ensemble(
        input_xyz=input_file, 
        output_dir=target_folder, 
        num_samples=1000,         # Number of randomized structures to generate
        max_dispatch=0.03        # Amplitude of random shifts in Ångströms
    )