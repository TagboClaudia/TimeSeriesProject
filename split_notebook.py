import json
import os
from pathlib import Path

def split_jupyter_notebook(input_path, num_parts=3, output_dir=None):
    """
    Split a Jupyter notebook into multiple parts while maintaining valid JSON structure.

    Parameters:
    -----------
    input_path : str
        Path to the input .ipynb file
    num_parts : int
        Number of parts to split into (default: 3)
    output_dir : str or None
        Directory to save output files. If None, uses input file's directory.

    Returns:
    --------
    list : List of output file paths
    """

    # Validate input
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    if not input_path.suffix == '.ipynb':
        raise ValueError(f"File must be a .ipynb file: {input_path}")

    # Set output directory
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Read and parse the notebook as JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # Extract cells
    cells = notebook.get('cells', [])
    total_cells = len(cells)

    if total_cells == 0:
        raise ValueError("Notebook has no cells")

    # Calculate cells per part (ceil division to distribute remainder)
    cells_per_part = (total_cells + num_parts - 1) // num_parts

    # Split cells into parts
    cell_parts = []
    for i in range(0, total_cells, cells_per_part):
        cell_parts.append(cells[i:i + cells_per_part])

    # Ensure we have exactly num_parts (pad with empty list if needed)
    while len(cell_parts) < num_parts:
        cell_parts.append([])

    # Create and save each part
    output_files = []
    base_name = input_path.stem

    for i, part_cells in enumerate(cell_parts, 1):
        # Create new notebook structure
        part_notebook = {
            "cells": part_cells,
            "metadata": notebook.get("metadata", {}),
            "nbformat": notebook.get("nbformat", 4),
            "nbformat_minor": notebook.get("nbformat_minor", 0)
        }

        # Add metadata to indicate this is a split part
        part_notebook["metadata"]["split_info"] = {
            "original_file": input_path.name,
            "part": i,
            "total_parts": num_parts,
            "total_cells_original": total_cells,
            "cells_in_this_part": len(part_cells)
        }

        # Save to file
        output_path = output_dir / f"{base_name}_part{i}_{num_parts}.ipynb"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(part_notebook, f, indent=1, ensure_ascii=False)

        output_files.append(str(output_path))
        print(f"  ✓ Saved part {i}/{num_parts}: {output_path.name} ({len(part_cells)} cells)")

    return output_files


def split_all_notebooks_in_project(project_root, num_parts=3):
    """
    Find and split all Jupyter notebooks in the project.

    Parameters:
    -----------
    project_root : str
        Root directory of the project
    num_parts : int
        Number of parts to split each notebook into

    Returns:
    --------
    dict : Dictionary mapping original notebook paths to their split parts
    """

    project_root = Path(project_root)
    all_notebooks = list(project_root.rglob("*.ipynb"))

    # Filter out already split notebooks (contain '_part' in name)
    all_notebooks = [nb for nb in all_notebooks if '_part' not in nb.name]

    if not all_notebooks:
        print("No Jupyter notebooks found in the project!")
        return {}

    print(f"Found {len(all_notebooks)} notebook(s) to split:")
    for nb in all_notebooks:
        print(f"  - {nb.relative_to(project_root)}")

    print("\n" + "="*60)
    print(f"Splitting each notebook into {num_parts} part(s)...")
    print("="*60)

    results = {}

    for notebook_path in all_notebooks:
        print(f"\n📓 Processing: {notebook_path.relative_to(project_root)}")
        try:
            # Split the notebook in its own directory
            split_parts = split_jupyter_notebook(
                input_path=notebook_path,
                num_parts=num_parts,
                output_dir=notebook_path.parent  # Save in same directory
            )
            results[str(notebook_path)] = split_parts
            print(f"  ✅ Successfully split into {len(split_parts)} parts")
        except Exception as e:
            print(f"  ❌ Error splitting {notebook_path.name}: {e}")

    return results


def split_specific_notebook(project_root, notebook_name, num_parts=3):
    """
    Split a specific notebook by name.

    Parameters:
    -----------
    project_root : str
        Root directory of the project
    notebook_name : str
        Name of the notebook file (e.g., "time_serie_feature-engineering.ipynb")
    num_parts : int
        Number of parts to split into
    """

    project_root = Path(project_root)
    notebook_path = None

    # Search for the notebook
    for nb_path in project_root.rglob(notebook_name):
        notebook_path = nb_path
        break

    if notebook_path is None:
        print(f"❌ Notebook '{notebook_name}' not found in project!")
        return None

    print(f"📓 Found notebook: {notebook_path.relative_to(project_root)}")
    print(f"🔄 Splitting into {num_parts} part(s)...")

    split_parts = split_jupyter_notebook(
        input_path=notebook_path,
        num_parts=num_parts,
        output_dir=notebook_path.parent
    )

    print(f"\n✅ Success! Split into {len(split_parts)} parts:")
    for part in split_parts:
        print(f"  - {Path(part).name}")

    return split_parts


# ========== MAIN EXECUTION FOR YOUR PROJECT ==========

if __name__ == "__main__":

    # Set your project root (current directory)
    project_root = Path(".")

    # Split only the feature engineering notebook
    print("="*60)
    print("JUPYTER NOTEBOOK SPLITTER - FEATURE ENGINEERING NOTEBOOK")
    print("="*60)

    split_specific_notebook(
        project_root=project_root,
        notebook_name="time_serie_feature-engineering.ipynb",
        num_parts=3
    )

    print("\n" + "="*60)
    print("✓ DONE! The notebook has been split into 3 parts.")
    print("="*60)
    print("\n📁 Location: notebooks/feature_engineering/")
    print("   • time_serie_feature-engineering_part1_3.ipynb  (first 33%)")
    print("   • time_serie_feature-engineering_part2_3.ipynb  (middle 33%)")
    print("   • time_serie_feature-engineering_part3_3.ipynb  (last 33%)")
    print("\n💡 Tip: Now you can read only the first part to analyze!")