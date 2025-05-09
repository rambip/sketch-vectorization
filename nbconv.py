"""
Notebook Converter - Convert between Jupyter notebooks and Python files.
Stores Python files alongside notebooks with metadata in a special comment.
"""

import json
import hashlib
from pathlib import Path
from nbconvert.preprocessors import ExecutePreprocessor
import nbformat
from nbformat import NotebookNode


NB_DIR = Path("notebooks")
PY_DIR = NB_DIR
HEADER_COMMENT = "# %%\n"
HASH_PREFIX = "# %% nb-hash="


class NotebookFile:
    """Class representing a Jupyter notebook file"""

    def __init__(self, path: Path, nb_data: NotebookNode):
        self.path = path
        self.nb_data = nb_data

    @staticmethod
    def from_code_cells(path: Path, raw_cells: list[str]):
        """Convert Python file to notebook format"""

        # Create cells from chunks
        cells = []
        for raw_cell in raw_cells:
            if raw_cell.startswith('"""') and raw_cell.endswith('"""'):
                # Remove only the triple quotes at the beginning and end
                content = raw_cell[3:-3]
                cell = {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": content,
                }
            else:
                # Code cell
                cell = {
                    "cell_type": "code",
                    "metadata": {},
                    "source": raw_cell,
                    "outputs": [],
                    "execution_count": None,
                }
            cells.append(cell)

        # Create notebook data
        nb_data = nbformat.from_dict(
            {
                "cells": cells,
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3",
                    },
                    "language_info": {"name": "python", "version": "3.10"},
                },
                "nbformat": 4,
                "nbformat_minor": 4,
            },
        )

        return NotebookFile(path, nb_data)  # type: ignore

    def write(self, ep: ExecutePreprocessor):
        ep.preprocess(self.nb_data, km=ep.km)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.nb_data, f, indent=2)

    def write_to_py(self, hash):
        """Convert notebook to Python file format"""
        py_path = self.path.parent / f"{self.path.stem}.py"

        # Generate Python content from notebook cells
        chunks = []
        for cell in self.nb_data["cells"]:
            if cell["cell_type"] == "markdown":
                chunks.append(f'"""\n{"".join(cell["source"])}\n"""')
            elif cell["cell_type"] == "code":
                chunks.append("".join(cell["source"]))

        with open(py_path, "w", encoding="utf-8") as f:
            f.write(HASH_PREFIX)
            f.write(hash + "\n")
            f.write("\n# %%\n".join(chunks))

    def compute_hash(self):
        source = [c["source"] for c in self.nb_data["cells"]]
        return hashlib.md5(json.dumps(source).encode()).hexdigest()


def extract_hash_and_chunks(content):
    """Extract metadata from Python file content"""
    if content.startswith(HASH_PREFIX):
        lines = content.splitlines()
        hash = lines[0][len(HASH_PREFIX) :]
        content = "\n".join(lines[1:])
    else:
        hash = None

    parts = content.split(HEADER_COMMENT)
    return hash, [x.strip() for x in parts]


def load_python_file(path):
    """Load a Python file with metadata"""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return extract_hash_and_chunks(content)


def load_notebook(path):
    """Load a notebook file"""
    with open(path, encoding="utf-8") as f:
        data = nbformat.read(f, as_version=4)
    return NotebookFile(path, data)


def process_file_pair(stem, py_path, nb_path, ep):
    """Process a pair of notebook and Python files with the same stem.
    Returns the authoritative NotebookFile representing the current state."""
    
    # Case: Only notebook exists
    if py_path is None and nb_path is not None:
        nb = load_notebook(nb_path)
        print(f"{stem}.ipynb -> {stem}.py")
        return nb
    
    # Case: Only Python file exists    
    if nb_path is None and py_path is not None:
        hash_val, chunks = load_python_file(py_path)
        new_nb_path = NB_DIR / f"{stem}.ipynb"
        nb = NotebookFile.from_code_cells(new_nb_path, chunks)
        print(f"{stem}.py -> {stem}.ipynb")
        return nb
    
    # Both files exist - determine which is authoritative
    ref_hash, chunks = load_python_file(py_path)
    nb = load_notebook(nb_path)
    
    # Create a notebook representation from Python chunks
    nb_from_py = NotebookFile.from_code_cells(nb_path, chunks)
    
    nb_hash = nb.compute_hash()
    py_hash = nb_from_py.compute_hash()
    
    # Determine what changed and which is authoritative
    if ref_hash == nb_hash == py_hash:
        # No changes - either is fine
        return nb
    elif ref_hash == py_hash and ref_hash != nb_hash:
        # Notebook changed - notebook is authoritative
        print(f"{stem}.ipynb -> {stem}.py")
        return nb
    elif ref_hash == nb_hash and ref_hash != py_hash:
        # Python file changed - Python file is authoritative
        print(f"{stem}.py -> {stem}.ipynb")
        return nb_from_py
    else:
        # Both changed independently - notebook is authoritative by default
        print(f"Conflict in {stem}: both files changed independently (using notebook)")
        return nb


def convert_all():
    """Convert all files that need conversion"""
    # Find all Python and notebook files
    py_files = {file.stem: file for file in NB_DIR.glob("*.py")}
    nb_files = {file.stem: file for file in PY_DIR.glob("*.ipynb")}
    all_stems = set(py_files.keys()) | set(nb_files.keys())

    # Create execute preprocessor with custom kernel
    ep = ExecutePreprocessor(timeout=600)

    for stem in all_stems:
        py_path = py_files.get(stem)
        nb_path = nb_files.get(stem)

        # Get the authoritative notebook file
        nb = process_file_pair(stem, py_path, nb_path, ep)
        
        # Write both files from the authoritative source
        if nb:
            nb.write(ep)
            hash_val = nb.compute_hash()
            nb.write_to_py(hash_val)


if __name__ == "__main__":
    convert_all()
