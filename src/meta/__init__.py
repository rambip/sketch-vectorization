from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent


def project_file(relative_path):
    return ROOT_DIR / relative_path
