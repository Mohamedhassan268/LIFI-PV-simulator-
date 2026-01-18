# utils/output_manager.py
"""
Output Manager - Organizes all test outputs into timestamped folders.

Creates folders like: outputs/Jan15_10pm_test1/
                      outputs/Jan15_10pm_test2/
                      
With subfolders: plots/, sensitivity/, csv/, validation/
"""

import os
from datetime import datetime
from pathlib import Path


def get_human_readable_timestamp():
    """
    Generate human-readable timestamp like 'Jan15_10pm'.
    """
    now = datetime.now()
    
    # Month abbreviation + day
    month_day = now.strftime('%b%d')  # Jan15, Feb22, etc.
    
    # Hour in 12h format with am/pm
    hour = now.strftime('%I%p').lstrip('0').lower()  # 10pm, 9am, etc.
    
    return f"{month_day}_{hour}"


def get_test_number(base_dir: str, timestamp: str) -> int:
    """
    Find the next available test number for today.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return 1
    
    existing = [d.name for d in base_path.iterdir() if d.is_dir()]
    
    # Find folders matching this timestamp pattern
    matching = [d for d in existing if d.startswith(timestamp)]
    
    if not matching:
        return 1
    
    # Extract test numbers
    test_nums = []
    for folder in matching:
        parts = folder.split('_test')
        if len(parts) == 2 and parts[1].isdigit():
            test_nums.append(int(parts[1]))
    
    return max(test_nums, default=0) + 1


def create_output_folder(base_dir: str = 'outputs') -> dict:
    """
    Create a new timestamped output folder with subfolders.
    
    Returns:
        dict with paths for each output type
    """
    timestamp = get_human_readable_timestamp()
    test_num = get_test_number(base_dir, timestamp)
    
    folder_name = f"{timestamp}_test{test_num}"
    folder_path = Path(base_dir) / folder_name
    
    # Create main folder and subfolders
    subfolders = ['plots', 'sensitivity', 'csv', 'validation']
    paths = {'root': str(folder_path)}
    
    for sub in subfolders:
        sub_path = folder_path / sub
        sub_path.mkdir(parents=True, exist_ok=True)
        paths[sub] = str(sub_path)
    
    # Create README with run info
    readme_path = folder_path / 'README.txt'
    with open(readme_path, 'w') as f:
        f.write(f"Li-Fi PV Simulator Output\n")
        f.write(f"========================\n\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Folder: {folder_name}\n\n")
        f.write(f"Contents:\n")
        f.write(f"  - plots/       : Figure outputs (PNG)\n")
        f.write(f"  - sensitivity/ : Sensitivity analysis charts\n")
        f.write(f"  - csv/         : Data exports (CSV)\n")
        f.write(f"  - validation/  : Paper validation results\n")
    
    print(f"\n📁 Created output folder: {folder_path}")
    
    return paths


# Global output paths - initialized on first use
_current_output = None


def get_output_paths(force_new: bool = False) -> dict:
    """
    Get current output paths. Creates new folder on first call.
    
    Args:
        force_new: If True, create a new test folder
        
    Returns:
        dict with paths for each output type
    """
    global _current_output
    
    if _current_output is None or force_new:
        _current_output = create_output_folder()
    
    return _current_output


def get_plots_dir() -> str:
    """Get the current plots output directory."""
    return get_output_paths()['plots']


def get_sensitivity_dir() -> str:
    """Get the current sensitivity output directory."""
    return get_output_paths()['sensitivity']


def get_csv_dir() -> str:
    """Get the current CSV output directory."""
    return get_output_paths()['csv']


def get_validation_dir() -> str:
    """Get the current validation output directory."""
    return get_output_paths()['validation']


def reset_output():
    """Reset output paths to force creation of new folder on next use."""
    global _current_output
    _current_output = None


# =============================================================================
# GENERAL SIMULATION OUTPUT MANAGEMENT
# =============================================================================

# Cache for general output directory
_general_output_cache = None


def get_general_output_dir(force_new: bool = True, base_dir: str = 'outputs') -> str:
    """
    Get output directory for general (non-paper) simulations.
    
    Creates a folder structure like:
        outputs/general/Jan16_11pm_test1/
        outputs/general/Jan16_11pm_test2/
    
    Args:
        force_new: If True, always create a new testN folder.
                   If False, reuse cached folder from this session.
        base_dir: Base output directory (default: 'outputs')
        
    Returns:
        Path to the test folder (e.g., 'outputs/general/Jan16_11pm_test1')
    """
    global _general_output_cache
    
    # Check cache (unless force_new)
    if not force_new and _general_output_cache is not None:
        return _general_output_cache
    
    # Build general folder path
    general_folder = Path(base_dir) / 'general'
    general_folder.mkdir(parents=True, exist_ok=True)
    
    # Get timestamp and test number
    timestamp = get_human_readable_timestamp()
    test_num = get_test_number(str(general_folder), timestamp)
    
    test_folder = general_folder / f"{timestamp}_test{test_num}"
    test_folder.mkdir(parents=True, exist_ok=True)
    
    # Create run info
    info_path = test_folder / 'run_info.txt'
    with open(info_path, 'w') as f:
        f.write(f"Li-Fi PV Simulator - General Simulation\n")
        f.write(f"=======================================\n\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Folder: {test_folder.name}\n")
    
    # Cache it
    _general_output_cache = str(test_folder)
    
    print(f"📁 Created general output: {test_folder}")
    
    return str(test_folder)


def reset_general_output():
    """Reset general output cache."""
    global _general_output_cache
    _general_output_cache = None


# =============================================================================
# PAPER-SPECIFIC OUTPUT MANAGEMENT
# =============================================================================

# Supported paper identifiers
PAPER_NAMES = [
    'correa_2025',
    'kadirvelu_2021',
    'xu_2024',
    'gonzalez_2024',
    'sarwar_2017',
    'oliveira_2024',
]

# Cache for paper output directories (to avoid creating multiple folders per run)
_paper_output_cache = {}


def _get_next_test_number(paper_dir: Path) -> int:
    """
    Find the next available test number for a paper folder.
    
    Args:
        paper_dir: Path to the paper folder (e.g., outputs/paper_correa_2025/)
        
    Returns:
        Next test number (1 if no tests exist)
    """
    if not paper_dir.exists():
        return 1
    
    existing = [d.name for d in paper_dir.iterdir() if d.is_dir()]
    
    # Find folders matching 'testN' pattern
    test_nums = []
    for folder in existing:
        if folder.startswith('test') and folder[4:].isdigit():
            test_nums.append(int(folder[4:]))
    
    return max(test_nums, default=0) + 1


def _create_run_info(test_dir: Path, paper_name: str):
    """
    Create run_info.txt with timestamp and metadata.
    
    Args:
        test_dir: Path to the test folder
        paper_name: Paper identifier
    """
    info_path = test_dir / 'run_info.txt'
    with open(info_path, 'w') as f:
        f.write(f"Li-Fi PV Simulator - Paper Validation\n")
        f.write(f"=====================================\n\n")
        f.write(f"Paper: {paper_name}\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Folder: {test_dir.name}\n")


def get_paper_output_dir(paper_name: str, force_new: bool = True, base_dir: str = 'outputs') -> str:
    """
    Get output directory for a specific paper validation.
    
    Creates a folder structure like:
        outputs/paper_correa_2025/test1/
        outputs/paper_correa_2025/test2/
    
    Args:
        paper_name: Paper identifier (e.g., 'correa_2025', 'xu_2024')
        force_new: If True, always create a new testN folder.
                   If False, reuse cached folder from this session.
        base_dir: Base output directory (default: 'outputs')
        
    Returns:
        Path to the test folder (e.g., 'outputs/paper_correa_2025/test3')
    """
    global _paper_output_cache
    
    # Validate paper name
    if paper_name not in PAPER_NAMES:
        print(f"⚠️  Warning: '{paper_name}' not in PAPER_NAMES. Adding dynamically.")
        PAPER_NAMES.append(paper_name)
    
    # Check cache (unless force_new)
    if not force_new and paper_name in _paper_output_cache:
        return _paper_output_cache[paper_name]
    
    # Build paper folder path
    paper_folder = Path(base_dir) / f"paper_{paper_name}"
    paper_folder.mkdir(parents=True, exist_ok=True)
    
    # Get next test number
    test_num = _get_next_test_number(paper_folder)
    test_folder = paper_folder / f"test{test_num}"
    test_folder.mkdir(parents=True, exist_ok=True)
    
    # Create run info
    _create_run_info(test_folder, paper_name)
    
    # Cache it
    _paper_output_cache[paper_name] = str(test_folder)
    
    print(f"📁 Created paper output: {test_folder}")
    
    return str(test_folder)


def reset_paper_output(paper_name: str = None):
    """
    Reset paper output cache.
    
    Args:
        paper_name: Specific paper to reset, or None to reset all.
    """
    global _paper_output_cache
    if paper_name is None:
        _paper_output_cache = {}
    elif paper_name in _paper_output_cache:
        del _paper_output_cache[paper_name]


# ========== TEST ==========

if __name__ == "__main__":
    print("Testing Output Manager...")
    
    # Create first test folder
    paths = get_output_paths(force_new=True)
    print(f"\nTest 1 paths:")
    for key, path in paths.items():
        print(f"  {key}: {path}")
    
    # Create second test folder
    paths2 = get_output_paths(force_new=True)
    print(f"\nTest 2 paths:")
    for key, path in paths2.items():
        print(f"  {key}: {path}")
    
    print("\n✅ Output manager working!")
