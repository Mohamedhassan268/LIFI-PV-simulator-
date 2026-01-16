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
