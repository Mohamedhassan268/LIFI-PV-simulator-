import importlib
import sys
import os
import argparse

def list_papers():
    """List available paper modules in the papers/ directory."""
    papers_dir = os.path.join(os.path.dirname(__file__), 'papers')
    papers = []
    if not os.path.exists(papers_dir):
        return []
        
    for f in os.listdir(papers_dir):
        if f.endswith('.py') and not f.startswith('_'):
            papers.append(f[:-3])
    return papers

def run_paper(paper_name):
    """Dynamically import and run a paper validation module."""
    try:
        module = importlib.import_module(f'papers.{paper_name}')
        if hasattr(module, 'run_validation'):
            module.run_validation()
        else:
            print(f"Error: Module 'papers/{paper_name}.py' has no 'run_validation()' function.")
    except ImportError:
        print(f"Error: Could not import paper '{paper_name}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Li-Fi PV Simulator Validation Runner")
    parser.add_argument('paper', nargs='?', help="Name of the paper module to run (e.g., kadirvelu_2021)")
    parser.add_argument('--list', action='store_true', help="List available papers")
    
    args = parser.parse_args()
    
    available_papers = list_papers()
    
    if args.list:
        print("Available Papers:")
        for p in available_papers:
            print(f"  - {p}")
        sys.exit(0)
        
    if not args.paper:
        print("Error: No paper specified.")
        print("Available papers:")
        for p in available_papers:
            print(f"  - {p}")
        print("\nUsage: python run_validation.py <paper_name>")
        sys.exit(1)
        
    if args.paper not in available_papers:
        print(f"Error: Paper '{args.paper}' not found.")
        print(f"Similar available: {available_papers}")
        sys.exit(1)
        
    run_paper(args.paper)
