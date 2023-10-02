import argparse
import run_models
import generate_figures
import importlib.util
import sys

def main(directory):
    # Import feature functions
    feature_functions_spec = importlib.util.spec_from_file_location('feature_functions', f".\{directory}\\feature_functions.py")
    feature_functions = importlib.util.module_from_spec(feature_functions_spec)
    feature_functions_spec.loader.exec_module(feature_functions)

    # Import create datasets
    create_datasets_spec = importlib.util.spec_from_file_location('create_datasets', f".\{directory}\create_datasets.py")
    create_datasets = importlib.util.module_from_spec(create_datasets_spec)
    create_datasets_spec.loader.exec_module(create_datasets)
    
    #import feature_functions

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    main(args.directory)