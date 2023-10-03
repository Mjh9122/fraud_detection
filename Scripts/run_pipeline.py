import argparse
import run_models
import generate_figures
import importlib.util

def main(directory):
    # Import feature functions
    feature_functions_spec = importlib.util.spec_from_file_location('feature_functions', f"./{directory}//feature_functions.py")
    feature_functions = importlib.util.module_from_spec(feature_functions_spec)
    feature_functions_spec.loader.exec_module(feature_functions)
    funcs = feature_functions.functions
    func_names = feature_functions.function_names

    # Import create datasets
    create_datasets_spec = importlib.util.spec_from_file_location('create_datasets', f"./{directory}/create_datasets.py")
    create_datasets = importlib.util.module_from_spec(create_datasets_spec)
    create_datasets_spec.loader.exec_module(create_datasets)
    create_datasets.main(funcs, func_names, 'Scripts//original_data.csv', f'{directory}/feature_data', verbose=True)
    
    # Run models 
    run_models.main(f'{directory}/feature_data', f'{directory}/score_data')

    # Make figures
    generate_figures.main(f'{directory}/score_data', f'{directory}/figures')

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    main(args.directory)