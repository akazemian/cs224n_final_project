import numpy as np
import torch
import argparse
from src.utils import load_paths, load_yaml, construct_model_id
from src.model_score.scorer import Scorer
from transformers import AutoTokenizer
from functools import partial

paths = load_paths() 

def main(model, subjects, devices):

    print(model)
    config = load_yaml('config')[model]
    
    tokenizer = AutoTokenizer.from_pretrained(config['model_iden'], cache_dir=paths["CACHE"]) if config['model_type'] == 'llm' else None

    scr = Scorer(model_name = config['model_iden'], 
            tokenizer=tokenizer,
            )
    for subject in subjects:

        identifier_fn = partial(
            construct_model_id,
            model_iden=model,
            dataset='ieeg', 
            subject=subject,
                )
        results = scr.get_scores(identifier_fn = identifier_fn,
                                model_iden=model,
                                subject=subject,
                                num_layers=config['num_layers'],
                                devices= devices)
            
        del results
        torch.cuda.empty_cache()


def parse_devices(device_arg):
    """
    Parses the --devices argument.
    If a single string is provided, return it as is.
    If a comma-separated string is provided, split it into a list.
    """
    if ',' in device_arg:
        return device_arg.split(',')  # Returns a list of devices
    return device_arg  # Returns a single device as a string

    
if __name__ == '__main__':
    # Create ArgumentParser object
    parser = argparse.ArgumentParser()

    # Required positional arguments
    parser.add_argument("--model", type=str, required=True, help="Model to use")
    parser.add_argument("--subjects", type=parse_devices, default=['03']) #,'05','07','10','11']
    parser.add_argument("--devices", type=parse_devices, default=[f'cuda:{i}' for i in range(8)], 
                        help="CUDA device(s), either a string ('cuda:0') or a comma-separated list ('cuda:0,cuda:1').")

    # Parse arguments
    args = parser.parse_args()  

    # Call main with required arguments explicitly and optional ones as kwargs
    main(args.model, subjects=args.subjects, devices=args.devices)
