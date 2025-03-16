import logging
import re
import yaml
import os
from typing import Literal 
from dotenv import load_dotenv
load_dotenv()
ROOT = os.getenv('ROOT')

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')
    
def load_yaml(file_name:Literal['config','paths']):
    with open(f"{ROOT}/FYP/{file_name}.yaml", "r") as f:
        yaml_file = yaml.safe_load(f)
    return yaml_file

def load_paths():
    paths = load_yaml('paths')
    resolved_paths = {
        key: value.replace('${ROOT}', ROOT).replace('${CACHE}', f"{ROOT}/.cache")
        for key, value in paths.items()
    }
    return resolved_paths

def extract_stimulus(file_name):
    # This regex searches for "stimulus=" followed by any characters that are not a hyphen
    pattern = re.compile(r"stimulus=(?P<stimulus>[^-]+)")
    match = pattern.search(file_name)
    if match:
        return match.group("stimulus")
    return None

def construct_model_id(model_iden: str, layer: str, dataset: str, stimulus:str = None, subject: str=None, time_lag:int=None, extension:str=None) -> str:

    parts = [f'model={model_iden}', f'layer={layer}', f'dataset={dataset}']

    if stimulus != None:
        if 'stimulus=' in stimulus:
            stimulus = extract_stimulus(stimulus).lower()
        parts.append(f'stimulus={stimulus}')

    if subject != None:
        parts.append(f'subject={subject}')

    if time_lag != None:
        parts.append(f'time_lag={time_lag}')
    
    if extension != None:
        parts.append(extension)

    return "_".join(parts).lower()

    

