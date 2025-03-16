import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import argparse
import re
from src.utils import load_paths, load_yaml, construct_model_id
from src.neural_data.data import load_file_paths, load_neural_data, convert_df_to_string, load_audio_path
from functools import partial
        

def main(model_name, dataset, devices, batch_size):

    paths = load_paths() 
    config = load_yaml('config')[model_name]
    stimulus_paths = load_file_paths(data_path=paths["DATA"], subject='03')
    stimulus_paths.sort()

    # Process each stimulus
    for stimulus_path in stimulus_paths[2:]:
        stimulus_df, _ = load_neural_data(data_dir=paths["DATA"], file_path=stimulus_path)
        stimulus_str = convert_df_to_string(stimulus_df)
        
        match = re.search(r'Jobs[1-3]', str(stimulus_path))
        stimulus_name = match.group(0).lower()
        identifier_fn = partial(
            construct_model_id,
            model_iden=model_name,
            dataset=dataset,
            stimulus= stimulus_name,
            )

        if config['model_type'] == 'llm': # Initialize model on the first device
            from src.model_embeddings.models.llm import LLM
            
            model = LLM(
                model_name=config['model_iden'],
                num_layers=config['num_layers'],
                max_window_size=config['max_window_size'],
                is_bidirectional = config['bidirectional']
            )
            token_ids = model.get_token_ids(input_string=stimulus_str)  # Use `module` for DataParallel models   
            # Process activations on the devices
            model_embeddings = model.get_embeddings(
                identifier_fn=identifier_fn,
                token_ids=token_ids,
                batch_size=batch_size,
                devices=devices
            )

        elif config['model_type'] == 'speech': 
            from src.model_embeddings.models.acoustic import WhisperDecoder
            if 'decoder' in model_name.lower():
                print('using decoder')
                model = WhisperDecoder(model_name=config['model_iden'], 
                                num_layers=config['num_layers'])         
                model_embeddings = model.get_decoder_embeddings(identifier_fn=identifier_fn,
                                            devices=devices,
                                            audio_file_path = Path(paths['STIMULI_WAV']) / load_audio_path(stimulus_path) , 
                                            neural_file_path = Path(paths["DATA"]) / stimulus_path)       
            else:
                from src.model_embeddings.models.acoustic import WhisperEncoder
                model = WhisperEncoder(model_name=config['model_iden'], 
                                num_layers=config['num_layers'])
                model_embeddings = model.get_embeddings(identifier_fn=identifier_fn,
                                                    devices=devices,
                                                    audio_file_path = Path(paths['STIMULI_WAV']) / load_audio_path(stimulus_path) , 
                                                    neural_file_path = Path(paths["DATA"]) / stimulus_path)
        elif config['model_type'] == 'acoustic': 
            from src.model_embeddings.models.acoustic import AcousticModel
            model = AcousticModel(model_name = config['model_iden']) 
            model_embeddings = model.get_embeddings(identifier_fn= identifier_fn, 
                                                    audio_file_path = Path(paths['STIMULI_WAV']) / load_audio_path(stimulus_path) , 
                                                    neural_file_path = Path(paths["DATA"]) / stimulus_path)
        else:
            raise NameError(f'The model: {model} is not available')
        
        del model_embeddings


if __name__ == "__main__":
    # Create ArgumentParser object
    parser = argparse.ArgumentParser()
    # Required positional arguments
    parser.add_argument("--model_name", type=str, required=True, help="Model to use")
    parser.add_argument("--dataset", type=str, default="ieeg", help="Dataset to extract activations from")
    parser.add_argument("--devices", type=list, nargs='+', default=[f'cuda:{i}' for i in range(8)], help="List of CUDA devices or ['cpu']")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")

    # Parse arguments
    args = parser.parse_args()

    # Call main with required arguments explicitly and optional ones as kwargs
    main(args.model_name, dataset=args.dataset, devices=args.devices, batch_size=args.batch_size)

