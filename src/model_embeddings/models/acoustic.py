import os
import torch
from torch import nn
import torch.nn.functional as F
import mne
import warnings
warnings.filterwarnings("ignore")
import torchaudio
import torchaudio.transforms as T
import chcochleagram
from transformers import WhisperProcessor, WhisperModel, WhisperForConditionalGeneration
import math
from tqdm import tqdm 
import gc
import traceback
import re
from concurrent.futures import ThreadPoolExecutor
from src.model_embeddings.utils import cache
from src.utils import load_paths, construct_model_id


paths = load_paths()
DATA = paths["DATA"]
CACHE = paths["CACHE"]


def audio_timestamp_to_encoder_frame(timestamp_ms):
    """Maps an audio timestamp (in ms) to the corresponding encoder frame index."""
    return math.floor(timestamp_ms // 20)


class AcousticModel:
    def __init__(self, model_name:str, time_lag:int=None):
        self.model_name = model_name
        self.sample_rate_orig = 44100
        self.sample_rate_target = 40000
        self.time_lag = time_lag
        self.layers = 'none'
            
    @staticmethod
    def cache_file(identifier: str) -> str:
        path = os.path.join(paths["EMBEDDINGS_PATH"], identifier)  # Ensure CACHE is used
        print(f"Caching path: {path}")
        return path
    
    def downsample(self, speech_waveform:torch.Tensor):
        resampler = torchaudio.transforms.Resample(orig_freq=self.sample_rate_orig, 
                                                   new_freq=self.sample_rate_target)
        downsampled_waveform = resampler(speech_waveform)
        return downsampled_waveform
        
    def get_acoustic_model(self, devices):
        if self.model_name == 'spectrogram':
            return Spectrogram(device=devices)
        elif self.model_name == 'cochleagram':
            return Cochleagram(device=devices)
        else:
            raise ValueError(f"Unknow acoustic model name: {self.model_name}")

    @cache(lambda identifier, **kwargs: AcousticModel.cache_file(identifier))    
    def get_embeddings_(self, identifier: str, audio_file_path: str, neural_file_path: str, devices: list):
        
        speech_waveform, _ = torchaudio.load(audio_file_path)
        downsampled_waveform = self.downsample(speech_waveform)
        model = self.get_acoustic_model(devices[0]) #always run on one device
        y = model(downsampled_waveform[0,:]) # single channel, shape [n_features, n_timepoints]
        data = mne.read_epochs(neural_file_path, preload=True, verbose=False) 
        word_onset_idx = data.metadata['Start'].to_list() #get word onset times in s
        all_words = []
        time_coefs = data.times # get the brain time window indeces in s [-t2,-t1,0,t1,t2...]
        for w in word_onset_idx:
            try:
                times_in_window = w + time_coefs # add time window to word onset to get the actual time window
                if self.time_lag:
                    times_in_window += self.time_lag #add the lag
                idx = times_in_window * self.sample_rate_target # multiply by sr to get corresponding points in y
                all_words.append(y[:, idx].unsqueeze(dim=0)) # extract points from i and concatenate 
            except IndexError:
                print('indexing problem...')
                print('file:',neural_file_path)
                print('word onset index:', w)
                pass
        embeddings = torch.concat(all_words,dim=0) # final shape [words, features, time_points] 
        return embeddings.cpu()
    

    def get_embeddings(self, identifier_fn, audio_file_path: str, neural_file_path: str, devices: list): 
        identifier = identifier_fn(self.layers)
        return self.get_embeddings_(self, identifier, audio_file_path, neural_file_path, devices)


class Cochleagram(nn.Module):
    def __init__(self, device):
     # Length of the input audio signal (currently must be fixed, due to filter construction)

        super(Cochleagram, self).__init__()
        self.device = device
        self.sr = 40000 # Sampling rate of the input audio
        self.use_rfft = True
        self.pad_factor= 1.25

        self.env_sr = self.sr # Sampling rate after downsampling
        self.window_size = 1001
        self.p = 50 #(1001 - 1)//2
        ### Define the cochlear filters using ERBCosFilters. 
        # These are the arguments used for filter construction of ERBCosFilters. See helpers/erb_filters.py for 
        # more documentation. 
        self.half_cos_filter_kwargs = {
            'n':50, # Number of filters to evenly tile the space
            'low_lim':50, # Lowest center frequency for full filter (if lowpass filters are used they can be centered lower)
            'high_lim':8000, # Highest center frequency 
            'sample_factor':4, # Positive integer that determines how densely ERB function will be sampled
            'full_filter':False, # Whether to use the full-filter. Must be False if rFFT is true. 
        }
        # These arguments are for the CochFilters class (generic to any filters). 
        self.coch_filter_kwargs = {'use_rfft':self.use_rfft,
                                    'pad_factor':self.pad_factor,
                                    'filter_kwargs':self.half_cos_filter_kwargs}

        self.downsampling_kwargs = {'window_size':self.window_size, 'padding': (self.p,self.p)} # Parameters for the downsampling filter (see downsampling.py)


    def forward(self, x):
        
        if x.shape[-1] % 2 != 0:
            x = F.pad(x, (0, 1), mode='constant', value=0)
            
        signal_size = x.shape[-1]
        # This (and most) cochleagrams use ERBCosFilters, however other types of filterbanks can be 
        # constructed for linear spaced filters or different shapes. Make a new CochlearFilter class for 
        # these. 
        filters = chcochleagram.cochlear_filters.ERBCosFilters(signal_size,
                                                            self.sr, 
                                                            **self.coch_filter_kwargs)
        ### Define an envelope extraction operation
        # Use the analytic amplitude of the hilbert transform here. Other types of envelope extraction 
        # are also implemented in envelope_extraction.py. Can use Identity if want the raw subbands. 
        envelope_extraction = chcochleagram.envelope_extraction.HilbertEnvelopeExtraction(signal_size,
                                                                                        self.sr, 
                                                                                        self.use_rfft, 
                                                                                        self.pad_factor)

        ### Define a downsampling operation
        # Downsample the extracted envelopes. Can use Identity if want the raw subbands. 
        downsampling_op = chcochleagram.downsampling.SincWithKaiserWindow(self.sr, 
                                                                          self.env_sr, 
                                                                          **self.downsampling_kwargs)

        cochleagram = chcochleagram.cochleagram.Cochleagram(filters, 
                                                            envelope_extraction,
                                                            downsampling_op,
                                                            # compression=compression
                                                            )
        y = cochleagram(x)
        return y
        


class Spectrogram(nn.Module):
    def __init__(self, device):
        super(Spectrogram, self).__init__()
        self.sample_rate_orig = 44100
        self.sample_rate_target = 40000
        self.hop_length = 1
        self.window_before, self.window_after = 200, 600
        self.n_fft = 1000
        self.device = device

    def forward(self, x):
        
        spectrogram = T.MelSpectrogram(sample_rate= self.sample_rate_target, 
                                       n_fft = self.n_fft, 
                                       hop_length= self.hop_length)
        y = spectrogram(x)
        return y
        




class WhisperEncoder:
    def __init__(self, model_name, num_layers, time_lag=None):
        self.model_name = model_name
        self.num_layers = num_layers 
        self.time_lag = time_lag

        self.sample_rate_orig = 44100
        # whisper takes 30 sec audio waveforms at 16kHz
        self.sample_rate_target = 16000
        self.whisper_time_limit = 30 
        self.time_after_onset = 1.5 #seconds
        self.time_before_onset = 0.5
        self.word_window = self.time_before_onset + self.time_after_onset

        if 'large' in self.model_name:
            self.hugging_face_iden = f"openai/{self.model_name}"
        else:
            self.hugging_face_iden = f"openai/{self.model_name}.en"
            
        self.processor = WhisperProcessor.from_pretrained(self.hugging_face_iden, cache_dir=CACHE)
        self.model = WhisperModel.from_pretrained(self.hugging_face_iden, cache_dir=CACHE)
            

    def downsample(self, speech_waveform):
        resampler = torchaudio.transforms.Resample(orig_freq=self.sample_rate_orig, 
                                                   new_freq=self.sample_rate_target)
        downsampled_waveform = resampler(speech_waveform)
        return downsampled_waveform
        

    @staticmethod
    def cache_file(identifier: str) -> str:
        path = os.path.join(paths["EMBEDDINGS_PATH"], identifier)  # Ensure CACHE is used
        print(f"Caching path: {path}")
        return path


    @cache(lambda identifier, **kwargs: WhisperEncoder.cache_file(identifier))    
    def get_embeddings_(self, identifier: str, layer:int, audio_file_path: str, neural_file_path: str,):
            
        speech_waveform, _ = torchaudio.load(audio_file_path)
        downsampled_waveform = self.downsample(speech_waveform)
        device = self.model.device

        # Load Whisper processor & model
        epochs = mne.read_epochs(neural_file_path, preload=True, verbose=False)
        all_words = []

        for i in tqdm(range(len(epochs.metadata))):

            word_start = epochs.metadata.loc[i,'Start']
            word_embedding = self.extract_word_embedding(downsampled_waveform, word_start, layer, device)
            all_words.append(word_embedding)
            del word_embedding
            gc.collect()

        all_embeddings = torch.concat(all_words,dim=0).permute(0, 2, 1) # final shape [words, features, time_points] 
        return all_embeddings.cpu()
    


    def get_embeddings(self, identifier_fn, audio_file_path: str, neural_file_path: str, devices: list): 
        
        layers_per_device = [list(range(i, self.num_layers, len(devices))) for i in range(len(devices))]
        print('layers per device',layers_per_device)

        def process_layer_chunk(model_instance, layers, device):
            # Move the LLM model and token IDs to the target device
            model_instance.model = model_instance.model.to(device)
            
            results = {}
            for layer in layers:
                try:
                    embeddings = model_instance.get_embeddings_(
                        identifier = identifier_fn(layer=layer), 
                        layer=layer, 
                        audio_file_path=audio_file_path, 
                        neural_file_path=neural_file_path
                    )
                    results[layer] = embeddings
                except RuntimeError as e:
                    traceback.print_exc()
                    print(f"Error on layer {layer}, device {device}: {e}")
                    torch.cuda.empty_cache()
            return results
        
        # Preload LLM instances for each device
        model_instances_per_device = {
            device: WhisperEncoder(
                model_name=self.model_name, 
                num_layers=self.num_layers, 
            ) for device in devices
        }
        
        # Initialize results dictionary
        all_results = {}
        
        # Perform parallel processing
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = {
                executor.submit(
                    process_layer_chunk, 
                    model_instances_per_device[device],  # Pass LLM instance
                    layers_per_device[device_idx], 
                    device, 

                ): device_idx
                for device_idx, device in enumerate(devices)
            }
        
            for future in futures:
                try:
                    results = future.result()
                    all_results.update(results)
                except Exception as e:
                    traceback.print_exc()
                    print(f"Error in thread execution: {e}")


    def extract_word_embedding(self, waveform, word_start, layer, device):

        forward_window = word_start + self.time_after_onset
        
        if forward_window < self.whisper_time_limit: # if the word is in the first 30 seconds

            end = int(forward_window*self.sample_rate_target)
            audio_data = waveform[1, :end+1] # extract waveform from the beginning of audio up until word end
            inputs = self.processor(audio_data.cpu().numpy(), sampling_rate=self.sample_rate_target, return_tensors="pt")

            # Load your audio array (already resampled to 16kHz)
            audio_length_ms = len(audio_data) / self.sample_rate_target * 1000  # Convert samples to milliseconds
            # Compute number of valid frames
            valid_frames = int(audio_length_ms / 10)
            # Create attention mask (1 for valid frames, 0 for padding)
            attention_mask = torch.zeros((1, 3000), dtype=torch.long)  # Initialize all as padding (this is necessary since whisper takes in 30 s of audio, for less, pad with zeros)
            attention_mask[:, :valid_frames] = 1  # Set real audio frames to 1
            
            # Run encoder to get hidden states
            with torch.no_grad():
                encoder_outputs = self.model.encoder(inputs.input_features.to(device), 
                                                attention_mask=attention_mask.to(device),
                                                output_hidden_states=True)
                    
            # Convert word start and end times to encoder and mel frames
            end_encoder_frame = audio_timestamp_to_encoder_frame(forward_window*1000)
            word_embed = encoder_outputs.hidden_states[layer][:,end_encoder_frame-100:end_encoder_frame+1,:] # 100 is the equivalent of self.word_window in encoder frames

        else:
            start_ish = int(forward_window - self.whisper_time_limit) * self.sample_rate_target # start of 30 second chunk to extract 
            end_ish = int(forward_window) * self.sample_rate_target # end of 30 second chunk to extract
            audio_data = waveform[1, start_ish:end_ish+1].numpy()
            inputs = self.processor(audio_data, sampling_rate=self.sample_rate_target, return_tensors="pt")

            # Run encoder to get hidden states
            with torch.no_grad():
                encoder_outputs = self.model.encoder(inputs.input_features.to(device), 
                                                output_hidden_states=True)
                
            start_idx = self.whisper_time_limit - self.word_window # for frames past 30 s, the word is always at the last 2 seconds
            start_frame = audio_timestamp_to_encoder_frame(start_idx*1000)
            word_embed = encoder_outputs.hidden_states[layer][:,start_frame-1:,:]
        
        del encoder_outputs, waveform, audio_data, inputs
        gc.collect()
        return word_embed



class WhisperDecoder(WhisperEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'large' in self.model_name:
            self.hugging_face_iden = f"openai/{self.model_name}"
            self.generator_model = WhisperForConditionalGeneration.from_pretrained(self.hugging_face_iden, cache_dir=CACHE)

        else:
            self.hugging_face_iden = f"openai/{self.model_name}.en"
            self.generator_model = WhisperForConditionalGeneration.from_pretrained(self.hugging_face_iden, cache_dir=CACHE)
            
    def downsample(self, speech_waveform):
        resampler = torchaudio.transforms.Resample(orig_freq=self.sample_rate_orig, 
                                                   new_freq=self.sample_rate_target)
        downsampled_waveform = resampler(speech_waveform)
        return downsampled_waveform
        
    @staticmethod
    def cache_file(identifier: str) -> str:
        path = os.path.join(paths["EMBEDDINGS_PATH"], identifier)
        print(f"Caching path: {path}")
        return path

    def get_token_ids(self, inputs, device):
        
        input_features = inputs.input_features.to(device)

        self.generator_model = self.generator_model.to(device)
        generation_output = self.generator_model.generate(
            input_features,
            output_hidden_states=False,
            return_dict_in_generate=True,
            language='en'
        )
        # Get the generated token ids and decode them to text.
        generated_ids = generation_output.sequences  # shape: [batch, seq_len]
        # transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_ids

    # def extract_decoder_word_embedding(self, waveform, word_end, layer, device):

    #     if word_end < self.whisper_time_limit:
    #         end = int(word_end * self.sample_rate_target)
    #         audio_data = waveform[1, :end+1]
    #         inputs = self.processor(audio_data.cpu().numpy(), 
    #                         sampling_rate=self.sample_rate_target, 
    #                         return_tensors="pt")

    #         # Get encoder outputs first
    #         audio_length_ms = len(audio_data) / self.sample_rate_target * 1000
    #         valid_frames = int(audio_length_ms / 10)
    #         attention_mask = torch.zeros((1, 3000), dtype=torch.long)
    #         attention_mask[:, :valid_frames] = 1
    #         with torch.no_grad():
    #             encoder_outputs = self.model.encoder(
    #                 inputs.input_features.to(device), 
    #                 attention_mask=attention_mask.to(device),
    #                 output_hidden_states=True
    #             )

    #         # Prepare a simple decoder input (using the start-of-transcript token)
    #         decoder_input_ids = self.get_token_ids(inputs, device)

    #         with torch.no_grad():
    #             decoder_outputs = self.model.decoder(
    #                 input_ids=decoder_input_ids,
    #                 encoder_hidden_states=encoder_outputs.last_hidden_state,
    #                 output_hidden_states=True
    #             )
    #         # Extract hidden states from the specified decoder layer.
    #         word_embed = decoder_outputs.hidden_states[layer][0][-1]  # shape: [batch, seq_len, hidden_size]
    #     else:
    #         start_ish = int((word_end - self.whisper_time_limit) * self.sample_rate_target)
    #         end_ish = int(word_end * self.sample_rate_target)
    #         audio_data = waveform[1, start_ish:end_ish+1].numpy()
    #         inputs = self.processor(audio_data, sampling_rate=self.sample_rate_target, return_tensors="pt")
    #         with torch.no_grad():
    #             encoder_outputs = self.model.encoder(
    #                 inputs.input_features.to(device), 
    #                 output_hidden_states=True
    #             )
    #         decoder_input_ids = self.get_token_ids(inputs, device)
    #         with torch.no_grad():
    #             decoder_outputs = self.model.decoder(
    #                 input_ids=decoder_input_ids,
    #                 encoder_hidden_states=encoder_outputs.last_hidden_state,
    #                 output_hidden_states=True
    #             )
    #         word_embed = decoder_outputs.hidden_states[layer][0][-1]
    #     del encoder_outputs, decoder_outputs, waveform, audio_data, inputs
    #     gc.collect()
    #     return word_embed
    def extract_decoder_word_embeddings_early(self, waveform, word_ends, layer, device):
        """
        Batch process word segments where word_end < self.whisper_time_limit.
        Returns a tensor of shape [batch_size, hidden_size].
        """
        batch_audio_segments = []
        for word_end in word_ends:
            end = int(word_end * self.sample_rate_target)
            # Extract audio from the beginning until the word end.
            audio_segment = waveform[1, :end+1].cpu().numpy()
            batch_audio_segments.append(audio_segment)
        # Process batch with padding.
        inputs = self.processor(batch_audio_segments,
                                sampling_rate=self.sample_rate_target,
                                return_tensors="pt",
                                padding=True)
        # Create custom attention masks for each sample.
        seq_len = inputs.input_features.size(-1)
        batch_attention_masks = []
        for word_end in word_ends:
            end = int(word_end * self.sample_rate_target)
            audio_length_ms = end / self.sample_rate_target * 1000
            valid_frames = int(audio_length_ms / 10)  # assuming 1 frame per 10 ms
            mask = torch.zeros(seq_len, dtype=torch.long)
            mask[:valid_frames] = 1
            batch_attention_masks.append(mask)
        attention_mask = torch.stack(batch_attention_masks, dim=0).to(device)

        inputs.input_features = inputs.input_features.to(device)
        with torch.no_grad():
            encoder_outputs = self.model.encoder(
                inputs.input_features,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        decoder_input_ids = self.get_token_ids(inputs, device)
        with torch.no_grad():
            decoder_outputs = self.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                output_hidden_states=True
            )
        # Extract the last token’s hidden state from the specified layer.
        batch_embeddings = decoder_outputs.hidden_states[layer][:, -1]
        return batch_embeddings


    def extract_decoder_word_embeddings_late(self, waveform, word_ends, layer, device):
        """
        Batch process word segments where word_end >= self.whisper_time_limit.
        Returns a tensor of shape [batch_size, hidden_size].
        """
        batch_audio_segments = []
        for word_end in word_ends:
            start_ish = int((word_end - self.whisper_time_limit) * self.sample_rate_target)
            end_ish = int(word_end * self.sample_rate_target)
            audio_segment = waveform[1, start_ish:end_ish+1].numpy()
            batch_audio_segments.append(audio_segment)
        inputs = self.processor(batch_audio_segments,
                                sampling_rate=self.sample_rate_target,
                                return_tensors="pt",
                                padding=True)
        inputs.input_features = inputs.input_features.to(device)
        with torch.no_grad():
            encoder_outputs = self.model.encoder(
                inputs.input_features,
                output_hidden_states=True
            )
        decoder_input_ids = self.get_token_ids(inputs, device)
        with torch.no_grad():
            decoder_outputs = self.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                output_hidden_states=True
            )
        batch_embeddings = decoder_outputs.hidden_states[layer][:, -1]
        return batch_embeddings


    def extract_decoder_word_embedding(self, waveform, word_ends, layer, device, batch_size=8):
        """
        Processes a mixed list of word_end times by partitioning them into short (word_end < whisper_time_limit)
        and long segments (word_end >= whisper_time_limit), batching each group separately,
        and then reassembling the embeddings in the original order.
        
        Args:
        waveform: the input audio tensor.
        word_ends: list of float values representing word end times.
        layer: the decoder layer from which to extract embeddings.
        device: the device to run the model on.
        batch_size: batch size for processing.
        
        Returns:
        A tensor of shape [num_words, hidden_size] with embeddings in the original word order.
        """
        # Partition indices and values.
        short_indices = []
        short_word_ends = []
        long_indices = []
        long_word_ends = []
        for idx, word_end in enumerate(word_ends):
            if word_end < self.whisper_time_limit:
                short_indices.append(idx)
                short_word_ends.append(word_end)
            else:
                long_indices.append(idx)
                long_word_ends.append(word_end)
        
        embeddings_dict = {}
        
        # Process short segments in batches.
        if short_word_ends:
            for i in range(0, len(short_word_ends), batch_size):
                batch_word_ends = short_word_ends[i:i+batch_size]
                batch_embeddings = self.extract_decoder_word_embeddings_early(waveform, batch_word_ends, layer, device)
                for j, emb in enumerate(batch_embeddings):
                    orig_idx = short_indices[i+j]
                    embeddings_dict[orig_idx] = emb.unsqueeze(0)
        
        # Process long segments in batches.
        if long_word_ends:
            for i in range(0, len(long_word_ends), batch_size):
                batch_word_ends = long_word_ends[i:i+batch_size]
                batch_embeddings = self.extract_decoder_word_embeddings_late(waveform, batch_word_ends, layer, device)
                for j, emb in enumerate(batch_embeddings):
                    orig_idx = long_indices[i+j]
                    embeddings_dict[orig_idx] = emb.unsqueeze(0)
        
        # Reassemble embeddings in the original order.
        all_embeddings = [embeddings_dict[idx] for idx in range(len(word_ends))]
        all_embeddings = torch.cat(all_embeddings, dim=0)
        return all_embeddings


    # @cache(lambda identifier, **kwargs: WhisperDecoder.cache_file(identifier))
    # def get_decoder_embeddings_(self, identifier: str, layer: int, audio_file_path: str, neural_file_path: str):
    #     """
    #     Extracts decoder representations (for a given layer) for each word.
    #     """
    #     print('****************** decoder test ************************')
    #     speech_waveform, _ = torchaudio.load(audio_file_path)
    #     downsampled_waveform = self.downsample(speech_waveform)
    #     device = self.model.device

    #     epochs = mne.read_epochs(neural_file_path, preload=True, verbose=False)
    #     all_words = []
    #     for i in tqdm(range(len(epochs.metadata))):
    #         word_end = epochs.metadata.loc[i, 'End'] # start time for the word
    #         word_embedding = self.extract_decoder_word_embedding(downsampled_waveform, word_end, layer, device)
    #         word_embedding= word_embedding.unsqueeze(dim=0)
    #         all_words.append(word_embedding)
    #         # print(word_embedding.shape)
    #         del word_embedding
    #         gc.collect()
    #         torch.cuda.empty_cache()

    #     all_embeddings = torch.concat(all_words, dim=0)
    #     return all_embeddings.cpu()

    @cache(lambda identifier, **kwargs: WhisperDecoder.cache_file(identifier))
    def get_decoder_embeddings_(self, identifier: str, layer: int, audio_file_path: str, neural_file_path: str):
        """
        Extracts decoder representations (for a given layer) for each word.
        """
        print('****************** decoder test ************************')
        speech_waveform, _ = torchaudio.load(audio_file_path)
        downsampled_waveform = self.downsample(speech_waveform)
        device = self.model.device

        epochs = mne.read_epochs(neural_file_path, preload=True, verbose=False)
        all_words = []
        word_ends = list(epochs.metadata['End'])
        word_embeddings = self.extract_decoder_word_embedding(downsampled_waveform, word_ends, layer, device)

        return word_embeddings.cpu()
    
    def get_decoder_embeddings(self, identifier_fn, audio_file_path: str, neural_file_path: str, devices: list):
        """
        Parallelizes extraction of decoder embeddings across devices.
        """
        layers_per_device = [list(range(i, self.num_layers, len(devices))) for i in range(len(devices))]
        print('Decoder layers per device:', layers_per_device)

        def process_layer_chunk(model_instance, layers, device):
            model_instance.model = model_instance.model.to(device)
            results = {}
            for layer in layers:
                try:
                    embeddings = model_instance.get_decoder_embeddings_(
                        identifier=identifier_fn(layer=layer),
                        layer=layer,
                        audio_file_path=audio_file_path,
                        neural_file_path=neural_file_path
                    )
                    results[layer] = embeddings
                except RuntimeError as e:
                    traceback.print_exc()
                    print(f"Error on decoder layer {layer}, device {device}: {e}")
                    torch.cuda.empty_cache()
            return results

        model_instances_per_device = {
            device: WhisperDecoder(model_name=self.model_name, num_layers=self.num_layers)
            for device in devices
        }
        
        all_results = {}
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = {
                executor.submit(
                    process_layer_chunk,
                    model_instances_per_device[device],
                    layers_per_device[device_idx],
                    device,
                ): device_idx
                for device_idx, device in enumerate(devices)
            }
            for future in futures:
                try:
                    results = future.result()
                    all_results.update(results)
                except Exception as e:
                    traceback.print_exc()
                    print(f"Error in thread execution: {e}")
        return all_results
