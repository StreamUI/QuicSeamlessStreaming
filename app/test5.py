import asyncio
import logging
import numpy as np
import torch
from pydub import AudioSegment
from scipy.io.wavfile import write as wav_write

from simuleval.data.segments import SpeechSegment
from seamless_communication.streaming.agents.seamless_streaming_s2st import SeamlessStreamingS2STJointVADAgent
from simuleval import options
from simuleval.utils.arguments import cli_argument_list

# Constants and Logger Setup
SAMPLE_RATE = 16000
TGT_LANG = "eng"  # Target language for translation
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("audio_transcoder")


class OutputSegments:
    """Manages the output segments."""
    def __init__(self, segments):
        if isinstance(segments, SpeechSegment):
            segments = [segments]
        self.segments = segments

    @property
    def is_empty(self):
        return all(segment.is_empty for segment in self.segments)

    @property
    def finished(self):
        return all(segment.finished for segment in self.segments)


# def build_streaming_system(model_configs):
#     """Initialize the Seamless model."""
#     agent = SeamlessStreamingS2STJointVADAgent.from_args(model_configs)
#     return agent

def build_streaming_system(model_configs):
    parser = options.general_parser()
    parser.add_argument("-f", "--f", help="a dummy argument to fool ipython", default="1")
    SeamlessStreamingS2STJointVADAgent.add_args(parser)
    args, _ = parser.parse_known_args(cli_argument_list(model_configs))
    system = SeamlessStreamingS2STJointVADAgent.from_args(args)
    return system


class Transcoder:
    """Transcoder class to handle model inference and audio processing."""
    def __init__(self, model_configs):
        self.agent = build_streaming_system(model_configs)
        device = torch.device(model_configs.get("device", "cpu"))
        dtype = torch.float16 if model_configs.get("dtype") == "fp16" else torch.float32
        self.agent.to(device, dtype=dtype)
        self.sample_rate = SAMPLE_RATE
        self.states = self.agent.build_states()
        self.lock = asyncio.Lock()

    def process_audio_file(self, file_path, output_file):
        """Process the audio file and save the translated audio."""
        audio = AudioSegment.from_file(file_path).set_channels(1).set_frame_rate(SAMPLE_RATE).set_sample_width(2)
        audio_data = np.frombuffer(audio.raw_data, dtype=np.int16).astype(np.float32) / 32768.0

        print(f"Audio data length: {len(audio_data)}")

        input_segment = SpeechSegment(content=audio_data, sample_rate=self.sample_rate, tgt_lang=TGT_LANG, finished=True)

        with torch.no_grad():
            output = self.agent.pushpop(input_segment, self.states)

        print(f"Output length: {len(output)}")

        output_segments = OutputSegments(output)
        logger.info(f"Output segments: {output_segments}")
        if not output_segments.is_empty:
            logger.info(f"Output segments: {output_segments}")
            translated_audio = []
            for segment in output_segments.segments:
                if isinstance(segment, SpeechSegment):
                    logger.info(f"Translated audio length: {len(segment.content)}")
                    # Collect the translated speech data
                    translated_audio.append(segment.content)

            # Combine all segments into a single array
            if translated_audio:
                logger.info(f"Translated audio length: {len(translated_audio)}")
                translated_audio_np = np.concatenate(translated_audio)
                # Convert normalized float32 audio to int16 format for saving as WAV
                translated_audio_int16 = (translated_audio_np * 32768).astype(np.int16)
                
                # Save the translated audio as a .wav file
                wav_write(output_file, self.sample_rate, translated_audio_int16)
                logger.info(f"Translated audio saved to {output_file}")


async def main(file_path, output_file):
    model_configs = {
            'monotonic_decoder_model_name': 'seamless_streaming_monotonic_decoder',
            'unity_model_name': 'seamless_streaming_unity',
            'sentencepiece_model': 'spm_256k_nllb100.model',
            'task': 's2st',
            'tgt_lang': TGT_LANG,
            'min_unit_chunk_size': 50,
            'decision_threshold': 0.7,
            'no_early_stop': True,
            'block_ngrams': True,
            'vocoder_name': 'vocoder_v2',
            'wav2vec_yaml': 'wav2vec.yaml',
            'min_starting_wait_w2vbert': 192,
            'config_yaml': 'cfg_fbank_u2t.yaml',
            'upstream_idx': 1,
            'detokenize_only': True,
            'device': 'cuda:0',
            'dtype': 'fp32',
            'max_len_a': 0,
            'max_len_b': 1000,
    }

    transcoder = Transcoder(model_configs)
    transcoder.process_audio_file(file_path, output_file)


if __name__ == "__main__":
    file_path = "spanish-long.wav"  # Path to your audio file
    output_file = "./outs/translated_audio.wav"  # Output translated audio file

    try:
        asyncio.run(main(file_path, output_file))
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
