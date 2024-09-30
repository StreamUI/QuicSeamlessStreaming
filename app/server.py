import asyncio
import logging
import numpy as np
import torch
from typing import Optional, Dict, Union, List

from qh3.asyncio import serve
from qh3.asyncio.protocol import QuicConnectionProtocol
from qh3.quic.configuration import QuicConfiguration
from qh3.quic.events import QuicEvent, StreamDataReceived, ConnectionTerminated, HandshakeCompleted
from qh3.tls import SessionTicket
from simuleval.data.segments import SpeechSegment, TextSegment, Segment, EmptySegment
from simuleval import options
from simuleval.utils.arguments import cli_argument_list
from simuleval.agents.pipeline import TreeAgentPipeline
from seamless_communication.streaming.agents.seamless_streaming_s2st import SeamlessStreamingS2STJointVADAgent

# Constants and Logger Setup
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
# TGT_LANG = "spa"
TGT_LANG = "eng"
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("server")

# M4T_P0_LANGS = [
#     "eng",
#     "arb", "ben", "cat", "ces", "cmn", "cym", "dan",
#     "deu", "est", "fin", "fra", "hin", "ind", "ita",
#     "jpn", "kor", "mlt", "nld", "pes", "pol", "por",
#     "ron", "rus", "slk", "spa", "swe", "swh", "tel",
#     "tgl", "tha", "tur", "ukr", "urd", "uzn", "vie",
# ]


class SessionTicketStore:
    """Session Ticket Store for QUIC TLS session resumption."""
    def __init__(self):
        self.tickets: Dict[bytes, SessionTicket] = {}

    def add(self, ticket: SessionTicket) -> None:
        self.tickets[ticket.ticket] = ticket

    def pop(self, label: bytes) -> Optional[SessionTicket]:
        return self.tickets.pop(label, None)


class OutputSegments:
    """Manages the output segments."""
    def __init__(self, segments: Union[List[Segment], Segment]):
        if isinstance(segments, Segment):
            segments = [segments]
        self.segments = segments

    @property
    def is_empty(self):
        return all(segment.is_empty for segment in self.segments)

    @property
    def finished(self):
        return all(segment.finished for segment in self.segments)


def build_streaming_system(model_configs, agent_class):
    parser = options.general_parser()
    parser.add_argument("-f", "--f", help="a dummy argument to fool ipython", default="1")
    agent_class.add_args(parser)
    args, _ = parser.parse_known_args(cli_argument_list(model_configs))
    system = agent_class.from_args(args)
    return system


class Transcoder:
    """Transcoder class to handle model inference and audio processing."""
    def __init__(self, model_configs):
        self.agent = build_streaming_system(model_configs, SeamlessStreamingS2STJointVADAgent)
        device = torch.device(model_configs.get("device", "cpu"))
        dtype = torch.float16 if model_configs.get("dtype") == "fp16" else torch.float32
        self.agent.to(device, dtype=dtype)
        self.sample_rate = SAMPLE_RATE
        self.states = self.agent.build_states()
        self.lock = asyncio.Lock()

        # Separate queues for different processing stages
        self.raw_input_queue = asyncio.Queue()
        self.preprocessed_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()
        self.end_stream = False

    async def run(self):
        """Main transcoder loop to process incoming audio."""
        tasks = [
            asyncio.create_task(self.ingest_audio()),
            asyncio.create_task(self.process_audio()),
            # asyncio.create_task(self.handle_output()),
        ]
        await asyncio.gather(*tasks)

    async def ingest_audio(self):
        """Handle raw audio input."""
        while not self.end_stream or not self.raw_input_queue.empty():
            try:
                audio_bytes, end_stream = await asyncio.wait_for(self.raw_input_queue.get(), timeout=1.0)
                # logger.info("[ingest_audio] Got input")
                self.raw_input_queue.task_done()
                # Preprocess audio bytes into SpeechSegment
                audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                input_segment = SpeechSegment(
                    content=audio_np,
                    sample_rate=self.sample_rate,
                    tgt_lang=TGT_LANG,
                    finished=end_stream,
                )
                await self.preprocessed_queue.put(input_segment)
                if end_stream:
                    logger.info("[ingest_audio] End of stream received")
                    self.end_stream = True
            except asyncio.TimeoutError:
                if self.end_stream and self.raw_input_queue.empty():
                    logger.info("[ingest_audio] No more input, ending ingest_audio.")
                    break

    async def process_audio(self):
        """Process preprocessed audio segments."""
        while not self.end_stream or not self.preprocessed_queue.empty():
            try:
                input_segment = await asyncio.wait_for(self.preprocessed_queue.get(), timeout=1.0)
                # logger.info("[process_audio] Processing input segment")
                self.preprocessed_queue.task_done()
                async with self.lock:
                    with torch.no_grad():
                        output = await asyncio.get_event_loop().run_in_executor(
                            None, self.agent.pushpop, input_segment, self.states
                        )
                output_segments = OutputSegments(output)
                # logger.info("[process_audio] Processed audio segments: %d, is empty: %s",
                #              len(output_segments.segments), output_segments.is_empty)
                if not output_segments.is_empty:
                    await self.output_queue.put(output_segments)
                if output_segments.finished:
                    logger.info("[process_audio] Finished processing, resetting states")
                    self.reset_states()
            except asyncio.TimeoutError:
                if self.end_stream and self.preprocessed_queue.empty():
                    logger.info("[process_audio] No more preprocessed data, ending process_audio.")
                    break
            except Exception as e:
                logger.exception(f"[process_audio] Exception: {e}")

    def reset_states(self):
        """Reset model states after finishing a translation."""
        logger.info("[reset_states] Resetting states")
        if isinstance(self.agent, TreeAgentPipeline):
            states_iter = self.states.values()
        else:
            states_iter = self.states
        for state in states_iter:
            state.reset()




class QuicStreamingAudioServerProtocol(QuicConnectionProtocol):
    """Handles QUIC streaming protocol events."""
    transcoder: Optional[Transcoder] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream_id = None
        self.audio_buffer = bytearray()
        self.end_of_stream = False
        self.transcoder_task = None
        self.sender_task = None

        # Initialize the transcoder if not already done
        if QuicStreamingAudioServerProtocol.transcoder is None:
            model_configs = self.get_model_configs()
            QuicStreamingAudioServerProtocol.transcoder = Transcoder(model_configs)

    def connection_made(self, transport):
        """Called when a connection is established."""
        super().connection_made(transport)
        logger.info("Connection made.")
        self.transcoder_task = asyncio.create_task(self.transcoder.run())
        self.sender_task = asyncio.create_task(self.send_output())

    def quic_event_received(self, event: QuicEvent) -> None:
        """Handle QUIC events."""
        if isinstance(event, StreamDataReceived):
            self.handle_stream_data(event)
        elif isinstance(event, ConnectionTerminated):
            self.handle_connection_terminated()

    def handle_stream_data(self, event: StreamDataReceived):
        """Processes incoming stream data."""
        self.stream_id = event.stream_id
        self.audio_buffer.extend(event.data)

        # Process complete audio chunks
        remainder = len(self.audio_buffer) % BYTES_PER_SAMPLE
        audio_bytes = self.audio_buffer[:-remainder] if remainder else self.audio_buffer
        logger.info(f"Audio buffer length after extending: {len(audio_bytes)}, remainder: {remainder}")
        self.audio_buffer = self.audio_buffer[-remainder:] if remainder else bytearray()
        asyncio.create_task(self.transcoder.raw_input_queue.put((audio_bytes, self.end_of_stream)))

    def handle_connection_terminated(self):
        """Handle connection termination and clean up."""
        logger.info(f"Connection terminated")
        self.cancel_tasks()

    async def send_output(self):
        """Sends output audio to the client."""
        try:
            while True:
                try:
                    output_segments = await asyncio.wait_for(self.transcoder.output_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    if self.transcoder.end_stream and self.transcoder.output_queue.empty():
                        logger.info("Ending send_output as end_stream is reached and output queue is empty.")
                        break
                    continue

                newStreamId = self._quic.get_next_available_stream_id(is_unidirectional=False)
                for segment in output_segments.segments:
                    if isinstance(segment, SpeechSegment):
                        translated_bytes = (np.array(segment.content) * 32768).astype(np.int16).tobytes()
                        logger.info("[send_output] Got speech segment, length: %d, stream_id: %d", len(translated_bytes), newStreamId)
                        self._quic.send_stream_data(newStreamId, translated_bytes, end_stream=False)
                        self.transmit()
                    elif isinstance(segment, TextSegment):
                        logger.info(f"[send_output] Got text segment: {segment.content}")
                    else:
                        pass
                        # logger.info("[send_output] Not a speech segment")
                self.transcoder.output_queue.task_done()

        except asyncio.CancelledError:
            logger.info("[send_output] Sender task cancelled")
        except Exception as e:
            logger.exception(f"[send_output] Error: {e}")

    def cancel_tasks(self):
        """Cancels running tasks."""
        logger.info("Cancelling tasks")
        if self.transcoder_task:
            self.transcoder_task.cancel()
        if self.sender_task:
            self.sender_task.cancel()
        self.end_of_stream = True

    @staticmethod
    def get_model_configs():
        """Returns the model configuration."""
        return {
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
            'dtype': 'fp16',
            'max_len_a': 0,
            'max_len_b': 1000,
        }


async def main(host: str, port: int, configuration: QuicConfiguration):
    await serve(
        host, port,
        configuration=configuration,
        create_protocol=lambda *args, **kwargs: QuicStreamingAudioServerProtocol(*args, **kwargs),
        session_ticket_fetcher=SessionTicketStore().pop,
        session_ticket_handler=SessionTicketStore().add,
    )
    await asyncio.Future()


if __name__ == "__main__":
    configuration = QuicConfiguration(is_client=False)
    configuration.load_cert_chain("cert.pem", "key.pem")

    host = "0.0.0.0"
    port = 4433

    try:
        logger.info("Starting QUIC audio translation server")
        asyncio.run(main(host, port, configuration))
    except KeyboardInterrupt:
        logger.info("Server shutdown by user.")
    except Exception as e:
        logger.exception(f"Error encountered: {e}")
