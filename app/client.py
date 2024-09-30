import asyncio
import logging
import ssl
from typing import cast
from pydub import AudioSegment

from qh3.asyncio.client import connect
from qh3.asyncio.protocol import QuicConnectionProtocol
from qh3.quic.configuration import QuicConfiguration
from qh3.quic.events import QuicEvent, StreamDataReceived
import pyaudio
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("client")

CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1  # 1 channel (mono audio)
RATE = 16000

# Buffer settings
# BUFFER_DURATION = 0.2  # Buffer duration in seconds
BUFFER_DURATION = 1.0 # Buffer duration in seconds
TARGET_BUFFER_SIZE = int(RATE * BUFFER_DURATION * 2)  # 2 bytes per sample for paInt16

class QuicAudioClientProtocol(QuicConnectionProtocol):
    def __init__(self, *args, output_queue=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_rtt_task = asyncio.create_task(self._log_rtt())
        self.stream_id = None  # This will hold the same stream_id for the session
        self.output_queue = output_queue

    async def _log_rtt(self):
        while True:
            stats = self._quic.get_stats()
            rtt = stats.smoothed_rtt  # In seconds
            logger.info(f"Current RTT: {rtt*1000:.2f} ms")
            await asyncio.sleep(5)  # Log every 5 seconds

    def connection_made(self, transport):
        super().connection_made(transport)
        # Open a single bidirectional stream
        self.stream_id = self._quic.get_next_available_stream_id(is_unidirectional=False)
        if self.stream_id is not None:
            logger.info(f"Opening stream {self.stream_id} for continuous audio streaming")
        else:
            logger.error("Failed to obtain a valid stream ID for audio streaming")

    def connection_lost(self, exc):
        if hasattr(self, '_log_rtt_task'):
            self._log_rtt_task.cancel()
        super().connection_lost(exc)

    def quic_event_received(self, event: QuicEvent) -> None:
        if isinstance(event, StreamDataReceived):
            stream_id = event.stream_id
            logger.debug(f"Received {len(event.data)} bytes from the server on stream {stream_id}")

            # Process the data immediately
            if self.output_queue:
                self.output_queue.put_nowait(event.data)

    async def send_audio(self, audio_data: bytes, end_of_stream: bool):
        if self.stream_id is None:
            raise ValueError("Stream ID is not initialized")

        # logger.debug(f"Sending {len(audio_data)} bytes to the server on stream {self.stream_id}")

        self._quic.send_stream_data(self.stream_id, audio_data, end_stream=False)
        self.transmit()


async def audio_file_reader(file_path, input_queue):
    loop = asyncio.get_event_loop()

    try:
        # Load the audio file using pydub
        audio = AudioSegment.from_file(file_path)

        # Convert to mono and match the target sample rate
        audio = audio.set_channels(1).set_frame_rate(RATE)

        # Get raw data and ensure it's in the correct format (int16)
        audio = audio.set_sample_width(2)  # 2 bytes per sample for 16-bit (paInt16)
        audio_data = audio.raw_data

        # Stream in 1024 byte chunks
        for i in range(0, len(audio_data), CHUNK_SIZE):
            chunk = audio_data[i:i + CHUNK_SIZE]
            await input_queue.put((chunk, False))

            # Simulate real-time streaming based on chunk duration
            await asyncio.sleep(CHUNK_SIZE / RATE)
        await input_queue.put((b'', True))
    except Exception as e:
        logger.error(f"Audio file reader encountered an error: {e}")



async def network_sender(client, input_queue):
    while True:
        audio_data, end_of_stream = await input_queue.get()
        logger.debug(f"Sending audio data: {len(audio_data)}")
        await client.send_audio(audio_data, end_of_stream)
        # if end_of_stream:
        #     logger.info("End of stream")
        #     break


async def audio_player(playback_stream, output_queue):
    loop = asyncio.get_event_loop()
    buffer = bytearray()
    try:
        while True:
            audio_data = await output_queue.get()
            buffer.extend(audio_data)
            # Play audio when buffer is large enough
            if len(buffer) >= TARGET_BUFFER_SIZE:
                await loop.run_in_executor(None, playback_stream.write, bytes(buffer))
                buffer.clear()
    except Exception as e:
        logger.error(f"Audio player encountered an error: {e}")


async def stream_audio(configuration: QuicConfiguration, host: str, port: int, file_path: str) -> None:
    audio_interface = pyaudio.PyAudio()

    # Open a stream for output (playback)
    playback_stream = audio_interface.open(format=FORMAT, channels=CHANNELS,
                                           rate=RATE, output=True)

    input_queue = asyncio.Queue()
    output_queue = asyncio.Queue()

    async with connect(
        host, port, configuration=configuration, create_protocol=lambda *args, **kwargs: QuicAudioClientProtocol(*args, output_queue=output_queue, **kwargs)
    ) as client:
        client = cast(QuicAudioClientProtocol, client)
        logger.info("Connected to %s:%d", host, port)

        try:
            reader_task = asyncio.create_task(audio_file_reader(file_path, input_queue))
            sender_task = asyncio.create_task(network_sender(client, input_queue))
            player_task = asyncio.create_task(audio_player(playback_stream, output_queue))

            await asyncio.gather(reader_task, sender_task, player_task)
        except KeyboardInterrupt:
            pass
        finally:
            # Clean up streams and terminate the audio interface
            try:
                if playback_stream.is_active():
                    playback_stream.stop_stream()
                playback_stream.close()
            except Exception as e:
                logger.error(f"Error closing audio output stream: {e}")

            audio_interface.terminate()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    configuration = QuicConfiguration(is_client=True)
    configuration.verify_mode = ssl.CERT_NONE

    host = "IP_ADDRESS"
    port = 4433
    file_path = "spanish.wav"  # Path to your local audio file
    # file_path = "audio.wav"  # Path to your local audio file

    try:
        asyncio.run(stream_audio(configuration=configuration, host=host, port=port, file_path=file_path))
    except KeyboardInterrupt:
        pass
