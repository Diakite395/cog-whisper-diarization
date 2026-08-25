import base64
import logging
import re
import subprocess
import tempfile
import time
from pathlib import Path as LocalPath
from typing import Optional

import numpy as np
import pandas as pd
import requests
import torch
import torchaudio
from cog import BaseModel, BaseRunner, Input, Path
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions
from pyannote.audio import Pipeline

WHISPER_MODEL_PATH = "/models/whisper/large-v3-turbo"
DIARIZATION_MODEL_PATH = "/models/diarization/pyannote--speaker-diarization-community-1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Output(BaseModel):
    segments: list
    language: Optional[str] = None
    num_speakers: Optional[int] = None


class Runner(BaseRunner):
    def setup(self) -> None:
        logger.info("Loading Whisper model from %s", WHISPER_MODEL_PATH)
        self.model = WhisperModel(
            model_size_or_path=WHISPER_MODEL_PATH,
            device="cuda",
            compute_type="float16",
        )
        logger.info("Whisper model loaded")
        logger.info("Loading diarization model from %s", DIARIZATION_MODEL_PATH)
        self.diarization_model = Pipeline.from_pretrained(DIARIZATION_MODEL_PATH).to(
            torch.device("cuda")
        )
        logger.info("Diarization model loaded")

    def run(
        self,
        file_string: Optional[str] = Input(
            description="Either provide: Base64 encoded audio file,", default=None
        ),
        file_url: Optional[str] = Input(
            description="Or provide: A direct audio file URL", default=None
        ),
        file: Optional[Path] = Input(description="Or an audio file", default=None),
        num_speakers: Optional[int] = Input(
            description="Number of speakers, leave empty to autodetect.",
            ge=1,
            le=50,
            default=None,
        ),
        translate: bool = Input(
            description="Translate the speech into English.",
            default=False,
        ),
        language: Optional[str] = Input(
            description="Language of the spoken words as a language code like 'en'. Leave empty to auto detect language.",
            default=None,
        ),
        prompt: Optional[str] = Input(
            description="Vocabulary: provide names, acronyms and loanwords in a list. Use punctuation for best accuracy.",
            default=None,
        ),
    ) -> Output:
        if file_string == "":
            file_string = None
        if file_url == "":
            file_url = None

        inputs = [file_string is not None, file_url is not None, file is not None]
        if sum(inputs) != 1:
            raise RuntimeError("Provide exactly one of file_string, file_url, or file")

        start_time = time.time()
        with tempfile.TemporaryDirectory() as directory:
            temp_dir = LocalPath(directory)
            source_path = temp_dir / "source.audio"
            wav_path = temp_dir / "audio.wav"

            if file is not None:
                source_path = LocalPath(file)
            if file_url is not None:
                download_file(file_url, source_path)
            if file_string is not None:
                write_base64_file(file_string, source_path)

            log_audio_metadata(source_path)

            normalize_start_time = time.time()
            audio_duration = normalize_audio(source_path, wav_path)
            logger.info("Audio normalized in %.2fs", time.time() - normalize_start_time)
            logger.info("Audio duration: %.2fs", audio_duration)

            segments, detected_language = self.speech_to_text(
                str(wav_path),
                language,
                translate=translate,
            )
            logger.info("Run completed in %.2fs", time.time() - start_time)
            return Output(
                segments=segments,
                language=detected_language,
            )

    def speech_to_text(
        self,
        audio_file_wav: str,
        num_speakers: Optional[int] = None,
        prompt: Optional[str] = None,
        language: Optional[str] = None,
        translate: bool = False,
    ) -> tuple[list[dict[str, object]], int, str]:
        start_time = time.time()
        gpu_type = get_gpu_type()
        logger.info("GPU type: %s", gpu_type)
        logger.info("Starting transcription")

        options = {
            "language": language,
            "beam_size": 2,
            "word_timestamps":True,
            "condition_on_previous_text": False,
            "log_prob_threshold": -1.0,
            "hallucination_silence_threshold": 2.0,
            "no_speech_threshold": 0.6,
            "no_repeat_ngram_size": 4,
        }

        segments, transcript_info = self.model.transcribe(audio_file_wav, **options)
        transcription = format_transcription_segments(list(segments))
        transcribe_end_time = time.time()
        logger.info(
            "Transcription completed in %.2fs. Detected language: %s. Segments: %s",
            transcribe_end_time - start_time,
            transcript_info.language,
            len(transcription),
        )

        return segments, transcript_info.language


def download_file(url: str, path: LocalPath) -> None:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)


def write_base64_file(file_string: str, path: LocalPath) -> None:
    encoded = file_string.split(",", 1)[1] if "," in file_string else file_string
    path.write_bytes(base64.b64decode(encoded))


def normalize_audio(input_path: LocalPath, wav_path: LocalPath) -> float:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-map",
            "0:a:0",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(wav_path),
        ],
        check=True,
    )
    return get_duration(wav_path)


def get_duration(path: LocalPath) -> float:
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    return float(output.decode().strip())


def log_audio_metadata(path: LocalPath) -> None:
    try:
        metadata = torchaudio.info(str(path))
        logger.info(
            "Audio metadata: sample_rate=%s, num_channels=%s, num_frames=%s, bits_per_sample=%s, encoding=%s",
            metadata.sample_rate,
            metadata.num_channels,
            metadata.num_frames,
            metadata.bits_per_sample,
            metadata.encoding,
        )
    except Exception as error:
        logger.info("Torchaudio could not read file metadata: %s", error)


def format_transcription_segments(segments: list[object]) -> list[dict[str, object]]:
    output_segments = []
    for segment in segments:
        output_segment = {
            "avg_logprob": segment.avg_logprob,
            "start": float(segment.start),
            "end": float(segment.end),
            "words": [],
        }
        if segment.words is not None:
            output_segment["words"] = [
                {
                    "start": float(word.start),
                    "end": float(word.end),
                    "word": word.word,
                    "probability": word.probability,
                }
                for word in segment.words
            ]
        output_segments.append(output_segment)
    return output_segments


def post_process_segments(
    segments: list[dict[str, object]]
) -> list[dict[str, object]]:
    return 


def get_gpu_type() -> str:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"]
        )
        return output.decode().strip()
    except Exception:
        return "Unknown"
