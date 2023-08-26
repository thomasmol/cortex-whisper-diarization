from typing import Optional
from pydantic import BaseModel
import base64
import contextlib
import datetime
import numpy as np
import subprocess
import os
import requests
import time
import torch
import wave

from faster_whisper import WhisperModel
from pyannote.audio import Audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering

class Item(BaseModel):
    # Add your input parameters here
    file_string: Optional[str] = None
    file_url: Optional[str] = None
    group_segments: Optional[bool] = True
    num_speakers: Optional[int] = 2
    prompt: Optional[str] = "Some people speaking."
    offset_seconds: Optional[int] = 0


model_name = "large-v2"
model = WhisperModel(model_name, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")
embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def predict(item, run_id, logger):
    item = Item(**item)
    
    if sum([item.file_string is not None, item.file_url is not None]) != 1:
        raise RuntimeError("Provide either file_string or file_url")
   
    try:
        # Generate a temporary filename
        temp_wav_filename = f"temp-{time.time_ns()}.wav"
        if item.file_string is not None:
            audio_data = base64.b64decode(item.file_string.split(',')[1] if ',' in item.file_string else item.file_string)
            temp_audio_filename = f"temp-{time.time_ns()}.audio"
            with open(temp_audio_filename, 'wb') as f:
                f.write(audio_data)

            subprocess.run([
                'ffmpeg',
                '-i', temp_audio_filename,
                '-ar', '16000',
                '-ac', '1',
                '-c:a', 'pcm_s16le',
                temp_wav_filename
            ])

            if os.path.exists(temp_audio_filename):
                os.remove(temp_audio_filename)

        elif item.file_url is not None:
            response = requests.get(item.file_url)
            temp_audio_filename = f"temp-{time.time_ns()}.audio"
            with open(temp_audio_filename, 'wb') as file:
                file.write(response.content)

            subprocess.run([
                'ffmpeg',
                '-i', temp_audio_filename,
                '-ar', '16000',
                '-ac', '1',
                '-c:a', 'pcm_s16le',
                temp_wav_filename
            ])

            if os.path.exists(temp_audio_filename):
                os.remove(temp_audio_filename)
        
        segments = speech_to_text(temp_wav_filename, item.num_speakers, item.prompt,
                                    item.offset_seconds, item.group_segments)

        print(f'done with inference')
        # Return the results as a JSON object
        return {"segments": segments}
    except Exception as e:
        raise RuntimeError("Error Running inference with local model", e)
    finally:
        # Clean up
        if os.path.exists(temp_wav_filename):
            os.remove(temp_wav_filename)


def convert_time(secs, offset_seconds=0):
        return datetime.timedelta(seconds=(round(secs) + offset_seconds))

def speech_to_text(audio_file_wav,
                    num_speakers=2,
                    prompt="People takling.",
                    offset_seconds=0,
                    group_segments=True):
    time_start = time.time()

    # Get duration
    with contextlib.closing(wave.open(audio_file_wav, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    # Transcribe audio
    print("starting whisper")
    options = dict(vad_filter=True,
                    initial_prompt=prompt,
                    word_timestamps=True)
    segments, _ = model.transcribe(audio_file_wav, **options)
    segments = list(segments)
    print("done with whisper")
    segments = [{
        'start':
        int(round(s.start + offset_seconds)),
        'end':
        int(round(s.end + offset_seconds)),
        'text':
        s.text,
        'words': [{
            'start': str(round(w.start + offset_seconds)),
            'end': str(round(w.end + offset_seconds)),
            'word': w.word
        } for w in s.words]
    } for s in segments]

    # Create embedding
    def segment_embedding(segment):
        audio = Audio()
        start = segment["start"]
        # Whisper overshoots the end timestamp in the last segment
        end = min(duration, segment["end"])
        clip = Segment(start, end)
        waveform, sample_rate = audio.crop(audio_file_wav, clip)
        return embedding_model(waveform[None])

    if num_speakers < 2:
        for segment in segments:
            segment['speaker'] = 'Speaker 1'
    else:
        print("starting embedding")
        embeddings = np.zeros(shape=(len(segments), 192))
        for i, segment in enumerate(segments):
            embeddings[i] = segment_embedding(segment)
        embeddings = np.nan_to_num(embeddings)
        print(f'Embedding shape: {embeddings.shape}')

        # Assign speaker label
        clustering = AgglomerativeClustering(num_speakers).fit(
            embeddings)
        labels = clustering.labels_
        for i in range(len(segments)):
            segments[i]["speaker"] = 'Speaker ' + str(labels[i] + 1)

    # Make output
    output = []  # Initialize an empty list for the output

    # Initialize the first group with the first segment
    current_group = {
        'start': str(segments[0]["start"]),
        'end': str(segments[0]["end"]),
        'speaker': segments[0]["speaker"],
        'text': segments[0]["text"],
        'words': segments[0]["words"]
    }

    for i in range(1, len(segments)):
        # Calculate time gap between consecutive segments
        time_gap = segments[i]["start"] - segments[i - 1]["end"]

        # If the current segment's speaker is the same as the previous segment's speaker, and the time gap is less than or equal to 2 seconds, group them
        if segments[i]["speaker"] == segments[
                i - 1]["speaker"] and time_gap <= 2 and group_segments:
            current_group["end"] = str(segments[i]["end"])
            current_group["text"] += " " + segments[i]["text"]
            current_group["words"] += segments[i]["words"]
        else:
            # Add the current_group to the output list
            output.append(current_group)

            # Start a new group with the current segment
            current_group = {
                'start': str(segments[i]["start"]),
                'end': str(segments[i]["end"]),
                'speaker': segments[i]["speaker"],
                'text': segments[i]["text"],
                'words': segments[i]["words"]
            }

    # Add the last group to the output list
    output.append(current_group)

    print("done with embedding")
    time_end = time.time()
    time_diff = time_end - time_start

    system_info = f"""-----Processing time: {time_diff:.5} seconds-----"""
    print(system_info)
    os.remove(audio_file_wav)
    return output