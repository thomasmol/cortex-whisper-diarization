from typing import Optional, Any, List
from pydantic import BaseModel
import base64
import datetime
import subprocess
import os
import requests
import time
import torch
import re

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

class Item(BaseModel):
    # Add your input parameters here
    file_string: Optional[str] = None
    file_url: Optional[str] = None
    group_segments: Optional[bool] = True
    prompt: Optional[str] = ""
    num_speakers: Optional[int] = None
    language: Optional[str] = None
    offset_seconds: Optional[int] = 0


model_name = "large-v3"
model = WhisperModel(
            model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16")
diarization_model = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="").to(
                torch.device("cuda"))

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
        
        segments, detected_num_speakers, detected_language = speech_to_text(temp_wav_filename, item.num_speakers,
                                           item.language, item.prompt, item.offset_seconds,
                                           item.group_segments, word_timestamps=True)

        print('done with inference')
        # Return the results as a JSON object
        return {"segments": segments, "num_speakers": detected_num_speakers, "language": detected_language}
    except Exception as e:
        raise RuntimeError("Error Running inference with local model", e)
    finally:
        # Clean up
        if os.path.exists(temp_wav_filename):
            os.remove(temp_wav_filename)


def convert_time(secs, offset_seconds=0):
        return datetime.timedelta(seconds=(round(secs) + offset_seconds))

def speech_to_text(audio_file_wav,
                    num_speakers=None,
                    language=None,
                    prompt="",
                    offset_seconds=0,
                    group_segments=True,
                    word_timestamps=True,):
    time_start = time.time()
    # Transcribe audio
    print("Starting transcribing")
    options = dict(vad_filter=True,
                    initial_prompt=prompt,
                    word_timestamps=word_timestamps,
                    language=language)
    segments, transcript_info = model.transcribe(audio_file_wav, **options)
    segments = list(segments)
    segments = [{
        'start':
        float(s.start + offset_seconds),
        'end':
        float(s.end + offset_seconds),
        'text':
        s.text,
        'words': [{
            'start': float(w.start + offset_seconds),
            'end': float(w.end + offset_seconds),
            'word': w.word
        } for w in s.words]
    } for s in segments]

    time_transcribing_end = time.time()
    print(
        f"Finished with transcribing, took {time_transcribing_end - time_start:.5} seconds"
    )

    diarization = diarization_model(audio_file_wav,
                                            num_speakers=num_speakers)

    time_diraization_end = time.time()
    print(
        f"Finished with diarization, took {time_diraization_end - time_transcribing_end:.5} seconds"
    )

    # Initialize variables to keep track of the current position in both lists
    margin = 0.1  # 0.1 seconds margin

    # Initialize an empty list to hold the final segments with speaker info
    final_segments = []

    diarization_list = list(diarization.itertracks(yield_label=True))
    unique_speakers = {speaker for _, _, speaker in diarization.itertracks(yield_label=True)}
    detected_num_speakers = len(unique_speakers)

    speaker_idx = 0
    n_speakers = len(diarization_list)

    # Iterate over each segment
    for segment in segments:
        segment_start = segment['start'] + offset_seconds
        segment_end = segment['end'] + offset_seconds
        segment_text = []
        segment_words = []

        # Iterate over each word in the segment
        for word in segment['words']:
            word_start = word['start'] + offset_seconds - margin
            word_end = word['end'] + offset_seconds + margin

            while speaker_idx < n_speakers:
                turn, _, speaker = diarization_list[speaker_idx]

                if turn.start <= word_end and turn.end >= word_start:
                    # Add word without modifications
                    segment_text.append(word['word'])
                    
                    # Strip here for individual word storage
                    word['word'] = word['word'].strip()
                    segment_words.append(word)

                    if turn.end <= word_end:
                        speaker_idx += 1

                    break
                elif turn.end < word_start:
                    speaker_idx += 1
                else:
                    break

        if segment_text:
            combined_text = ''.join(segment_text)
            cleaned_text = re.sub('  ', ' ', combined_text).strip()
            new_segment = {
                'start': segment_start - offset_seconds,
                'end': segment_end - offset_seconds,
                'speaker': speaker,
                'text': cleaned_text,
                'words': segment_words
            }
            final_segments.append(new_segment)

    time_merging_end = time.time()
    print(
        f"Finished with merging, took {time_merging_end - time_diraization_end:.5} seconds"
    )
    segments = final_segments
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

    time_cleaning_end = time.time()
    print(
        f"Finished with cleaning, took {time_cleaning_end - time_merging_end:.5} seconds"
    )
    time_end = time.time()
    time_diff = time_end - time_start

    system_info = f"""Processing time: {time_diff:.5} seconds"""
    print(system_info)
    return output, detected_num_speakers, transcript_info.language