import wave  
import os
import re

import numpy as np  
from whisper_online import *  
from pydub import AudioSegment  
from googletrans import Translator

audio_path = "audio/test_fixed.wav"  # 使用轉換後的音頻文件  
src_lan = "en"  # source language  
tgt_lan = "en"  # target language 


# 检查文件是否存在  
if not os.path.isfile(audio_path):  
    print("File not found:", audio_path)  
    exit(1)  

# 檢查文件格式  
def get_audio_format(file_path):  
    try:  
        with wave.open(file_path, 'rb') as wf:  
            return 'wav'  
    except wave.Error:  
        return 'unknown'  

audio_format = get_audio_format(audio_path)  

# 如果格式不是 WAV，則進行轉換  
if audio_format != 'wav':  
    audio = AudioSegment.from_file(audio_path)  
    audio_path_wav = audio_path.replace('.wav', '_converted.wav')  
    audio.export(audio_path_wav, format='wav')  
    audio_path = audio_path_wav  

translator = Translator()
total_inference_time = 0

s = time.time()
# Initialize the ASR model  
asr = FasterWhisperASR(src_lan, "large-v2")  
# asr.set_translate_task()  # it will translate from source language into target language  
print("[Loading large-v2 models]")
# Create the online ASR processor  
online = OnlineASRProcessor(asr)  
e = time.time()
load_model_time = e-s
  
# Open the WAV file  
with wave.open(audio_path, 'rb') as wf:  
    # Get audio parameters  
    n_channels = wf.getnchannels()  
    sample_width = wf.getsampwidth()  
    frame_rate = wf.getframerate()  
    n_frames = wf.getnframes()  
  
    # Calculate the chunk size (e.g., 1 second per chunk)  
    chunk_duration = 1  # chunk duration in seconds  
    chunk_size = int(frame_rate * chunk_duration)  
    buffer = ''
    print("[START TRANSLATE]")
    # Processing loop  
    while True:  
        audio_chunk = wf.readframes(chunk_size)  
        if not audio_chunk:  
            break  
  
        # Convert audio chunk to numpy array  
        audio_chunk_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0  
        if n_channels > 1:  
            audio_chunk_np = audio_chunk_np.reshape(-1, n_channels)  
            audio_chunk_np = audio_chunk_np.mean(axis=1)  # Convert to mono by averaging channels  
        s = time.time()
        online.insert_audio_chunk(audio_chunk_np)  
        o = online.process_iter()  
        a,b = online.prompt()
        e = time.time()
        if o[-1] != '':
            buffer += o[-1]
        if buffer != '':
            print(f"[{e-s:.3f}]", translator.translate(buffer, dest='zh-TW').text)
        total_inference_time += (e-s)

        # print(o, "time: ", e-s)  # do something with current partial output  
  
# At the end of this audio processing  
o = online.finish() 
print("="*100) 
if o[-1] != '':
    buffer += o[-1]
if buffer != '':
    print(f"[{e-s:.3f}]", translator.translate(buffer, dest='zh-TW').text)
print("="*100) 
print("Loading time: ", load_model_time)
print("Total inference time: ", total_inference_time)
print(buffer)

# Refresh if you're going to re-use the object for the next audio  
online.init()  