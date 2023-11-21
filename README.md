# Run Whisper-jax on large audio files



This is work in progress

Transcribe a single audio file with NB Whisper:
```
python run_whisper_jax.py NbAiLab/nb-whisper-large-beta /path/to/transcription/dir -a /path/to/audio/file
```

Transcribe all mp3 files in a directory:
```
python run_whisper_jax.py NbAiLab/nb-whisper-large-beta /path/to/transcription/dir -m -d /path/to/audio/file -f mp3 
```

By default, the script produces two files per transcription, a jsonl file with timecoded transcriptions and a txt file
with only the transcription text, but this can be controled with the `-r` flag. Transcription files have the same file
stem as audio files, unless a file stem is passed with the `-s` flag (only works with single audio file transcription).