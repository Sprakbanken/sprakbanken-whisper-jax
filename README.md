# Run Whisper-jax on large audio files
Transcribe audio files with Whisper Jax

## Install
* Make pyenv virtualenv
`pyenv virtualenv 3.11.2 myenv`
* Activate virtualenv
`pyenv shell myenv`
* Install package (assuming you are in the root dir of this package)
`pip install .`

## Basic usage
Assuming your virtualenv is activated.

Transcribe a single audio file with NB Whisper:
```
sprakbanken-whisper-jax NbAiLab/nb-whisper-large-beta /path/to/transcription/dir -a /path/to/audio/file
```

Transcribe all mp3 files in a directory:
```
python run_whisper_jax.py NbAiLab/nb-whisper-large-beta /path/to/transcription/dir -m -d /path/to/audio/file -f mp3 
```

By default, the script produces two files per transcription, a jsonl file with timecoded transcriptions and a txt file
with only the transcription text, but this can be controled with the `-r` flag. Transcription files have the same file
stem as audio files, unless a file stem is passed with the `-s` flag (only works with single audio file transcription).

Output from `sprakbanken-whisper-jax --help`:
```
Transcribe audio files with Jax Whisper

positional arguments:
  model                 Model to use
  out_dir               Path to output directory

options:
  -h, --help            show this help message and exit
  -m, --transcribe_many
                        Transcribe many files. Requires --audio_dir
  -a AUDIO_FILE, --audio_file AUDIO_FILE
                        Path to audio file
  -s FILESTEM, --filestem FILESTEM
                        Filestem to use for output files
  -d AUDIO_DIR, --audio_dir AUDIO_DIR
                        Path to directory with audio files
  -t TASK, --task TASK  Task to run
  -l LANGUAGE, --language LANGUAGE
                        Language to use
  -f AUDIO_FORMAT, --audio-format AUDIO_FORMAT
                        Audio format
  -r TRANSCRIPTION_FORMAT, --transcription-format TRANSCRIPTION_FORMAT
                        Transcription format. Specify 'txt' or 'jsonl'. By default, both are produced
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size
  -R, --return-timestamps
                        Return timestamps
```

