from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp
from pathlib import Path
import json
import argparse


def transcribe(
    pipeline, audio_file, task="transcribe", language="no", return_timestamps=True
):
    print(f"Transcribing {audio_file}")
    try:
        return pipeline(
            audio_file, task=task, language=language, return_timestamps=return_timestamps
        )
    except Exception as e:
        print(f"Transcription failed for {audio_file}")
        print(e)
        return dict(text="", chunks=[])


def save_transcription(audio_file, transcript, outdir, format=None, filestem=None):
    try:
        assert format in ["txt", "jsonl", "srt", None]
    except AssertionError:
        print("Format not supported")
        return

    try:
        assert isinstance(transcript, dict) and ["text", "chunks"] == list(
            transcript.keys()
        )
    except AssertionError:
        print("Transcript not in correct format")
        return

    if filestem is None:
        filestem = Path(audio_file).stem
    outpath = Path(outdir)
    if format is None:
        textpath = outpath / f"{filestem}.txt"
        jsonlpath = outpath / f"{filestem}.jsonl"
        with textpath.open("w") as f:
            f.write(transcript["text"])

        with jsonlpath.open("w") as f:
            for line in transcript["chunks"]:
                json.dump(line, f, ensure_ascii=False)
                f.write("\n")

    elif format == "txt":
        textpath = outpath / f"{filestem}.txt"
        with textpath.open("w") as f:
            f.write(transcript["text"])
    elif format == "jsonl":
        jsonlpath = outpath / f"{filestem}.jsonl"
        with jsonlpath.open("w") as f:
            for line in transcript["chunks"]:
                json.dump(line, f, ensure_ascii=False)
                f.write("\n")

    elif format == "srt":
        raise NotImplementedError("SRT format not implemented yet")


def transcribe_all(
    pipeline,
    audio_dir,
    outdir,
    task="transcribe",
    language="no",
    audio_format="wav",
    transcription_format=None,
    return_timestamps=True,
):
    try:
        assert audio_format in ["wav", "mp3"]
    except AssertionError:
        print("Audio format not supported")
        return
    audio_dir = Path(audio_dir)
    outdir = Path(outdir)
    transcribed = list(set([f.stem for f in outdir.glob("*.*")]))
    for audio_file in audio_dir.glob(f"*.{audio_format}"):
        if audio_file.stem in transcribed:
            continue
        transcript = transcribe(
            pipeline,
            str(audio_file),
            task=task,
            language=language,
            return_timestamps=return_timestamps,
        )
        save_transcription(audio_file, transcript, outdir, format=transcription_format)
        transcribed.append(audio_file.stem)

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files with Jax Whisper"
    )
    parser.add_argument("model", type=str, help="Model to use")
    parser.add_argument("out_dir", type=str, help="Path to output directory")
    parser.add_argument(
        "-m",
        "--transcribe_many",
        action="store_true",
        help="Transcribe many files. Requires --audio_dir",
    )
    parser.add_argument("-a", "--audio_file", type=str, help="Path to audio file")
    parser.add_argument(
        "-s",
        "--filestem",
        type=str,
        default=None,
        help="Filestem to use for output files",
    )
    parser.add_argument(
        "-d", "--audio_dir", type=str, help="Path to directory with audio files"
    )
    parser.add_argument(
        "-t", "--task", type=str, default="transcribe", help="Task to run"
    )
    parser.add_argument(
        "-l", "--language", type=str, default=None, help="Language to use"
    )
    parser.add_argument(
        "-f", "--audio-format", type=str, default="wav", help="Audio format"
    )
    parser.add_argument(
        "-r",
        "--transcription-format",
        type=str,
        default=None,
        help="Transcription format. Specify 'txt' or 'jsonl'. By default, both are produced",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument(
        "-R", "--return-timestamps", action="store_true", help="Return timestamps"
    )
    args = parser.parse_args()

    if args.batch_size is None:
        pipeline = FlaxWhisperPipline(args.model, dtype=jnp.bfloat16)
    else:
        pipeline = FlaxWhisperPipline(
            args.model, dtype=jnp.bfloat16, batch_size=args.batch_size
        )
    print(f"Loaded model {args.model}")
    if not args.transcribe_many:
        try:
            assert args.audio_file is not None
            transcript = transcribe(
                pipeline,
                args.audio_file,
                task=args.task,
                language=args.language,
                return_timestamps=args.return_timestamps,
            )
            save_transcription(
                args.audio_file,
                transcript,
                args.out_dir,
                format=args.transcription_format,
                filestem=args.filestem,
            )
        except AssertionError:
            print("Must specify --audio_file when transcribing a single file")
    else:
        try:
            assert args.audio_dir is not None
            transcribe_all(
                pipeline,
                args.audio_dir,
                args.out_dir,
                task=args.task,
                language=args.language,
                audio_format=args.audio_format,
                transcription_format=args.transcription_format,
                return_timestamps=args.return_timestamps,
            )
        except AssertionError:
            print("Must specify --audio_dir when transcribing many files")


if __name__ == "__main__":
    main()