#!/usr/bin/env python3
import argparse
import io
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import librosa
import numpy as np
import soundfile as sf

from whisper_online import add_shared_args, asr_factory, set_logging


logger = logging.getLogger(__name__)
SAMPLING_RATE = 16000


def transcribe_clip(asr, audio_16k):
	"""Transcribe one clip and return plain text."""
	res = asr.transcribe(audio_16k)
	words = asr.ts_words(res)
	text = asr.sep.join(w for _, _, w in words).strip()
	return text


def read_wav_to_mono_float32_16k(raw_bytes):
	"""Decode wav bytes, force mono float32 16 kHz for Whisper."""
	with io.BytesIO(raw_bytes) as bio:
		audio, sr = sf.read(bio, dtype="float32", always_2d=False)

	if isinstance(audio, np.ndarray) and audio.ndim == 2:
		audio = np.mean(audio, axis=1)

	if sr != SAMPLING_RATE:
		audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)

	return audio.astype(np.float32)


class TranscribeHandler(BaseHTTPRequestHandler):
	server_version = "WhisperHTTP/0.1"

	def do_POST(self):
		if self.path != "/transcribe":
			self.send_error(404, "Not found")
			return

		try:
			content_len = int(self.headers.get("Content-Length", "0"))
		except ValueError:
			self.send_error(400, "Invalid Content-Length")
			return

		if content_len <= 0:
			self.send_error(400, "Empty body")
			return

		raw = self.rfile.read(content_len)

		try:
			audio = read_wav_to_mono_float32_16k(raw)
			text = transcribe_clip(self.server.asr, audio)
		except Exception as exc:
			logger.exception("Transcription failed")
			self._send_json(
				500,
				{
					"ok": False,
					"error": str(exc),
					"text": "",
				},
			)
			return

		self._send_json(200, {"ok": True, "text": text})

	def do_GET(self):
		if self.path == "/health":
			self._send_json(200, {"ok": True})
			return
		self.send_error(404, "Not found")

	def log_message(self, fmt, *args):
		logger.info("%s - %s", self.address_string(), fmt % args)

	def _send_json(self, status_code, payload):
		body = json.dumps(payload).encode("utf-8")
		self.send_response(status_code)
		self.send_header("Content-Type", "application/json")
		self.send_header("Content-Length", str(len(body)))
		self.end_headers()
		self.wfile.write(body)


def parse_args():
	parser = argparse.ArgumentParser(
		description="HTTP server for one-shot whisper transcription"
	)
	parser.add_argument("--host", type=str, default="0.0.0.0")
	parser.add_argument("--port", type=int, default=43100)
	parser.add_argument(
		"--warmup-file",
		type=str,
		dest="warmup_file",
		default=None,
		help="Optional wav file for warmup",
	)
	add_shared_args(parser)
	return parser.parse_args()


def main():
	args = parse_args()
	set_logging(args, logger, other="")
	requested_lan = getattr(args, "lan", None)
	if requested_lan != "en":
		logger.warning("Forcing English transcription mode (requested language: %s)", requested_lan)
	args.lan = "en"

	asr, _ = asr_factory(args)

	if args.warmup_file:
		from whisper_online import load_audio_chunk

		warmup_audio = load_audio_chunk(args.warmup_file, 0, 1)
		asr.transcribe(warmup_audio)
		logger.info("ASR warmed up with %s", args.warmup_file)

	server = ThreadingHTTPServer((args.host, args.port), TranscribeHandler)
	server.asr = asr

	logger.info("HTTP transcription server listening on %s:%s", args.host, args.port)
	try:
		server.serve_forever()
	except KeyboardInterrupt:
		logger.info("Shutting down")
	finally:
		server.server_close()


if __name__ == "__main__":
	main()
