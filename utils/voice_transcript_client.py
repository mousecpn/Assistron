#!/usr/bin/env python3
import argparse
import io
import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import sounddevice as sd
import soundfile as sf
from pynput import keyboard


class PushToTalkClient:
    def __init__(self, server_url, samplerate=16000, channels=1):
        self.server_url = server_url.rstrip("/") + "/transcribe"
        self.samplerate = samplerate
        self.channels = channels

        self._recording = False
        self._frames = []
        self._lock = threading.Lock()

    def _audio_callback(self, indata, frames, time_info, status):
        del frames, time_info
        if status:
            print(f"Audio callback status: {status}")

        with self._lock:
            if self._recording:
                # Store a copy because sounddevice buffer is reused.
                self._frames.append(indata.copy())

    def _start_recording(self):
        with self._lock:
            if self._recording:
                return
            self._frames = []
            self._recording = True
        print("Recording... release SPACE to send")

    def _stop_recording_and_send(self):
        with self._lock:
            if not self._recording:
                return
            self._recording = False
            frames = list(self._frames)

        if not frames:
            print("No audio captured")
            return

        audio = np.concatenate(frames, axis=0)
        if self.channels == 1:
            audio = audio[:, 0]

        wav_bytes = self._to_wav_bytes(audio)
        print("Sending audio to server...")
        text = self._send_request(wav_bytes)
        print(f"Transcript: {text}")

    def _to_wav_bytes(self, audio):
        buf = io.BytesIO()
        sf.write(buf, audio, self.samplerate, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def _send_request(self, wav_bytes):
        req = urllib.request.Request(
            self.server_url,
            data=wav_bytes,
            headers={"Content-Type": "audio/wav"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP error {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Connection error: {exc}") from exc

        if not payload.get("ok"):
            raise RuntimeError(payload.get("error", "Unknown server error"))
        return payload.get("text", "")

    def run(self):
        print("Hold SPACE to record. Release SPACE to send. Press ESC to exit.")

        with sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="float32",
            callback=self._audio_callback,
        ):

            def on_press(key):
                if key == keyboard.Key.space:
                    self._start_recording()

            def on_release(key):
                if key == keyboard.Key.space:
                    try:
                        self._stop_recording_and_send()
                    except Exception as exc:
                        print(f"Send failed: {exc}")
                elif key == keyboard.Key.esc:
                    return False

            with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
                listener.join()


def spellchecker_prompt(prompt, prompt_history=None):
    """Optional helper to fix ASR typos using a spellchecker API."""
    # inaccurate words → replace with more common synonyms (e.g., "grasp" → "pick up")
    if 'grasp' in prompt.lower():
        prompt = prompt.lower().replace('grasp', 'pick up')
    if 'jar' in prompt.lower():
        prompt = prompt.lower().replace('jar', 'drawer')


    ## adding description
    # if 'box' in prompt.lower() and 'brown' not in prompt.lower():
    #     prompt = prompt.lower().replace('box', 'brown box')
    # if 'drawer' in prompt.lower() and 'white' not in prompt.lower():
    #     prompt = prompt.lower().replace('drawer', 'white drawer')
    # if ' pen' in prompt.lower() and 'marker' not in prompt.lower():
    #     prompt = prompt.lower().replace(' pen', ' marker pen')
    
    ## strange rules
    if 'grip' in prompt.lower():
        prompt = prompt.lower().replace('grip', 'grape')
    # the word begins with "gra" will be replaced with "grape" (e.g., "grab" -> "grape")
    import re
    prompt = re.sub(r'\bgra\w*\b', 'grape', prompt, flags=re.IGNORECASE)


    if 'grape' in prompt.lower() and 'drawer' in prompt.lower():
        ## if open the drawer exists in the prompt history, no need to add open the drawer again
        if prompt_history and any(('open' in p.lower() and 'drawer' in p.lower()) for p in prompt_history):
            pass
        else:   
            prompt = 'open the drawer, and put the grape in the drawer'

    return prompt
    


# =============================================================================
# Voice recorder (unchanged)
# =============================================================================
class VoiceRecorder:
    """Records audio while joystick voice button is held, sends to whisper server on release."""

    def __init__(self, server_url="http://127.0.0.1:43100", samplerate=16000, channels=1):
        self.server_url = server_url.rstrip("/") + "/transcribe"
        self.samplerate = samplerate
        self.channels = channels

        self._recording = False
        self._frames = []
        self._lock = threading.Lock()
        self._prompt_lock = threading.Lock()
        self._current_prompt = "stop"
        self._prompt_history = []

        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()

    def _audio_callback(self, indata, frames, time_info, status):
        with self._lock:
            if self._recording:
                self._frames.append(indata.copy())

    def update_button(self, voice_pressed):
        if voice_pressed and not self._recording:
            self._start_recording()
        elif not voice_pressed and self._recording:
            self._stop_recording_and_send()

    def update_prompt(self, new_prompt):
        with self._prompt_lock:
            self._current_prompt = new_prompt

    def _start_recording(self):
        with self._lock:
            self._frames = []
            self._recording = True
        print("🎙️ Voice recording started... release button to send")

    def _stop_recording_and_send(self):
        with self._lock:
            self._recording = False
            frames = list(self._frames)
        if not frames:
            print("⚠️ No audio captured")
            return
        audio = np.concatenate(frames, axis=0)
        if self.channels == 1 and audio.ndim == 2:
            audio = audio[:, 0]
        threading.Thread(target=self._send_async, args=(audio,), daemon=True).start()

    def _send_async(self, audio):
        try:
            buf = io.BytesIO()
            sf.write(buf, audio, self.samplerate, format="WAV", subtype="PCM_16")
            wav_bytes = buf.getvalue()
            print("📤 Sending audio to whisper server...")
            req = urllib.request.Request(
                self.server_url,
                data=wav_bytes,
                headers={"Content-Type": "audio/wav"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if not payload.get("ok"):
                print(f"❌ Whisper server error: {payload.get('error', 'Unknown')}")
                return
            text = payload.get("text", "").strip()
            if text:
                corrected = spellchecker_prompt(text, self.prompt_history)
                with self._prompt_lock:
                    self._current_prompt = corrected
                    self._prompt_history.append(corrected)
                print(f"💬 Voice prompt updated: {text} → {corrected}")
            else:
                print("⚠️ Whisper returned empty transcript, keeping previous prompt")
        except Exception as exc:
            print(f"❌ Voice transcription failed: {exc}")

    @property
    def prompt(self):
        with self._prompt_lock:
            return self._current_prompt

    @property
    def prompt_history(self):
        with self._prompt_lock:
            return list(self._prompt_history)

    def close(self):
        self._stream.stop()
        self._stream.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Push-to-talk whisper client")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:43100")
    parser.add_argument("--samplerate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    client = PushToTalkClient(
        server_url=args.server,
        samplerate=args.samplerate,
        channels=args.channels,
    )
    client.run()


if __name__ == "__main__":
    main()
