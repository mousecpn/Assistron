#!/usr/bin/env python3
"""
Test script: VoiceRecorder + joystick Y-button via ROS2.

Hold the Y button on the joystick to record audio from the microphone.
Release to send the recording to the whisper server and print the transcript.
Press Ctrl+C to exit.

Requirements:
  - ROS2 joy node publishing /joy topic
  - Whisper HTTP server running at --server (default http://127.0.0.1:43100)
  - sounddevice, soundfile, numpy
"""
import argparse
import io
import json
import threading
import time
import urllib.request
import urllib.error

import numpy as np
import sounddevice as sd
import soundfile as sf

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy

# Y button index on Xbox-style controllers
Y_BUTTON = 3


class VoiceRecorder:
    """Records audio while button is held, sends to whisper server on release."""

    def __init__(self, server_url="http://127.0.0.1:43100", samplerate=16000, channels=1):
        self.server_url = server_url.rstrip("/") + "/transcribe"
        self.samplerate = samplerate
        self.channels = channels

        self._recording = False
        self._frames = []
        self._lock = threading.Lock()
        self._prompt_lock = threading.Lock()
        self._current_prompt = "(no voice command yet)"

        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()
        print(f"[VoiceRecorder] Audio stream opened (sr={samplerate}, ch={channels})")
        print(f"[VoiceRecorder] Whisper endpoint: {self.server_url}")

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[Audio status] {status}")
        with self._lock:
            if self._recording:
                self._frames.append(indata.copy())

    def update_button(self, pressed: bool):
        if pressed and not self._recording:
            self._start_recording()
        elif not pressed and self._recording:
            self._stop_recording_and_send()

    def _start_recording(self):
        with self._lock:
            self._frames = []
            self._recording = True
        print("🎙️  Recording... release Y button to send")

    def _stop_recording_and_send(self):
        with self._lock:
            self._recording = False
            frames = list(self._frames)

        if not frames:
            print("⚠️  No audio captured (pressed too briefly?)")
            return

        audio = np.concatenate(frames, axis=0)
        if self.channels == 1 and audio.ndim == 2:
            audio = audio[:, 0]

        duration = len(audio) / self.samplerate
        print(f"📤 Captured {duration:.2f}s of audio, sending to whisper server...")
        threading.Thread(target=self._send_async, args=(audio,), daemon=True).start()

    def _send_async(self, audio):
        try:
            buf = io.BytesIO()
            sf.write(buf, audio, self.samplerate, format="WAV", subtype="PCM_16")
            wav_bytes = buf.getvalue()

            req = urllib.request.Request(
                self.server_url,
                data=wav_bytes,
                headers={"Content-Type": "audio/wav"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                payload = json.loads(resp.read().decode("utf-8"))

            if not payload.get("ok"):
                print(f"❌ Whisper error: {payload.get('error', 'Unknown')}")
                return

            text = payload.get("text", "").strip()
            if text:
                with self._prompt_lock:
                    self._current_prompt = text
                print(f"✅ Transcript: \"{text}\"")
            else:
                print("⚠️  Empty transcript returned")
        except Exception as exc:
            print(f"❌ Transcription failed: {exc}")

    @property
    def prompt(self):
        with self._prompt_lock:
            return self._current_prompt

    def close(self):
        self._stream.stop()
        self._stream.close()


def main():
    parser = argparse.ArgumentParser(description="Test VoiceRecorder with joystick Y button (ROS2)")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:43100",
                        help="Whisper HTTP server URL")
    parser.add_argument("--samplerate", type=int, default=16000)
    parser.add_argument("--button", type=int, default=Y_BUTTON,
                        help="Joystick button index to use as voice trigger (default: 3 = Y)")
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node("test_voice_joystick")

    recorder = VoiceRecorder(server_url=args.server, samplerate=args.samplerate)
    button_idx = args.button

    def joy_callback(msg: Joy):
        if len(msg.buttons) <= button_idx:
            return
        pressed = bool(msg.buttons[button_idx])
        recorder.update_button(pressed)

    node.create_subscription(Joy, '/joy', joy_callback, 1)

    print(f"🎮 Listening on /joy topic, button index {button_idx} as voice trigger")
    print(f"   Hold button {button_idx} → speak → release to transcribe")
    print(f"   Press Ctrl+C to exit\n")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        recorder.close()
        node.destroy_node()
        rclpy.shutdown()
        print(f"Final prompt: \"{recorder.prompt}\"")


if __name__ == "__main__":
    main()
