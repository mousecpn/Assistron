import threading
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont


class SharedControlStatusWindows:
	def __init__(self, snapshot_provider, logger=None, title="PI05 Shared Control"):
		self.snapshot_provider = snapshot_provider
		self.logger = logger
		self.title = title
		self.root = None
		self._closed = threading.Event()
		self._started = threading.Event()
		self._prompt_var = None
		self._extra_frame = None
		self._extra_vars = {}
		self._state_labels = {}
		self._font_scale = 1.0
		self._base_size = (960, 600)
		self._state_order = (
			("AUTO", "auto"),
			("ASSIST", "assist"),
		)
		self._thread = threading.Thread(target=self._run, daemon=True)
		self._thread.start()

	def _log_warning(self, message):
		if self.logger is not None:
			self.logger.warning(message)

	def _run(self):
		if tk is None or ttk is None:
			self._log_warning("tkinter is not available; status window disabled.")
			self._started.set()
			return

		try:
			root = tk.Tk()
		except Exception as exc:
			self._log_warning(f"Status window disabled: {exc}")
			self._started.set()
			return

		self.root = root
		self._configure_window()
		self._build_ui()
		self._started.set()
		self._refresh()
		root.mainloop()
		self._closed.set()

	def _configure_window(self):
		self.root.title(self.title)
		self.root.geometry("960x600")
		self.root.minsize(860, 540)
		self.root.configure(bg="#0f172a")

		self._title_font = tkfont.Font(family="DejaVu Sans", size=24, weight="bold")
		self._subtitle_font = tkfont.Font(family="DejaVu Sans", size=13)
		self._section_font = tkfont.Font(family="DejaVu Sans", size=13, weight="bold")
		self._value_font = tkfont.Font(family="DejaVu Sans", size=24, weight="bold")
		self._body_font = tkfont.Font(family="DejaVu Sans", size=12)
		self._state_label_font = tkfont.Font(family="DejaVu Sans", size=15, weight="bold")
		self._state_value_font = tkfont.Font(family="DejaVu Sans", size=17, weight="bold")
		self._state_active_font = tkfont.Font(family="DejaVu Sans", size=20, weight="bold")

		self._style = ttk.Style(self.root)
		try:
			self._style.theme_use("clam")
		except Exception:
			pass

		self._style.configure("Root.TFrame", background="#0f172a")
		self._style.configure("Card.TFrame", background="#111827")
		self._style.configure("Title.TLabel", background="#0f172a", foreground="#e5e7eb", font=self._title_font)
		self._style.configure("Section.TLabel", background="#111827", foreground="#93c5fd", font=self._section_font)
		self._base_wraplength = 820
		self._style.configure("Value.TLabel", background="#111827", foreground="#f9fafb", font=self._value_font, wraplength=self._base_wraplength)
		self._style.configure("Body.TLabel", background="#111827", foreground="#cbd5e1", font=self._body_font, wraplength=self._base_wraplength)
		self._style.configure("StateLabel.TLabel", background="#111827", foreground="#cbd5e1", font=self._state_label_font)
		self._style.configure("StateValue.TLabel", background="#111827", foreground="#94a3b8", font=self._state_value_font)
		self._style.configure("StateActive.TFrame", background="#2b0f16")
		self._style.configure("StateActiveLabel.TLabel", background="#2b0f16", foreground="#fda4af", font=self._state_active_font)
		self._style.configure("StateActiveValue.TLabel", background="#2b0f16", foreground="#fb7185", font=self._state_active_font)
		self._assist_font = tkfont.Font(family="DejaVu Sans", size=36, weight="bold")
		self._style.configure("Assist.TLabel", background="#0f172a", foreground="#facc15", font=self._assist_font)
		self._style.configure("AssistHidden.TLabel", background="#0f172a", foreground="#0f172a", font=self._assist_font)

		self.root.protocol("WM_DELETE_WINDOW", self.close)
		self.root.bind("<Configure>", self._on_resize)

	def _build_ui(self):
		root_frame = ttk.Frame(self.root, style="Root.TFrame", padding=24)
		root_frame.pack(fill="both", expand=True)

		header_row = ttk.Frame(root_frame, style="Root.TFrame")
		header_row.pack(fill="x", anchor="w")
		header = ttk.Label(header_row, text="Shared Control Panel", style="Title.TLabel")
		header.pack(side="left", anchor="w")

		subtitle = ttk.Label(
			root_frame,
			text="Live instruction, full FSM state list, and optional extras",
			style="Body.TLabel",
		)
		subtitle.pack(anchor="w", pady=(6, 16))
		self._subtitle_label = subtitle

		card = ttk.Frame(root_frame, style="Card.TFrame", padding=22)
		card.pack(fill="both", expand=True)

		ttk.Label(card, text="Language Instruction", style="Section.TLabel").pack(anchor="w")
		self._prompt_var = tk.StringVar(value="-")
		self._prompt_label = ttk.Label(card, textvariable=self._prompt_var, style="Value.TLabel")
		self._prompt_label.pack(anchor="w", pady=(8, 18))

		ttk.Separator(card, orient="horizontal").pack(fill="x", pady=(0, 12))

		ttk.Label(card, text="FSM States", style="Section.TLabel").pack(anchor="w")
		state_grid = ttk.Frame(card, style="Card.TFrame")
		state_grid.pack(fill="x", expand=False, pady=(10, 12))
		for column_index, (label_text, state_value) in enumerate(self._state_order):
			state_card = ttk.Frame(state_grid, style="Card.TFrame", padding=(14, 12))
			state_card.grid(row=0, column=column_index, sticky="nsew", padx=8)
			state_grid.columnconfigure(column_index, weight=1)

			label_widget = ttk.Label(state_card, text=label_text, style="StateLabel.TLabel")
			label_widget.pack(anchor="w")
			self._state_labels[state_value] = {
				"frame": state_card,
				"label": label_widget,
			}

		self._extra_frame = ttk.Frame(card, style="Card.TFrame")
		self._extra_frame.pack(fill="x", expand=False, pady=(4, 0))
		self._render_extra_fields({})

	def _render_extra_fields(self, extra_fields):
		for child in self._extra_frame.winfo_children():
			child.destroy()

		self._extra_vars = {}
		if not extra_fields:
			return

		ttk.Label(self._extra_frame, text="Extra Fields", style="Section.TLabel").pack(anchor="w", pady=(0, 6))
		for label, value in extra_fields.items():
			row = ttk.Frame(self._extra_frame, style="Card.TFrame")
			row.pack(fill="x", pady=2)
			ttk.Label(row, text=f"{label}:", style="Body.TLabel").pack(side="left")
			var = tk.StringVar(value=str(value))
			ttk.Label(row, textvariable=var, style="Body.TLabel").pack(side="left", padx=(6, 0))
			self._extra_vars[label] = var

	def _refresh(self):
		if self._closed.is_set() or self.root is None:
			return

		snapshot = self.snapshot_provider() or {}
		self._prompt_var.set(snapshot.get("language_instruction", "-"))
		self._apply_state_highlight(snapshot.get("fsm_state", "-").lower())

		extra_fields = snapshot.get("extra_fields", {}) or {}
		if set(extra_fields.keys()) != set(self._extra_vars.keys()):
			self._render_extra_fields(extra_fields)
		else:
			for label, value in extra_fields.items():
				self._extra_vars[label].set(str(value))

		self.root.after(120, self._refresh)

	def _apply_state_highlight(self, current_state):
		if current_state in ("manual", "shared", "shared_control"):
			current_state = "assist"
		for state_value, widgets in self._state_labels.items():
			is_active = state_value == current_state
			if is_active:
				widgets["frame"].configure(style="StateActive.TFrame")
				widgets["label"].configure(style="StateActiveLabel.TLabel")
			else:
				widgets["frame"].configure(style="Card.TFrame")
				widgets["label"].configure(style="StateLabel.TLabel")

	def _on_resize(self, event):
		if self.root is None or event.widget is not self.root:
			return

		scale = min(event.width / self._base_size[0], event.height / self._base_size[1])
		scale = max(0.5, min(scale, 2.0))
		if abs(scale - self._font_scale) < 0.02:
			return

		self._font_scale = scale
		self._title_font.configure(size=max(8, int(24 * scale)))
		self._subtitle_font.configure(size=max(7, int(13 * scale)))
		self._section_font.configure(size=max(7, int(13 * scale)))
		self._value_font.configure(size=max(8, int(24 * scale)))
		self._body_font.configure(size=max(7, int(12 * scale)))
		self._state_label_font.configure(size=max(8, int(15 * scale)))
		self._state_value_font.configure(size=max(8, int(17 * scale)))
		self._state_active_font.configure(size=max(8, int(20 * scale)))
		self._assist_font.configure(size=max(10, int(36 * scale)))

		wraplength = max(200, int(self._base_wraplength * scale))
		self._style.configure("Value.TLabel", wraplength=wraplength)
		self._style.configure("Body.TLabel", wraplength=wraplength)
		self._prompt_label.configure(wraplength=wraplength)
		self._subtitle_label.configure(wraplength=wraplength)

	def close(self):
		self._closed.set()
		if self.root is not None:
			try:
				self.root.after(0, self.root.quit)
			except Exception:
				pass
