"""
Volleyball Set Labelling Tool
APS360 Project — Data Collection

Usage:
    python labelling_tool.py
    Then use File > Open Video to load an MP4.

Controls:
    Space       — Play / Pause
    Left/Right  — Step one frame backward / forward
    Shift+Left  — Jump 1 second back
    Shift+Right — Jump 1 second forward
    M           — Mark current frame as set point
    Enter       — Confirm clip during preview
    Esc         — Cancel current operation
    Ctrl+Q      — Quit
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import os
import csv
import uuid
import time
import threading
import queue
from PIL import Image, ImageTk
from datetime import datetime


# ─── Constants ────────────────────────────────────────────────────────────────

DEFAULT_CLIP_DURATION = 1.0        # seconds before the mark
PLAYBACK_SPEEDS       = [0.25, 0.5, 1.0, 1.5, 2.0]
SET_DIRECTIONS        = ["Left Side", "Middle", "Right Side", "Pipe", "Dump"]


# ─── Main Application ────────────────────────────────────────────────────────

class LabellingTool(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Volleyball Set Labelling Tool")
        self.configure(bg="#1e1e1e")

        # ── Video state ──
        self.cap = None
        self.video_path = None
        self.total_frames = 0
        self.fps = 30.0
        self.frame_width = 0
        self.frame_height = 0
        self.current_frame_idx = 0
        self.current_frame = None       # numpy BGR
        self.display_scale = 1.0        # ratio: display / original
        self.playing = False
        self.play_speed = 1.0
        self.after_id = None

        # ── Threaded decode state ──
        self._frame_queue = queue.Queue(maxsize=8)
        self._decode_thread = None
        self._decode_stop = threading.Event()

        # ── Display cache ──
        self._display_w = 0             # pre-computed display dimensions
        self._display_h = 0
        self._img_x = 0
        self._img_y = 0
        self._canvas_image_id = None    # persistent canvas image item

        # ── Marker state ──
        self.markers = []               # list of saved clip info
        self.pending_marker_frame = None
        self.clip_duration = DEFAULT_CLIP_DURATION

        # ── Preview playback state ──
        self.previewing = False
        self.preview_after_id = None
        self._preview_frames = []       # pre-loaded frames for looping preview
        self._preview_idx = 0
        self._preview_start_time = None

        # ── Label output ──
        self.labels_dir = None
        self.master_csv_path = None

        # ── Build UI ──
        self._build_menu()
        self._build_layout()
        self._bind_keys()

        # Centre window
        self.update_idletasks()
        self.geometry("1200x750")
        self.minsize(900, 600)

    # ─── Menu Bar ─────────────────────────────────────────────────────────────

    def _build_menu(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Video…", command=self._open_video,
                              accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self._quit,
                              accelerator="Ctrl+Q")
        menubar.add_cascade(label="File", menu=file_menu)
        self.config(menu=menubar)
        self.bind_all("<Control-o>", lambda e: self._open_video())
        self.bind_all("<Control-q>", lambda e: self._quit())

    # ─── Layout ───────────────────────────────────────────────────────────────

    def _build_layout(self):
        # Main horizontal split: video area (left) + tool panel (right)
        self.main_pane = tk.PanedWindow(self, orient=tk.HORIZONTAL,
                                        bg="#2d2d2d", sashwidth=4)
        self.main_pane.pack(fill=tk.BOTH, expand=True)

        # ── Left: video + timeline ──
        left_frame = tk.Frame(self.main_pane, bg="#1e1e1e")
        self.main_pane.add(left_frame, stretch="always")

        self.canvas = tk.Canvas(left_frame, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5, 0))
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Timeline
        timeline_frame = tk.Frame(left_frame, bg="#1e1e1e")
        timeline_frame.pack(fill=tk.X, padx=5, pady=5)

        self.time_label = tk.Label(timeline_frame, text="00:00 / 00:00",
                                   bg="#1e1e1e", fg="white", font=("Menlo", 11))
        self.time_label.pack(side=tk.LEFT, padx=(0, 8))

        self.timeline = ttk.Scale(timeline_frame, from_=0, to=100,
                                  orient=tk.HORIZONTAL,
                                  command=self._on_timeline_seek)
        self.timeline.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.frame_label = tk.Label(timeline_frame, text="Frame: 0 / 0",
                                    bg="#1e1e1e", fg="#aaa", font=("Menlo", 10))
        self.frame_label.pack(side=tk.RIGHT, padx=(8, 0))

        # ── Right: tool panel ──
        right_frame = tk.Frame(self.main_pane, bg="#2d2d2d", width=260)
        self.main_pane.add(right_frame, stretch="never")

        pad = dict(padx=12, pady=4)
        header_font = ("Helvetica", 13, "bold")
        label_font = ("Helvetica", 11)

        tk.Label(right_frame, text="Controls", font=header_font,
                 bg="#2d2d2d", fg="white").pack(padx=12, anchor="w", pady=(12, 4))

        # Play / Pause button
        self.play_btn = tk.Button(right_frame, text="▶  Play", width=20,
                                  command=self._toggle_play,
                                  font=label_font)
        self.play_btn.pack(**pad)

        # Playback speed
        speed_frame = tk.Frame(right_frame, bg="#2d2d2d")
        speed_frame.pack(**pad, fill=tk.X)
        tk.Label(speed_frame, text="Speed:", bg="#2d2d2d", fg="white",
                 font=label_font).pack(side=tk.LEFT)
        self.speed_var = tk.StringVar(value="1.0×")
        self.speed_combo = ttk.Combobox(speed_frame, textvariable=self.speed_var,
                                        values=[f"{s}×" for s in PLAYBACK_SPEEDS],
                                        width=6, state="readonly")
        self.speed_combo.pack(side=tk.LEFT, padx=(8, 0))
        self.speed_combo.bind("<<ComboboxSelected>>", self._on_speed_change)

        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # ── Clip settings ──
        tk.Label(right_frame, text="Clip Settings", font=header_font,
                 bg="#2d2d2d", fg="white").pack(**pad, anchor="w")

        dur_frame = tk.Frame(right_frame, bg="#2d2d2d")
        dur_frame.pack(**pad, fill=tk.X)
        tk.Label(dur_frame, text="Clip length (sec):", bg="#2d2d2d", fg="white",
                 font=label_font).pack(side=tk.LEFT)
        self.dur_var = tk.StringVar(value=str(DEFAULT_CLIP_DURATION))
        self.dur_entry = tk.Entry(dur_frame, textvariable=self.dur_var, width=6,
                                  font=label_font)
        self.dur_entry.pack(side=tk.LEFT, padx=(8, 0))

        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # ── Mark set ──
        tk.Label(right_frame, text="Labelling", font=header_font,
                 bg="#2d2d2d", fg="white").pack(**pad, anchor="w")

        self.mark_btn = tk.Button(right_frame, text="⚑  Mark Set (M)", width=20,
                                  command=self._mark_set, font=label_font,
                                  bg="#c0392b", fg="white",
                                  activebackground="#e74c3c")
        self.mark_btn.pack(padx=12, pady=(0, 4))

        # Confirm button (shown during preview)
        self.confirm_btn = tk.Button(right_frame, text="✓  Confirm (Enter)",
                                     width=20, command=self._confirm_preview,
                                     font=label_font, bg="#27ae60", fg="white",
                                     activebackground="#2ecc71")
        self.confirm_btn.pack(padx=12, pady=(0, 4))
        self.confirm_btn.pack_forget()  # hidden by default

        # Cancel button
        self.cancel_btn = tk.Button(right_frame, text="✕  Cancel (Esc)", width=20,
                                    command=self._cancel_all, font=label_font,
                                    state=tk.DISABLED)
        self.cancel_btn.pack(padx=12, pady=(0, 4))

        self.status_label = tk.Label(right_frame, text="Load a video to begin.",
                                     bg="#2d2d2d", fg="#aaa", font=label_font,
                                     wraplength=230, justify=tk.LEFT)
        self.status_label.pack(**pad, anchor="w")

        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # ── Markers list ──
        tk.Label(right_frame, text="Saved Labels", font=header_font,
                 bg="#2d2d2d", fg="white").pack(**pad, anchor="w")

        self.marker_listbox = tk.Listbox(right_frame, bg="#1e1e1e", fg="white",
                                         font=("Menlo", 10), selectbackground="#444",
                                         height=10)
        self.marker_listbox.pack(padx=12, fill=tk.BOTH, expand=True, pady=(0, 8))

        # ── Quit button ──
        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        self.quit_btn = tk.Button(right_frame, text="Quit (Ctrl+Q)", width=20,
                                  command=self._quit, font=label_font)
        self.quit_btn.pack(padx=12, pady=(4, 12))

    # ─── Keyboard Bindings ────────────────────────────────────────────────────

    def _bind_keys(self):
        self.bind("<space>", lambda e: self._toggle_play())
        self.bind("<Left>", lambda e: self._step_frame(-1))
        self.bind("<Right>", lambda e: self._step_frame(1))
        self.bind("<Shift-Left>", lambda e: self._jump_seconds(-1))
        self.bind("<Shift-Right>", lambda e: self._jump_seconds(1))
        self.bind("<m>", lambda e: self._mark_set())
        self.bind("<M>", lambda e: self._mark_set())
        self.bind("<Escape>", lambda e: self._cancel_all())
        self.bind("<Return>", lambda e: self._confirm_preview())

    # ─── Video Loading ────────────────────────────────────────────────────────

    def _open_video(self):
        path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All", "*.*")]
        )
        if not path:
            return
        self._load_video(path)

    def _load_video(self, path):
        self._stop_decode_thread()
        if self.cap is not None:
            self.cap.release()
            self.playing = False

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Cannot open video:\n{path}")
            return

        self.video_path = path
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        raw_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = raw_fps if 1.0 <= raw_fps <= 240.0 else 30.0
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame_idx = 0

        # Set up output directory in the same folder as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.labels_dir = os.path.join(script_dir, "clips")
        os.makedirs(self.labels_dir, exist_ok=True)
        self.master_csv_path = os.path.join(self.labels_dir, "labels.csv")

        # Initialise CSV if it doesn't exist
        if not os.path.exists(self.master_csv_path):
            with open(self.master_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["clip_id", "source_video", "mark_frame",
                                 "start_frame", "end_frame", "clip_duration_sec",
                                 "set_direction", "setter_x", "setter_y",
                                 "timestamp"])

        # Pre-compute display dimensions
        self._recompute_display_size()

        self.timeline.configure(to=max(self.total_frames - 1, 1))
        self._seek_frame(0)
        self._update_status(
            f"Video loaded ({self.total_frames} frames, {self.fps:.1f} fps).\n"
            "Navigate and press M to mark a set.")
        self.title(f"Volleyball Set Labelling Tool — {os.path.basename(path)}")
        self.markers.clear()
        self.marker_listbox.delete(0, tk.END)

    # ─── Display Helpers ──────────────────────────────────────────────────────

    def _recompute_display_size(self):
        """Pre-compute the display dimensions for the current canvas size."""
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10 or self.frame_width == 0:
            return
        scale = min(cw / self.frame_width, ch / self.frame_height)
        self.display_scale = scale
        self._display_w = int(self.frame_width * scale)
        self._display_h = int(self.frame_height * scale)
        self._img_x = (cw - self._display_w) // 2
        self._img_y = (ch - self._display_h) // 2

    def _frame_to_photo(self, frame):
        """Convert a BGR numpy frame to an ImageTk.PhotoImage (fast path)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self._display_w, self._display_h),
                             interpolation=cv2.INTER_NEAREST)
        img = Image.fromarray(resized)
        return ImageTk.PhotoImage(image=img)

    # ─── Frame Display ────────────────────────────────────────────────────────

    def _seek_frame(self, idx):
        """Seek to a specific frame by index (random access)."""
        if self.cap is None:
            return
        idx = max(0, min(idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_idx = idx
            self.current_frame = frame
            self._display_frame(frame)
            self._update_timeline()

    def _display_frame(self, frame):
        """Render an OpenCV BGR frame onto the tkinter canvas."""
        if self._display_w < 1 or self._display_h < 1:
            self._recompute_display_size()
            if self._display_w < 1:
                return

        self._photo = self._frame_to_photo(frame)

        # Reuse existing canvas image item if possible
        if self._canvas_image_id is not None:
            self.canvas.itemconfig(self._canvas_image_id, image=self._photo)
            self.canvas.coords(self._canvas_image_id, self._img_x, self._img_y)
        else:
            self.canvas.delete("all")
            self._canvas_image_id = self.canvas.create_image(
                self._img_x, self._img_y, anchor=tk.NW, image=self._photo)

        # Remove old overlays
        self.canvas.delete("overlay")

        # If previewing, show preview label
        if self.previewing:
            cw = self.canvas.winfo_width()
            self.canvas.create_text(cw // 2, 25,
                                    text="Previewing clip (looping) — Enter to confirm, Esc to cancel",
                                    fill="#00ccff", font=("Helvetica", 14, "bold"),
                                    tags="overlay")

    def _display_frame_fast(self, photo):
        """Display a pre-converted PhotoImage (fastest path for playback)."""
        self._photo = photo
        if self._canvas_image_id is not None:
            self.canvas.itemconfig(self._canvas_image_id, image=self._photo)
        else:
            self.canvas.delete("all")
            self._canvas_image_id = self.canvas.create_image(
                self._img_x, self._img_y, anchor=tk.NW, image=self._photo)

        self.canvas.delete("overlay")

    def _on_canvas_resize(self, event):
        self._canvas_image_id = None  # invalidate cached canvas item
        self._recompute_display_size()
        if self.current_frame is not None:
            self._display_frame(self.current_frame)

    def _update_timeline(self):
        self.timeline.set(self.current_frame_idx)
        total_sec = self.total_frames / self.fps
        cur_sec = self.current_frame_idx / self.fps
        self.time_label.config(
            text=f"{self._fmt_time(cur_sec)} / {self._fmt_time(total_sec)}")
        self.frame_label.config(
            text=f"Frame: {self.current_frame_idx} / {self.total_frames - 1}")

    @staticmethod
    def _fmt_time(seconds):
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"

    # ─── Threaded Decode ──────────────────────────────────────────────────────

    def _decode_worker(self, cap_path, start_idx):
        """Background thread: decode raw frames only (no tkinter calls)."""
        cap = cv2.VideoCapture(cap_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        idx = start_idx
        while not self._decode_stop.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            try:
                self._frame_queue.put((idx, frame), timeout=0.1)
            except queue.Full:
                if self._decode_stop.is_set():
                    break
                continue
            idx += 1
        cap.release()

    def _start_decode_thread(self, start_idx):
        self._stop_decode_thread()
        self._decode_stop.clear()
        # Drain the queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        self._decode_thread = threading.Thread(
            target=self._decode_worker,
            args=(self.video_path, start_idx),
            daemon=True)
        self._decode_thread.start()

    def _stop_decode_thread(self):
        self._decode_stop.set()
        if self._decode_thread is not None:
            self._decode_thread.join(timeout=1.0)
            self._decode_thread = None
        # Drain queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

    # ─── Playback ─────────────────────────────────────────────────────────────

    def _toggle_play(self):
        if self.cap is None:
            return
        if self.previewing:
            return
        self.playing = not self.playing
        self.play_btn.config(text="⏸  Pause" if self.playing else "▶  Play")
        if self.playing:
            # Start decode thread from next frame
            self._start_decode_thread(self.current_frame_idx + 1)
            self._play_start_time = time.perf_counter()
            self._play_start_frame = self.current_frame_idx
            self._play_loop()

    def _play_loop(self):
        if not self.playing or self.cap is None:
            return

        try:
            idx, frame = self._frame_queue.get_nowait()
        except queue.Empty:
            # No frame ready yet, try again soon
            self.after_id = self.after(2, self._play_loop)
            return

        # Keep real-time playback for high-FPS videos by skipping stale frames
        # when UI rendering is slower than decode speed.
        elapsed = time.perf_counter() - self._play_start_time
        expected_idx = self._play_start_frame + int(elapsed * self.fps * self.play_speed)
        while idx < expected_idx:
            try:
                idx, frame = self._frame_queue.get_nowait()
            except queue.Empty:
                break

        if idx >= self.total_frames:
            self._pause()
            return

        self.current_frame_idx = idx
        self.current_frame = frame
        # Convert to PhotoImage on main thread (tkinter is not thread-safe)
        photo = self._frame_to_photo(frame)
        self._display_frame_fast(photo)
        self._update_timeline()

        # Time-based scheduling: compute when the next frame should appear
        elapsed = time.perf_counter() - self._play_start_time
        frames_played = idx - self._play_start_frame
        target_time = frames_played / (self.fps * self.play_speed)
        delay_sec = target_time - elapsed
        delay_ms = max(1, int(delay_sec * 1000))

        self.after_id = self.after(delay_ms, self._play_loop)

    def _step_frame(self, delta):
        if self.cap is None or self.previewing:
            return
        self._pause()
        self._seek_frame(self.current_frame_idx + delta)

    def _jump_seconds(self, sec):
        if self.cap is None or self.previewing:
            return
        self._pause()
        delta_frames = int(sec * self.fps)
        self._seek_frame(self.current_frame_idx + delta_frames)

    def _pause(self):
        self.playing = False
        self.play_btn.config(text="▶  Play")
        self._stop_decode_thread()
        if self.after_id is not None:
            self.after_cancel(self.after_id)
            self.after_id = None

    def _on_timeline_seek(self, val):
        if self.cap is None:
            return
        idx = int(float(val))
        if idx != self.current_frame_idx:
            self._pause()
            self._seek_frame(idx)

    def _on_speed_change(self, event):
        txt = self.speed_var.get().replace("×", "")
        try:
            self.play_speed = float(txt)
        except ValueError:
            self.play_speed = 1.0

    # ─── Cancel ───────────────────────────────────────────────────────────────

    def _cancel_all(self):
        """Cancel any active operation."""
        if self.previewing:
            self._cancel_preview()

    def _set_cancel_enabled(self, enabled):
        self.cancel_btn.config(state=tk.NORMAL if enabled else tk.DISABLED)

    # ─── Marking ──────────────────────────────────────────────────────────────

    def _mark_set(self):
        if self.cap is None:
            messagebox.showinfo("No video", "Please open a video first.")
            return
        if self.previewing:
            return

        self._pause()
        self.pending_marker_frame = self.current_frame_idx
        self._set_cancel_enabled(True)
        self._update_status(
            f"Marked frame {self.current_frame_idx}.\n"
            "Previewing clip…")
        self._start_preview()

    # ─── Preview Playback (looping) ───────────────────────────────────────────

    def _start_preview(self):
        """Pre-load all clip frames, then loop them."""
        try:
            clip_dur = float(self.dur_var.get())
            if clip_dur <= 0:
                raise ValueError
        except ValueError:
            clip_dur = DEFAULT_CLIP_DURATION

        mark_frame = self.pending_marker_frame
        num_frames = int(clip_dur * self.fps)
        start_frame = max(0, mark_frame - num_frames + 1)

        self._update_status("Loading preview frames…")
        self.update_idletasks()

        # Pre-load all frames for smooth looping
        self._preview_frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for i in range(start_frame, mark_frame + 1):
            ret, frame = self.cap.read()
            if not ret:
                break
            photo = self._frame_to_photo(frame)
            self._preview_frames.append((i, frame, photo))

        if not self._preview_frames:
            self._update_status("Failed to load preview frames.")
            return

        self.previewing = True
        self._preview_idx = 0
        self._preview_start_time = time.perf_counter()

        # Show confirm button
        self.confirm_btn.pack(padx=12, pady=(0, 4))

        self._update_status(
            f"Previewing {clip_dur}s clip (looping).\n"
            "Press Enter or click Confirm when satisfied.\n"
            "Press Esc to cancel.")

        self._preview_loop()

    def _preview_loop(self):
        """Play pre-loaded frames in a loop."""
        if not self.previewing or not self._preview_frames:
            return

        elapsed = time.perf_counter() - self._preview_start_time
        self._preview_idx = int(elapsed * self.fps) % len(self._preview_frames)
        idx, frame, photo = self._preview_frames[self._preview_idx]
        self.current_frame_idx = idx
        self.current_frame = frame
        self._display_frame_fast(photo)
        self._update_timeline()

        # Show looping overlay text
        cw = self.canvas.winfo_width()
        self.canvas.delete("overlay_text")
        self.canvas.create_text(cw // 2, 25,
                                text="Previewing clip (looping) — Enter to confirm, Esc to cancel",
                                fill="#00ccff", font=("Helvetica", 14, "bold"),
                                tags="overlay_text")

        delay = max(1, int(1000 / self.fps))
        self.preview_after_id = self.after(delay, self._preview_loop)

    def _confirm_preview(self):
        """User confirmed the clip — show the label dialog."""
        if not self.previewing:
            return

        # Stop the preview loop
        self.previewing = False
        if self.preview_after_id is not None:
            self.after_cancel(self.preview_after_id)
            self.preview_after_id = None

        self.confirm_btn.pack_forget()

        # Grab first frame (for setter position marking)
        first_frame = None
        if self._preview_frames:
            _first_idx, first_frame, _first_photo = self._preview_frames[0]
            last_idx, last_frame, last_photo = self._preview_frames[-1]
            self.current_frame_idx = last_idx
            self.current_frame = last_frame
            self._display_frame_fast(last_photo)

        self._preview_frames = []  # free memory
        self._preview_start_time = None

        if first_frame is not None:
            self._show_label_dialog(first_frame)
        else:
            self._cancel_mark()

    def _cancel_preview(self):
        """Cancel the preview and return to normal navigation."""
        self.previewing = False
        if self.preview_after_id is not None:
            self.after_cancel(self.preview_after_id)
            self.preview_after_id = None
        self._preview_frames = []
        self._preview_start_time = None
        self.confirm_btn.pack_forget()
        self.pending_marker_frame = None
        self._set_cancel_enabled(False)

        if self.current_frame is not None:
            self._seek_frame(self.current_frame_idx)
        self._update_status("Cancelled. Navigate and press M to mark a set.")

    def _cancel_mark(self):
        """Cancel the current mark and return to normal navigation."""
        self.pending_marker_frame = None
        self._set_cancel_enabled(False)
        self.confirm_btn.pack_forget()
        self._update_status("Cancelled.")
        if self.current_frame is not None:
            self._display_frame(self.current_frame)

    # ─── Label Dialog ─────────────────────────────────────────────────────────

    def _show_label_dialog(self, first_frame):
        """Show a dialog asking for setter position (click) + set direction."""
        preview_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

        max_preview = 700
        fh, fw = preview_rgb.shape[:2]
        pscale = min(max_preview / fw, max_preview / fh, 1.0)
        disp_w, disp_h = int(fw * pscale), int(fh * pscale)
        preview_resized = cv2.resize(preview_rgb, (disp_w, disp_h))

        dlg = tk.Toplevel(self)
        dlg.title("Label Set Direction & Setter Position")
        dlg.configure(bg="#2d2d2d")
        dlg.transient(self)
        dlg.grab_set()
        dlg.resizable(False, False)

        tk.Label(dlg, text="First frame — click on the setter",
                 font=("Helvetica", 13, "bold"),
                 bg="#2d2d2d", fg="white").pack(padx=12, pady=(12, 4))

        # ── Clickable canvas for setter position ──
        img = Image.fromarray(preview_resized)
        photo = ImageTk.PhotoImage(image=img)

        img_canvas = tk.Canvas(dlg, width=disp_w, height=disp_h,
                               bg="#1e1e1e", highlightthickness=0,
                               cursor="crosshair")
        img_canvas.pack(padx=12, pady=4)
        img_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        img_canvas.image = photo  # prevent GC

        setter_pos = [None]     # mutable; will hold (frame_x, frame_y)
        marker_ids = []         # canvas item ids for the crosshair

        def on_canvas_click(event):
            # Remove previous marker
            for mid in marker_ids:
                img_canvas.delete(mid)
            marker_ids.clear()

            x, y = event.x, event.y
            # Clamp to image bounds
            x = max(0, min(x, disp_w - 1))
            y = max(0, min(y, disp_h - 1))

            # Draw crosshair + circle
            r = 7
            marker_ids.append(img_canvas.create_oval(
                x - r, y - r, x + r, y + r,
                outline="#ff3333", width=2))
            marker_ids.append(img_canvas.create_line(
                x - r - 4, y, x + r + 4, y, fill="#ff3333", width=2))
            marker_ids.append(img_canvas.create_line(
                x, y - r - 4, x, y + r + 4, fill="#ff3333", width=2))

            # Convert display coords → full-frame pixel coords
            frame_x = int(round(x / pscale))
            frame_y = int(round(y / pscale))
            setter_pos[0] = (frame_x, frame_y)
            setter_label.config(
                text=f"Setter marked at ({frame_x}, {frame_y})",
                fg="#33ff66")

        img_canvas.bind("<Button-1>", on_canvas_click)

        setter_label = tk.Label(dlg, text="⬆ Click on the setter above (required)",
                                font=("Helvetica", 10), bg="#2d2d2d", fg="#ffcc00")
        setter_label.pack(padx=12, pady=(0, 4))

        # ── Set direction radio buttons ──
        tk.Label(dlg, text="Set Direction (required):",
                 font=("Helvetica", 11), bg="#2d2d2d", fg="white"
                 ).pack(padx=12, pady=(8, 2), anchor="w")

        dir_var = tk.StringVar(value="")
        for d in SET_DIRECTIONS:
            tk.Radiobutton(dlg, text=d, variable=dir_var, value=d,
                           bg="#2d2d2d", fg="white", selectcolor="#444",
                           activebackground="#2d2d2d", activeforeground="white",
                           font=("Helvetica", 11)
                           ).pack(padx=24, anchor="w")

        # ── Action buttons ──
        btn_frame = tk.Frame(dlg, bg="#2d2d2d")
        btn_frame.pack(padx=12, pady=12, fill=tk.X)

        def on_save():
            direction = dir_var.get()
            if not direction:
                messagebox.showwarning("Label required",
                                       "Please select a set direction before saving.",
                                       parent=dlg)
                return
            if setter_pos[0] is None:
                messagebox.showwarning("Setter position required",
                                       "Please click on the setter in the image above.",
                                       parent=dlg)
                return
            dlg.destroy()
            self._save_label(direction,
                             setter_pos[0][0], setter_pos[0][1])

        def on_cancel():
            dlg.destroy()
            self._cancel_mark()

        tk.Button(btn_frame, text="Save", command=on_save,
                  bg="#27ae60", fg="white", font=("Helvetica", 11, "bold"),
                  width=10).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(btn_frame, text="Cancel", command=on_cancel,
                  font=("Helvetica", 11), width=10).pack(side=tk.LEFT)

        dlg.bind("<Escape>", lambda e: on_cancel())

        dlg.update_idletasks()
        dw, dh = dlg.winfo_width(), dlg.winfo_height()
        px, py = self.winfo_x(), self.winfo_y()
        pw2, ph2 = self.winfo_width(), self.winfo_height()
        dlg.geometry(f"+{px + (pw2 - dw) // 2}+{py + (ph2 - dh) // 2}")

    # ─── Label Saving ─────────────────────────────────────────────────────────

    def _save_label(self, direction, setter_x, setter_y):
        """Append a row to the labels CSV (no clip video is saved)."""
        try:
            clip_dur = float(self.dur_var.get())
            if clip_dur <= 0:
                raise ValueError
        except ValueError:
            clip_dur = DEFAULT_CLIP_DURATION
            self.dur_var.set(str(clip_dur))

        mark_frame = self.pending_marker_frame
        num_frames = int(clip_dur * self.fps)
        start_frame = max(0, mark_frame - num_frames + 1)
        end_frame = mark_frame

        clip_id = uuid.uuid4().hex[:10]

        with open(self.master_csv_path, "a", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                clip_id,
                os.path.basename(self.video_path),
                mark_frame,
                start_frame,
                end_frame,
                clip_dur,
                direction,
                setter_x,
                setter_y,
                datetime.now().isoformat(timespec="seconds")
            ])

        # Reset state
        self.pending_marker_frame = None
        self._set_cancel_enabled(False)

        self.markers.append((mark_frame, clip_id, direction))
        frame_time = self._fmt_time(mark_frame / self.fps)
        self.marker_listbox.insert(
            tk.END,
            f"{clip_id}  |  {frame_time}  |  {direction}")

        self._update_status(
            f"Saved: {clip_id} ({direction})\n"
            f"frames {start_frame}–{end_frame}\n"
            f"setter at ({setter_x}, {setter_y})")

        self._seek_frame(mark_frame)

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _update_status(self, msg):
        self.status_label.config(text=msg)

    # ─── Quit / Cleanup ───────────────────────────────────────────────────────

    def _quit(self):
        if self.markers:
            if not messagebox.askyesno("Quit",
                                       f"{len(self.markers)} labels saved this session.\n"
                                       "Are you sure you want to quit?"):
                return
        self.destroy()

    def destroy(self):
        self.playing = False
        self.previewing = False
        self._stop_decode_thread()
        if self.after_id is not None:
            self.after_cancel(self.after_id)
        if self.preview_after_id is not None:
            self.after_cancel(self.preview_after_id)
        if self.cap is not None:
            self.cap.release()
        super().destroy()


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = LabellingTool()
    app.mainloop()
