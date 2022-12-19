from dataclasses import dataclass
import pyaudio
import tkinter as Tk
from tkinter import HORIZONTAL, LEFT, RIGHT, BOTTOM, TOP, ttk
import librosa
import numpy as np
import threading
import time
from collections import defaultdict
from synth.components.composers import WaveAdder
from synth.components.envelopes import ADSREnvelope
from synth.components.oscillators import (
    SineOscillator,
    SquareOscillator,
    SawtoothOscillator,
    TriangleOscillator,
    ModulatedOscillator,
)
from synth.components.oscillators.base_oscillator import Oscillator

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

BUFFER_SIZE = 2**13
SAMPLE_RATE = 44100
NOTE_AMP = 0.1
PLOT_SAMPLE_PERIOD = 10

key_to_note = {
    "z": "c4",
    "s": "c#4",
    "x": "d4",
    "d": "d#4",
    "c": "e4",
    "v": "f4",
    "g": "f#4",
    "b": "g4",
    "h": "g#4",
    "n": "a4",
    "j": "a#4",
    "m": "b4",
    ",": "c5",
    "q":"c#5"
}


def is_modulator(osc_ind):
    return osc_ind % 2 == 1


def get_signal_function(osc_states, adsr_state):
    f = lambda idx_c, idx_m, note: ModulatedOscillator(
        osc_states[idx_c].waveform(
            freq=librosa.note_to_hz(note),
            amp=osc_states[idx_c].amp,
            phase=osc_states[idx_c].phase,
        ),
        ADSREnvelope(
            attack_duration=adsr_state.attack,
            decay_duration=adsr_state.decay,
            sustain_level=adsr_state.sustain,
            release_duration=adsr_state.release,
        ),
        osc_states[idx_m].waveform(
            freq=librosa.note_to_hz(note) * osc_states[idx_m].ratio
            if osc_states[idx_m].use_ratio
            else osc_states[idx_m].freq,
            amp=osc_states[idx_m].amp,
            phase=osc_states[idx_m].phase,
        ),
        amp_mod=lambda x, y: x * y,
        freq_mod=lambda x, y: x + y,
    )

    return lambda note: iter(WaveAdder(f(0, 1, note), f(2, 3, note)))


def update_signal():
    global osc_states
    global adsr_state
    global notes_dict
    global signal_function
    global line
    global canvas

    if osc_states and adsr_state:
        signal_function = get_signal_function(osc_states, adsr_state)
        for note in notes_dict:
            notes_dict[note][0] = signal_function(note)


def get_samples(notes_dict):
    # Return samples in int16 format
    samples = []
    for _ in range(BUFFER_SIZE):
        samples.append([next(signal[0]) for _, signal in notes_dict.items()])
    samples = np.array(samples).sum(axis=1) * 0.3
    samples = np.int16(samples.clip(-0.8, 0.8) * 32767)
    return samples.reshape(BUFFER_SIZE, -1)


def handle_key_event(event_type, ch):
    # print(event_type, ch)
    global signal_function
    global notes_dict
    note = key_to_note[ch]
    if event_type == "press":
        notes_dict[note] = [signal_function(note), False]
    elif event_type == "release":
        notes_dict[note][0].trigger_release()
        notes_dict[note][1] = True


class KeyTracker:
    last_press_time = defaultdict(int)
    cnt = defaultdict(int)

    def is_pressed(self, ch):
        return time.time() - self.last_press_time[ch] < 0.1

    def report_key_press(self, event):
        if event.char == "q":
            root.destroy()
        if not self.is_pressed(event.char) and self.cnt[event.char] == 0:
            handle_key_event("press", event.char)
        self.cnt[event.char] += 1
        self.last_press_time[event.char] = time.time()

    def report_key_release(self, event):
        timer = threading.Timer(0.1, self.report_key_release_callback, args=[event])
        timer.start()

    def report_key_release_callback(self, event):
        self.cnt[event.char] -= 1
        if not self.is_pressed(event.char) and self.cnt[event.char] == 0:
            handle_key_event("release", event.char)


@dataclass(frozen=True)
class OscState:
    waveform: Oscillator
    amp: float
    phase: float
    ratio: float
    freq: float
    use_ratio: bool


class SingleOsc:
    waveform_to_osc = {
        "sine": SineOscillator,
        "square": SquareOscillator,
        "sawtooth": SawtoothOscillator,
        "triangle": TriangleOscillator,
    }
    waveform_options = list(waveform_to_osc.keys())

    def __init__(self, parent, number, state_change_callback):
        self.number = number
        self.is_modulator = is_modulator(number)
        self.column_no = number // 2

        self.frame = ttk.Labelframe(
            parent,
            text=("modulator" if self.is_modulator else "carrier")
            + " "
            + str(self.column_no + 1),
            width=200,
            height=300,
        )

        self.waveform = Tk.StringVar(parent, "sine")

        self.waveform_frame = ttk.Labelframe(self.frame, text="waveform")
        for w in SingleOsc.waveform_options:
            ttk.Radiobutton(
                self.waveform_frame,
                text=w,
                variable=self.waveform,
                value=w,
            ).pack()

        self.amp = Tk.DoubleVar(parent, 100 if self.is_modulator else 0.5)
        Tk.Scale(
            self.frame,
            label="amplitude",
            variable=self.amp,
            from_=0,
            to=300 if self.is_modulator else 2,
            orient=HORIZONTAL,
            resolution=1 if self.is_modulator else 0.1,
        ).pack()

        self.phase = Tk.DoubleVar(parent, 0)
        Tk.Scale(
            self.frame,
            label="phase",
            variable=self.phase,
            from_=0,
            to=360,
            orient=HORIZONTAL,
            resolution=20,
        ).pack()

        if self.is_modulator:
            self.ratio = Tk.DoubleVar(parent, 0.1)
            Tk.Scale(
                self.frame,
                label="ratio",
                variable=self.ratio,
                from_=0.01,
                to=10,
                orient=HORIZONTAL,
                resolution=0.01,
            ).pack()
            self.freq = Tk.DoubleVar(parent, 5)
            Tk.Scale(
                self.frame,
                label="frequency",
                variable=self.freq,
                from_=1,
                to=100,
                orient=HORIZONTAL,
                resolution=1,
            ).pack()
            self.use_ratio = Tk.DoubleVar(parent, False)
            Tk.Checkbutton(
                self.frame,
                text="use ratio",
                variable=self.use_ratio,
                onvalue=True,
                offvalue=False,
            ).pack()

        fn = lambda a, b, c: state_change_callback(self.number, self.get_state())
        self.waveform.trace("w", fn)
        self.amp.trace("w", fn)
        self.phase.trace("w", fn)
        if self.is_modulator:
            self.ratio.trace("w", fn)
            self.freq.trace("w", fn)
            self.use_ratio.trace("w", fn)

        self.waveform_frame.pack()
        self.frame.grid(row=0 if self.is_modulator else 1, column=self.column_no)

    def get_state(self):
        return OscState(
            SingleOsc.waveform_to_osc[self.waveform.get()],
            self.amp.get(),
            self.phase.get(),
            *self.get_modulator_state(),
        )

    def get_modulator_state(self):
        if self.is_modulator:
            return self.ratio.get(), self.freq.get(), self.use_ratio.get()
        else:
            return None, None, None


class MultiOsc:
    def __init__(self, parent, state_change_callback):
        oscs = [SingleOsc(parent, x, self.update_state) for x in range(4)]

        self.state_change_callback = state_change_callback

        self.osc_states = [osc.get_state() for osc in oscs]
        state_change_callback(self.osc_states)

    def update_state(self, osc_number, state):
        self.osc_states[osc_number] = state
        self.state_change_callback(self.osc_states)


@dataclass(frozen=True)
class ADSRState:
    attack: float
    decay: float
    sustain: float
    release: float

class ADSR:
    def __init__(self, parent, state_change_callback):
        self.state_change_callback = state_change_callback

        self.adsr_frame = ttk.Labelframe(
            parent, text="Attack Decay Sustain Release (ADSR) envelope", width=200, height=300)

        self.attack = Tk.DoubleVar(parent, 0.05)
        self.decay = Tk.DoubleVar(parent, 0.2)
        self.sustain = Tk.DoubleVar(parent, 0.7)
        self.release = Tk.DoubleVar(parent, 0.3)
        Tk.Scale(
            self.adsr_frame,
            label="attack",
            variable=self.attack,
            from_=0,
            to=0.2,
            orient=HORIZONTAL,
            resolution=0.05,
        ).pack()
        Tk.Scale(
            self.adsr_frame,
            label="decay",
            variable=self.decay,
            from_=0,
            to=0.5,
            orient=HORIZONTAL,
            resolution=0.1,
        ).pack()
        Tk.Scale(
            self.adsr_frame,
            label="sustain",
            variable=self.sustain,
            from_=0,
            to=1,
            orient=HORIZONTAL,
            resolution=0.1,
        ).pack()
        Tk.Scale(
            self.adsr_frame,
            label="release",
            variable=self.release,
            from_=0,
            to=0.5,
            orient=HORIZONTAL,
            resolution=0.1,
        ).pack()

        fn = lambda a, b, c: self.state_change_callback(self.get_state())
        self.attack.trace("w", fn)
        self.decay.trace("w", fn)
        self.sustain.trace("w", fn)
        self.release.trace("w", fn)
        fn(None, None, None)

        self.adsr_frame.grid(row=0, column=2, rowspan=2)

    def get_state(self):
        return ADSRState(
            self.attack.get(), self.decay.get(), self.sustain.get(), self.release.get()
        )


stream = pyaudio.PyAudio().open(
    rate=SAMPLE_RATE,
    channels=1,
    format=pyaudio.paInt16,
    output=True,
    frames_per_buffer=BUFFER_SIZE,
)

root = Tk.Tk()
root.title("synthesizeme.com")

fig = Figure(figsize=(5*1.5, 4*1.5), dpi=100)
my_plot = fig.add_subplot(1, 1, 1)
my_plot.set_xlim(0, SAMPLE_RATE)
my_plot.set_ylim(-12000, 12000)
my_plot.set_xticks(np.arange(0, SAMPLE_RATE, 10000))
my_plot.set_ylabel('Amplitude')
my_plot.set_xlabel('Time')
my_plot.set_title("Current Wave Being Played")
xs = SAMPLE_RATE / BUFFER_SIZE * np.arange(0, BUFFER_SIZE, PLOT_SAMPLE_PERIOD)
(line,) = my_plot.plot(xs, [0] * len(xs), color="blue")
line.set_xdata(xs)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
W1 = canvas.get_tk_widget()
W1.grid(row=0, column=3, rowspan=2)

key_tracker = KeyTracker()
for key in key_to_note:
    root.bind(f"<KeyPress-{key}>", key_tracker.report_key_press)
    root.bind(f"<KeyRelease-{key}>", key_tracker.report_key_release)

signal_function = None
notes_dict = {}
adsr_state = None
osc_states = None


def update_adsr_state(new_adsr_state):
    global adsr_state
    print("Attack Decay Sustain Relsease (adsr) state changed", new_adsr_state)
    adsr_state = new_adsr_state
    update_signal()


def update_osc_states(new_osc_states):
    global osc_states
    print("oscillator states changed", new_osc_states)
    osc_states = new_osc_states
    update_signal()


adsr = ADSR(root, update_adsr_state)
osc = MultiOsc(root, update_osc_states)

print("Press q when you want to exit")

try:
    while True:
        root.update()
        if notes_dict:
            samples = get_samples(notes_dict)
            ys = samples[::PLOT_SAMPLE_PERIOD]
            stream.write(samples.tobytes())
            ended_notes = [k for k, o in notes_dict.items() if o[0].ended and o[1]]
            for note in ended_notes:
                del notes_dict[note]
        else:
            ys = [0] * len(xs)
        line.set_ydata(ys)
        canvas.draw()

except KeyboardInterrupt as err:
    stream.close()
