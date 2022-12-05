import pyaudio
import tkinter as Tk
from tkinter import HORIZONTAL, ttk   	
import librosa
import numpy as np
import threading
import time
from collections import defaultdict
from synth.components.envelopes import ADSREnvelope
from synth.components.oscillators import SineOscillator, SquareOscillator, SawtoothOscillator, TriangleOscillator, ModulatedOscillator
# from matplotlib.figure import Figure 
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

BUFFER_SIZE = 256
SAMPLE_RATE = 44100
NOTE_AMP = 0.1

key_to_note = {
    'z': 'c4', 
    's': 'c#4', 
    'x': 'd4', 
    'd': 'd#4', 
    'c': 'e4', 
    'v': 'f4', 
    'g': 'f#4', 
    'b': 'g4',
    'h': 'g#4',
    'n': 'a4',
    'j': 'a#4',
    'm': 'b4',
    ',': 'c5',
}

name_to_osc = {
    'sine':  SineOscillator,
    'square': SquareOscillator,
    'sawtooth': SawtoothOscillator,
    'triangle': TriangleOscillator
}
osc_options = list(name_to_osc.keys())

def get_osc_function(osc, attack, decay, sustain, release):
    return (
        lambda note: iter(
            ModulatedOscillator(
                osc(freq=librosa.note_to_hz(note)),
                ADSREnvelope(attack_duration=attack, decay_duration=decay, sustain_level=sustain, release_duration=release),
                amp_mod=lambda x,y:x*y
            )
        )
    )

def get_samples(notes_dict):
    # Return samples in int16 format
    samples = []
    for _ in range(BUFFER_SIZE):
        samples.append([next(osc[0]) for _, osc in notes_dict.items()])
    samples = np.array(samples).sum(axis=1) * 0.3
    samples = np.int16(samples.clip(-0.8, 0.8) * 32767)
    return samples.reshape(BUFFER_SIZE, -1)


def handle_key_event(event_type, ch):
    print(event_type, ch)
    global osc_function
    global notes_dict
    note = key_to_note[ch]
    if event_type == 'press':
        notes_dict[note] = [osc_function(note), False]
    elif event_type == 'release':        
        notes_dict[note][0].trigger_release()
        notes_dict[note][1] = True

class KeyTracker():
    last_press_time = defaultdict(int)
    cnt = defaultdict(int)

    def is_pressed(self, ch):
        return time.time() - self.last_press_time[ch] < .1

    def report_key_press(self, event):
        if not self.is_pressed(event.char) and self.cnt[event.char] == 0:
            handle_key_event('press', event.char)
        self.cnt[event.char] += 1
        self.last_press_time[event.char] = time.time()

    def report_key_release(self, event):
        timer = threading.Timer(.1, self.report_key_release_callback, args=[event])
        timer.start()

    def report_key_release_callback(self, event):
        self.cnt[event.char] -= 1
        if not self.is_pressed(event.char) and self.cnt[event.char] == 0:
            handle_key_event('release', event.char)


def update_osc(*args):
    global cur_osc_str
    global cur_attack
    global cur_decay
    global cur_sustain
    global cur_release
    global notes_dict
    global osc_function

    osc = name_to_osc[cur_osc_str.get()]
    osc_function = get_osc_function(osc, cur_attack.get(), cur_decay.get(), cur_sustain.get(), cur_release.get())
    
    for note in notes_dict:
        notes_dict[note][0] = osc_function(note)
    
root = Tk.Tk()

osc_frame = ttk.Labelframe(root, text='oscillator', width=200, height=300)
cur_osc_str = Tk.StringVar(root, "sine")
waveform_frame = ttk.Labelframe(osc_frame, text='waveform')
for osc_name in osc_options:
    ttk.Radiobutton(waveform_frame, text=osc_name, variable=cur_osc_str, value=osc_name).pack()
waveform_frame.pack()
osc_frame.pack()

adsr_frame = ttk.Labelframe(root, text='adsr envelope', width=200, height=300)
cur_attack = Tk.DoubleVar(root, 0.05)
cur_decay = Tk.DoubleVar(root, 0.2)
cur_sustain = Tk.DoubleVar(root, 0.7)
cur_release = Tk.DoubleVar(root, 0.3)
attack_scale = Tk.Scale(adsr_frame, label='attack', variable=cur_attack, from_=0, to=0.2, orient=HORIZONTAL, resolution=0.05)
decay_scale = Tk.Scale(adsr_frame, label='decay', variable=cur_decay, from_=0, to=0.5, orient=HORIZONTAL, resolution=0.1)
sustain_scale = Tk.Scale(adsr_frame, label='sustain', variable=cur_sustain, from_=0, to=1, orient=HORIZONTAL, resolution=0.1)
release_scale = Tk.Scale(adsr_frame, label='release', variable=cur_release, from_=0, to=0.5, orient=HORIZONTAL, resolution=0.1)
attack_scale.pack()
decay_scale.pack()
sustain_scale.pack()
release_scale.pack()
adsr_frame.pack()

key_tracker = KeyTracker()
for key in key_to_note:
    root.bind(f"<KeyPress-{key}>", key_tracker.report_key_press)
    root.bind(f"<KeyRelease-{key}>", key_tracker.report_key_release)


cur_osc_str.trace("w", update_osc)
cur_attack.trace("w", update_osc)
cur_decay.trace("w", update_osc)
cur_sustain.trace("w", update_osc)
cur_release.trace("w", update_osc)


stream = pyaudio.PyAudio().open(
    rate=SAMPLE_RATE,
    channels=1,
    format=pyaudio.paInt16,
    output=True,
    frames_per_buffer=BUFFER_SIZE,
)


# fig = Figure(figsize=(5, 4), dpi=100)
# my_plot = fig.add_subplot(1, 1, 1)
# my_plot.set_ylim(-32767, 32767)
# xs = SAMPLE_RATE/BUFFER_SIZE * np.arange(0, BUFFER_SIZE, 20)
# line, = my_plot.plot(xs, [0] * len(xs), color = 'blue') 
# line.set_xdata(xs)                       
# canvas = FigureCanvasTkAgg(fig, master = root)
# canvas.draw()
# W1 = canvas.get_tk_widget()
# W1.pack()

try:
    osc_function = get_osc_function(SineOscillator, cur_attack.get(), cur_decay.get(), cur_sustain.get(), cur_release.get())
    notes_dict = {}
    while True:
        root.update()
        
        if notes_dict:
            samples = get_samples(notes_dict)
            # line.set_ydata(samples[::20])
            # canvas.draw()
            stream.write(samples.tobytes())
            # print(notes_dict.keys())
            ended_notes = [
                k for k, o in notes_dict.items() if o[0].ended and o[1]
            ]
            for note in ended_notes:
                del notes_dict[note]
except KeyboardInterrupt as err:
    stream.close()


