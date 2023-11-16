import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import filedialog
from scipy.io.wavfile import read
from pydub import AudioSegment
import os
import threading

class SpectrumAnalyzer(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Spectrum Analyzer")
        
        self.filename = tk.StringVar()
        self.create_widgets()
        
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def create_widgets(self):
        tk.Label(self, text="MP3 File:").pack(padx=20, pady=(20,0))
        
        tk.Entry(self, textvariable=self.filename, state="readonly", width=50).pack(padx=20, pady=(0,10))
        
        tk.Button(self, text="Open", command=self.load_file).pack(pady=(0,20))
        
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        
        self.anim = None
        
    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("MP3 files", "*.mp3")])
        if file_path:
            self.filename.set(file_path)
            self.update_spectrum()
            
    def update_spectrum(self):
        filename = self.convert_mp3_to_wav(self.filename.get())
        self.samplerate, self.data = read(filename)
        self.times = np.arange(len(self.data))/float(self.samplerate)
        
        downsample_factor = 20
        self.data = self.data[::downsample_factor]
        self.times = self.times[::downsample_factor]
        
        self.ax.clear()
        self.line, = self.ax.plot([], [], lw=2)
        self.ax.set_xlim(0, np.max(self.times))
        self.ax.set_ylim(np.min(self.data), np.max(self.data))
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Amplitude')
        
        # 设置 repeat=False 防止动画完成后重新开始
        self.anim = FuncAnimation(self.fig, self.animate, init_func=self.init, frames=len(self.times)//1, interval=0.01, blit=True, repeat=False)
        self.canvas.draw()

    def convert_mp3_to_wav(self, mp3_filename):
        wav_filename = mp3_filename.replace(".mp3", ".wav")
        if not os.path.exists(wav_filename):
            audio = AudioSegment.from_mp3(mp3_filename)
            audio.export(wav_filename, format="wav")
        return wav_filename
    
    def init(self):
        self.line.set_data([], [])
        return (self.line,)

    def animate(self, i):
        self.line.set_data(self.times[:i*4], self.data[:i*4])
        return (self.line,)
    
    def on_closing(self):
        if tk.messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.destroy()
            
if __name__ == "__main__":
    def run_app():
        app = SpectrumAnalyzer()
        app.mainloop()

    thread = threading.Thread(target=run_app)
    thread.start()
