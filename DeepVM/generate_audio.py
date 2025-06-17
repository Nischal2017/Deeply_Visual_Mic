from scipy.io import wavfile
import numpy as np
import torch
import matplotlib.pyplot as plt

def save_spectrogram(sound_data, sampling_rate, filename='spectrogram.png'):
    # Plot a spectrogram of the sound data
    plt.figure()
    plt.specgram(sound_data, Fs=sampling_rate, cmap=plt.get_cmap('jet'))
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar().set_label('Power Spectral Density (dB)')

    # Save the figure to a file
    plt.savefig(filename)
    plt.close()  # Close the figure to free up memory

audio_tensor = torch.load('./tensor_data.pt')
audio_array = audio_tensor.cpu().numpy()

print(np.squeeze(audio_array).shape)
print(set(np.squeeze(audio_array)))
save_spectrogram(np.squeeze(audio_array), 2200)
wavfile.write('output1.wav', 2200, audio_array)