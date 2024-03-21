import torch
from vocos import Vocos
import torchaudio
import numpy as np

vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz")

numpy_array = np.loadtxt("audio_tokens.txt")
audio_tokens = torch.tensor(numpy_array, dtype=torch.long)
features = vocos.codes_to_features(audio_tokens)
bandwidth_id = torch.tensor([2])

audio = vocos.decode(features, bandwidth_id=bandwidth_id)


torchaudio.save('test_3.wav',
                  audio[0].unsqueeze(0),
                  sample_rate=16000)