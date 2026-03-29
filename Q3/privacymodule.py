import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F_audio
import numpy as np


GENDER_PITCH_FACTORS = {
    ("M", "F"): 1.25,
    ("F", "M"): 0.80,
    ("M", "M"): 1.0,
    ("F", "F"): 1.0,
}


class SpectralEnvelopeShift(nn.Module):
    def __init__(self, n_fft=512, hop_length=128):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, waveform, shift_factor):
        if abs(shift_factor - 1.0) < 1e-6:
            return waveform

        spec = torch.stft(
            waveform.squeeze(0),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True,
        )

        magnitude = spec.abs()
        phase = spec.angle()

        freq_bins = magnitude.shape[0]
        new_magnitude = torch.zeros_like(magnitude)

        for i in range(freq_bins):
            src = i / shift_factor
            lo = int(src)
            hi = lo + 1
            frac = src - lo
            if lo < 0:
                continue
            if hi >= freq_bins:
                if lo < freq_bins:
                    new_magnitude[i] = magnitude[lo] * (1 - frac)
                continue
            new_magnitude[i] = magnitude[lo] * (1 - frac) + magnitude[hi] * frac

        shifted_spec = torch.polar(new_magnitude, phase)
        out = torch.istft(
            shifted_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            length=waveform.shape[-1],
        )
        return out.unsqueeze(0)


class PitchShiftModule(nn.Module):
    def __init__(self, sample_rate=16000):
        super().__init__()
        self.sample_rate = sample_rate

    def forward(self, waveform, pitch_factor):
        if abs(pitch_factor - 1.0) < 1e-6:
            return waveform

        orig_len = waveform.shape[-1]
        new_freq = int(self.sample_rate * pitch_factor)

        resampled = torchaudio.functional.resample(waveform, self.sample_rate, new_freq)
        restored = torchaudio.functional.resample(resampled, new_freq, self.sample_rate)

        if restored.shape[-1] > orig_len:
            restored = restored[..., :orig_len]
        elif restored.shape[-1] < orig_len:
            pad = orig_len - restored.shape[-1]
            restored = torch.nn.functional.pad(restored, (0, pad))

        return restored


class PrivacyPreservingModule(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=512, hop_length=128):
        super().__init__()
        self.sample_rate = sample_rate
        self.pitch_shifter = PitchShiftModule(sample_rate)
        self.formant_shifter = SpectralEnvelopeShift(n_fft, hop_length)
        self.pitch_factors = GENDER_PITCH_FACTORS

    def forward(self, waveform, src_gender, tgt_gender):
        src_gender = src_gender.upper()
        tgt_gender = tgt_gender.upper()

        pitch_factor = self.pitch_factors.get((src_gender, tgt_gender), 1.0)
        formant_factor = 1.0 + (pitch_factor - 1.0) * 0.5

        out = self.pitch_shifter(waveform, pitch_factor)
        out = self.formant_shifter(out, formant_factor)

        rms_in = waveform.pow(2).mean().sqrt().clamp(min=1e-8)
        rms_out = out.pow(2).mean().sqrt().clamp(min=1e-8)
        out = out * (rms_in / rms_out)

        return out

    def transform(self, waveform, src_gender, tgt_gender):
        with torch.no_grad():
            return self.forward(waveform, src_gender, tgt_gender)


def load_audio(path, target_sr=16000):
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32")
    waveform = torch.from_numpy(data)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform, target_sr


def save_audio(waveform, path, sample_rate=16000):
    import soundfile as sf
    data = waveform.squeeze(0).numpy()
    sf.write(str(path), data, sample_rate)
