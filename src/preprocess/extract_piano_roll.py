import torch
import pretty_midi
import numpy as np
import os


class YourDatasetClass:
    def _midi2piano_roll(self, filename, filename2=None, mix_lambda=-1):
        # Initialize pianoroll
        pianoroll = torch.zeros([512, 128]) + 0.01

        # Load MIDI file
        try:
            pm = pretty_midi.PrettyMIDI(filename)
            pianoroll = pm.get_piano_roll(fs=51.2)
            pianoroll = pianoroll.T
            pianoroll = torch.from_numpy(pianoroll).float()
        except Exception as e:  # It's better to catch specific exceptions
            print(f"there is a loading error: {e}")

        target_length = self.target_length
        n_frames = pianoroll.shape[0]
        p = target_length - n_frames

        # Cut and pad if different length
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            pianoroll = m(pianoroll)
        elif p < 0:
            max_index = n_frames - target_length
            random_index = torch.randint(0, max_index, (1,))  # for random crop
            pianoroll = pianoroll[random_index : random_index + target_length, :]

        return pianoroll

    # TODO: implement saving and loading files
    def preprocess_and_save(self, midi_filenames, save_dir):
        for filename in midi_filenames:
            pianoroll, time_mapping = self._midi2piano_roll(filename)
            save_path = os.path.join(save_dir, os.path.basename(filename) + ".pt")
            torch.save(
                {"pianoroll": pianoroll, "time_mapping": time_mapping}, save_path
            )
