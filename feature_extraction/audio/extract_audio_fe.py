import os
import numpy as np
import json
import subprocess
import argparse

def extract_features(audio_file_path, config_file_path='opensmile/config/is09-13/IS13_ComParE.conf'):
    out = audio_file_path.split('/')[-1].replace('.wav', '_features.txt')
    output_file_path = 'features/' + out

    os.makedirs('features', exist_ok=True)

    command = [
        'opensmile/build/progsrc/smilextract/SMILExtract',
        '-C', config_file_path,
        '-I', audio_file_path,
        '-O', output_file_path
    ]
    
    subprocess.run(command, check=True)

    raw_features = []
    with open(output_file_path) as f:
        for line in f:
            if line.startswith('@') or line == '\n':
                continue
            raw_features = line.strip().split(',')
            raw_features = [float(i) for i in raw_features[1:-1]]

    return raw_features

def process_audio_files(audio_dir, num_utterances):
    feature_list = []
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]

    for idx, audio_file in enumerate(audio_files):
        if idx >= num_utterances:
            break
        audio_file_path = os.path.join(audio_dir, audio_file)
        features = extract_features(audio_file_path)
        feature_list.append(features)

    return np.array(feature_list)

def main():
    parser = argparse.ArgumentParser(description='Extract audio features from .wav files and save to .npy file.')
    parser.add_argument('num_utterances', type=int, help='Number of utterances to process', default=None)   
    args = parser.parse_args()

    data_path = '../../data/'
    audio_dir = os.path.join(data_path, 'all_videos_and_audios/train/')

    num_utt = 0
    for f in os.listdir(audio_dir):
        if f.endswith('.wav'):
            num_utt += 1

    if args.num_utterances is None:
        args.num_utterances = num_utt
    
    features_array = process_audio_files(audio_dir, args.num_utterances)

    output_npy_file = 'audio_embs.npy'
    np.save(output_npy_file, features_array)

    print(f'Successfully extracted features for {features_array.shape[0]} utterances and saved to {output_npy_file}')

    os.remove('features', exist_ok=True) # Clean up

if __name__ == '__main__':
    main()
