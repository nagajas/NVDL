import torch
import numpy as np
import cv2
import os
from C3D import C3D
import torch.nn as nn

torch.backends.cudnn.benchmark = True

def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)

class C3DFeatureExtractor(nn.Module):
    def _init_(self, pretrained_model, output_dim=128):
        super(C3DFeatureExtractor, self)._init_()
        self.features = nn.Sequential(*list(pretrained_model.children())[:-1])
        self.reduce_dim = nn.Linear(4096, output_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.reduce_dim(x)
        return x

def process_video(video_path, feature_extractor, device):
    cap = cv2.VideoCapture(video_path)
    retaining = True
    clip = []
    features_list = []

    while retaining:
        retaining, frame = cap.read()
        if not retaining and frame is None:
            continue

        tmp_ = center_crop(cv2.resize(frame, (171, 128)))
        tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
        clip.append(tmp)

        if len(clip) == 16:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs).to(device)
            with torch.no_grad():
                features = feature_extractor(inputs)

            features_list.append(features.cpu().numpy())
            clip.pop(0)

    cap.release()
    return np.vstack(features_list)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    model = C3D(num_classes=7)
    checkpoint = torch.load('./c3d-pretrained.pth', map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device).eval()

    feature_extractor = C3DFeatureExtractor(model, output_dim=128)
    feature_extractor.to(device).eval()

    dataset_dir = '../../data/all_videos_and_audios'
    mp4_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.mp4')]
    
    all_features = []
    for video_file in mp4_files:
        print(f"Processing video: {video_file}")
        features = process_video(video_file, feature_extractor, device)
        all_features.append(features)

    all_features = np.vstack(all_features)
    np.save('video_features.npy', all_features)
    print("All features saved to 'video_embs.npy'")

if _name_ == '_main_':
    main()