import os
import cv2
import sys
import random
import argparse
import paddle
import tempfile
import subprocess
import face_alignment
import numpy as np
import os.path as osp
from tqdm import tqdm
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, required=True)
parser.add_argument('--face', type=str, required=True)
parser.add_argument('--audio', type=str, required=True)
parser.add_argument('--max_mel_chunks', type=int, default=None)
parser.add_argument('--save_root', type=str, default='./results')
parser.add_argument('--save_name', type=str, default=None)
parser.add_argument('--fps', type=float, default=25)
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--mask', default="mask3.jpg", type=str)
# run
parser.add_argument('--save_crop', action='store_true')
parser.add_argument('--tag', type=str, default='StyleSync Inference...')
args = parser.parse_args()

import __init_paths
import audio
import utils


def main():
    # face
    temp_face_file = tempfile.NamedTemporaryFile(suffix=".mp4")
    if not os.path.isfile(args.face):
        fnames = list(glob(os.path.join(args.face, '*.jpg')))
        sorted_fnames = sorted(fnames, key=lambda f: int(os.path.basename(f).split('.')[0]))
        full_frames = [cv2.imread(f) for f in sorted_fnames]
        fps = args.fps
    elif args.face.split('.')[-1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps
    elif args.face.split('.')[-1] in ['mp4', 'mov', 'MOV', 'MP4', 'webm']:
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        try:
            assert fps == args.fps
        except:
            print('Converting video to fps 25...')
            video_name = temp_face_file.name
            command = 'ffmpeg -loglevel panic  -i {} -qscale 0 -strict -2 -r {} -y {}'.format(args.face, fps, video_name)
            subprocess.call(command, shell=True)
            video_stream = cv2.VideoCapture(video_name)
        print('Reading video frames...')
        full_frames = []
        while 1:
            print('Reading {}...'.format(len(full_frames)), end='\r')
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            full_frames.append(frame)
            if args.max_mel_chunks and len(full_frames) > args.max_mel_chunks + 10:
                video_stream.release()
                break

    # audio
    temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav")
    if osp.basename(args.audio).split('.')[1] in ['wav', 'mp3']:
        wav_path = args.audio
    elif os.path.basename(args.audio).split('.')[1] in ['mp4', 'avi', 'MP4', 'AVI', 'MOV', 'mov', 'webm']:
        print('Extracting raw audio...')
        audio_name = temp_audio_file.name
        command = 'ffmpeg -i {} -loglevel error -y -f wav -acodec pcm_s16le -ar 16000 {}'.format(args.audio, audio_name)
        subprocess.call(command, shell=True)
        wav_path = audio_name

    # run
    with paddle.no_grad():
        print("Loading model...")
        model = paddle.jit.load(args.checkpoint_path)
        model.eval()

        save_name = args.save_name or '{}_{}.mp4'.format(osp.basename(args.face).split('.')[0], osp.basename(args.audio).split('.')[0])
        save_path = os.path.join(args.save_root, save_name)
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        print("=====>", save_path)
        infer_one(model, full_frames, wav_path, save_path)

    temp_face_file.close()
    temp_audio_file.close()


def infer_one(model, imgs, wav_path, save_path):
    out_video_p = tempfile.NamedTemporaryFile(suffix=".avi")
    crop_out_video_p = tempfile.NamedTemporaryFile(suffix=".avi")
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')
    restorer = utils.AlignRestore()
    lmk_smoother = utils.laplacianSmooth()
    mel_chunks, _ = utils.read_wav(wav_path)
    mel_chunks = np.asarray(mel_chunks[:args.max_mel_chunks])
    img_mask = 1. - cv2.resize(cv2.imread(args.mask), (args.img_size, args.img_size), interpolation=cv2.INTER_AREA) / 255.
    img_idxs_org = list(range(len(imgs)))
    img_idxs_dst = []
    while (len(img_idxs_dst) < len(mel_chunks) + 10):
        img_idxs_dst += img_idxs_org
        img_idxs_org = img_idxs_org[::-1]
    frame_h, frame_w, _ = imgs[0].shape
    out = cv2.VideoWriter(out_video_p.name, cv2.VideoWriter_fourcc(*'DIVX'), 25, (frame_w, frame_h))

    # img prep
    skip_begin = 0
    face_all = []
    box_all = []
    affine_matrix_all = []
    face_crop_data_dict = {}
    print('Run face cropping...')
    for i, m in tqdm(enumerate(mel_chunks), total=len(mel_chunks)):
        img_idx = img_idxs_dst[i]
        if img_idx in face_crop_data_dict:
            _data = face_crop_data_dict[img_idx]
            affine_matrix_all.append(_data['affine_matrix'])
            box_all.append(_data['box'])
            face_all.append(_data['face'])
            continue
        img = imgs[img_idx].copy()
        try:
            re_lmks = fa.get_landmarks(img.copy())
            points = lmk_smoother.smooth(re_lmks[0])
            lmk3_ = np.zeros((3, 2))
            lmk3_[0] = points[17:22].mean(0)
            lmk3_[1] = points[22:27].mean(0)
            lmk3_[2] = points[27:36].mean(0)
        except Exception as e:
            print('Face detection fail...\n[{}]'.format(e))
            if len(affine_matrix_all) == 0:
                skip_begin += 1
                continue
            affine_matrix_all.append(affine_matrix_all[-1])
            box_all.append(box_all[-1])
            face_all.append(face_all[-1])
            continue
        face, affine_matrix = restorer.align_warp_face(img.copy(), lmks3=lmk3_, smooth=True)
        box = [0, 0, face.shape[1], face.shape[0]]
        if i == 0 and args.save_crop:
            out_crop = cv2.VideoWriter(crop_out_video_p.name, cv2.VideoWriter_fourcc(*'DIVX'), 25, (face.shape[1], face.shape[0]))
        face = cv2.resize(face, (args.img_size, args.img_size), interpolation=cv2.INTER_CUBIC)
        affine_matrix_all.append(affine_matrix)
        box_all.append(box)
        face_all.append(face)
        face_crop_data_dict[img_idx] = {'affine_matrix': affine_matrix, 'box': box, 'face': face}
    while len(face_all) < len(mel_chunks):
        assert skip_begin > 0
        affine_matrix_all += affine_matrix_all[::-1]
        box_all += box_all[::-1]
        face_all += face_all[::-1]

    print('Run generation...')
    for i, m in tqdm(enumerate(mel_chunks), total=len(mel_chunks)):
        img = imgs[img_idxs_dst[i]].copy()
        face = face_all[i].copy()
        box = box_all[i]
        affine_matrix = affine_matrix_all[i]
        face_masked = face.copy() * img_mask
        ref_face = face.copy()

        # infer
        ref_faces = ref_face[np.newaxis, :]
        face_masked = face_masked[np.newaxis, :]
        mel_batch = mel_chunks[[max(0, x) for x in range(i - 4, i + 1)]]
        mel_batch = np.transpose(mel_batch, (0, 2, 1))
        img_batch = np.concatenate((face_masked, ref_faces), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
        img_batch = paddle.to_tensor(np.transpose(img_batch, (0, 3, 1, 2)), dtype='float32')
        mel_batch = paddle.to_tensor(np.transpose(mel_batch, (0, 3, 1, 2)), dtype='float32')

        pred = model(img_batch, mel_batch)
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1)
        pred = pred.astype(np.uint8)[0]

        # save
        x1, y1, x2, y2 = box
        pred = cv2.resize(pred, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
        if args.save_crop:
            out_crop.write(pred)
        out_img = restorer.restore_img(img, pred, affine_matrix)
        out.write(out_img)

    # write video
    out.release()
    command = 'ffmpeg -loglevel panic -y -i {} -i {} -vcodec libx264 -crf 12 -pix_fmt yuv420p -shortest {}'.format(
        wav_path, out_video_p.name, save_path)
    subprocess.call(command, shell=True)
    if args.save_crop:
        out_crop.release()
        command = 'ffmpeg -loglevel panic -y -i {} -i {} -vcodec libx264 -crf 12 -pix_fmt yuv420p -shortest {}'.format(
            wav_path, crop_out_video_p.name, save_path[:-4] + "_crop.mp4")
        subprocess.call(command, shell=True)

    out_video_p.close()
    crop_out_video_p.close()
    print('[DONE]')


if __name__ == '__main__':
    main()
