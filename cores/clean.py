import os
import subprocess
from tqdm import tqdm
import time
import numpy as np
import cv2
import torch
from models import runmodel
from util import data, util, filt
from util import image_processing as impro
from .init import video_init
from multiprocessing import Queue, Process
from threading import Thread

def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vf', 'fps=30',
        os.path.join(output_dir, 'output_%06d.jpg')
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def get_mosaic_positions(opt, netM, imagepaths, savemask=True):
    # resume
    continue_flag = False
    if os.path.isfile(os.path.join(opt.temp_dir, 'step.json')):
        step = util.loadjson(os.path.join(opt.temp_dir, 'step.json'))
        resume_frame = int(step['frame'])
        if int(step['step']) > 2:
            pre_positions = np.load(os.path.join(opt.temp_dir, 'mosaic_positions.npy'))
            return pre_positions
        if int(step['step']) >= 2 and resume_frame > 0:
            pre_positions = np.load(os.path.join(opt.temp_dir, 'mosaic_positions.npy'))
            continue_flag = True
            imagepaths = imagepaths[resume_frame:]

    positions = []
    t1 = time.time()
    if not opt.no_preview:
        cv2.namedWindow('mosaic mask', cv2.WINDOW_NORMAL)
    print('Step:2/4 -- Find mosaic location')

    img_read_pool = Queue(4)

    def loader(imagepaths):
        for imagepath in imagepaths:
            img_origin = impro.imread(os.path.join(opt.temp_dir + '/video2image', imagepath))
            img_read_pool.put(img_origin)

    t = Thread(target=loader, args=(imagepaths,))
    t.setDaemon(True)
    t.start()

    total_iterations = len(imagepaths)

    with tqdm(total=total_iterations, unit='image') as pbar:
        for i, imagepath in enumerate(imagepaths, 1):
            img_origin = img_read_pool.get()
            x, y, size, mask = runmodel.get_mosaic_position(img_origin, netM, opt)
            positions.append([x, y, size])

            if savemask:
                t = Thread(target=cv2.imwrite, args=(os.path.join(opt.temp_dir + '/mosaic_mask', imagepath), mask,))
                t.start()

            if i % 1000 == 0:
                save_positions = np.array(positions)

                if continue_flag:
                    save_positions = np.concatenate((pre_positions, save_positions), axis=0)

                np.save(os.path.join(opt.temp_dir, 'mosaic_positions.npy'), save_positions)
                step = {'step': 2, 'frame': i + resume_frame}
                util.savejson(os.path.join(opt.temp_dir, 'step.json'), step)

            # Preview result and print
            if not opt.no_preview:
                cv2.imshow('mosaic mask', mask)
                cv2.waitKey(1) & 0xFF

            t2 = time.time()
            pbar.set_description(f'Processing: {i}/{total_iterations}')
            pbar.set_postfix_str(f'Time: {util.counttime(t1, t2, i, total_iterations)}')
            pbar.update(1)

    if not opt.no_preview:
        cv2.destroyAllWindows()
    print('\nOptimize mosaic locations...')
    positions = np.array(positions)
    if continue_flag:
        positions = np.concatenate((pre_positions, positions), axis=0)
    for i in range(3):
        positions[:, i] = filt.medfilt(positions[:, i], opt.medfilt_num)
    step = {'step': 3, 'frame': 0}
    util.savejson(os.path.join(opt.temp_dir, 'step.json'), step)
    np.save(os.path.join(opt.temp_dir, 'mosaic_positions.npy'), positions)

    return positions

def cleanmosaic_video_fusion(opt, netG, netM):
    path = opt.media_path
    N, T, S = 2, 5, 3
    LEFT_FRAME = (N * S)
    POOL_NUM = LEFT_FRAME * 2 + 1
    INPUT_SIZE = 256
    FRAME_POS = np.linspace(0, (T - 1) * S, T, dtype=np.int64)
    img_pool = []
    previous_frame = None
    init_flag = True

    fps, imagepaths, height, width = video_init(opt, path)
    start_frame = int(imagepaths[0][7:13])
    positions = get_mosaic_positions(opt, netM, imagepaths, savemask=True)[(start_frame - 1):]
    t1 = time.time()
    if not opt.no_preview:
        cv2.namedWindow('clean', cv2.WINDOW_NORMAL)

    # clean mosaic
    print('Step:3/4 -- Clean Mosaic:')
    length = len(imagepaths)
    write_pool = Queue(4)
    show_pool = Queue(4)

    def write_result():
        while True:
            save_ori, imagepath, img_origin, img_fake, x, y, size = write_pool.get()
            if save_ori:
                img_result = img_origin
            else:
                mask = cv2.imread(os.path.join(opt.temp_dir + '/mosaic_mask', imagepath), 0)
                img_result = impro.replace_mosaic(img_origin, img_fake, mask, x, y, size, opt.no_feather)
            if not opt.no_preview:
                show_pool.put(img_result.copy())
            cv2.imwrite(os.path.join(opt.temp_dir + '/replace_mosaic', imagepath), img_result)
            os.remove(os.path.join(opt.temp_dir + '/video2image', imagepath))

    t = Thread(target=write_result, args=())
    t.setDaemon(True)
    t.start()

    with tqdm(total=length, unit='image') as pbar:
        for i, imagepath in enumerate(imagepaths, 0):
            x, y, size = positions[i][0], positions[i][1], positions[i][2]
            input_stream = []
            # image read stream
            if i == 0:  # init
                for j in range(POOL_NUM):
                    img_pool.append(
                        impro.imread(os.path.join(opt.temp_dir + '/video2image', imagepaths[np.clip(i + j - LEFT_FRAME, 0, len(imagepaths) - 1)])))
            else:  # load next frame
                img_pool.pop(0)
                img_pool.append(
                    impro.imread(os.path.join(opt.temp_dir + '/video2image', imagepaths[np.clip(i + LEFT_FRAME, 0, len(imagepaths) - 1)])))
            img_origin = img_pool[LEFT_FRAME]

            # preview result and print
            if not opt.no_preview:
                if show_pool.qsize() > 3:
                    cv2.imshow('clean', show_pool.get())
                    cv2.waitKey(1) & 0xFF

            if size > 50:
                try:  # Avoid unknown errors
                    work_zone_width = 2 * size
                    work_zone_height = int(work_zone_width / (width / height))

                    for pos in FRAME_POS:
                        work_zone = img_pool[pos][y - work_zone_height // 2:y + work_zone_height // 2, x - work_zone_width // 2:x + work_zone_width // 2]
                        resized_work_zone = impro.resize(work_zone, INPUT_SIZE, interpolation=cv2.INTER_CUBIC)[:, :, ::-1]
                        input_stream.append(resized_work_zone)

                    if init_flag:
                        init_flag = False
                        previous_frame = input_stream[N]
                        previous_frame = data.im2tensor(previous_frame, bgr2rgb=True, gpu_id=opt.gpu_id)

                    input_stream = np.array(input_stream).reshape(1, T, INPUT_SIZE, INPUT_SIZE, 3).transpose((0, 4, 1, 2, 3))
                    input_stream = data.to_tensor(data.normalize(input_stream), gpu_id=opt.gpu_id)
                    with torch.no_grad():
                        unmosaic_pred = netG(input_stream, previous_frame)
                    img_fake = data.tensor2im(unmosaic_pred, rgb2bgr=True)
                    previous_frame = unmosaic_pred
                    write_pool.put([False, imagepath, img_origin.copy(), img_fake.copy(), x, y, size])
                except Exception as e:
                    init_flag = True
                    print('Error:', e)
            else:
                write_pool.put([True, imagepath, img_origin.copy(), -1, -1, -1])
                init_flag = True

            t2 = time.time()
            pbar.set_description(f'Processing: {i + 1}/{length}')
            pbar.set_postfix_str(f'Time: {util.counttime(t1, t2, i + 1, len(imagepaths))}')
            pbar.update(1)

    write_pool.close()
    show_pool.close()
    if not opt.no_preview:
        cv2.destroyAllWindows()
    print('Step:4/4 -- Convert images to video')

    # Convert images to video
    output_filename = os.path.join(opt.result_dir, os.path.splitext(os.path.basename(path))[0] + '_clean.mp4')
    input_pattern = os.path.join(opt.temp_dir, 'replace_mosaic/output_%06d.' + opt.tempimage_type)

    # Get video dimensions
    video_info = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=p=0', path], stdout=subprocess.PIPE).stdout.decode().strip().split(',')
    video_width = int(video_info[0])
    video_height = int(video_info[1])

    # Calculate the desired output width and height based on the video's aspect ratio
    output_width = 1280
    output_height = int(output_width * video_height / video_width)

    cmd = [
        'ffmpeg',
        '-r', str(fps),
        '-i', input_pattern,
        '-vf', f'scale={output_width}:{output_height}',
        '-c:v', 'libx264',
        '-crf', '18',
        '-preset', 'fast',
        '-y', output_filename
    ]

    pbar.write('Converting images to video...')

    frame_counter = 0
    with tqdm(total=length, ncols=80) as pbar_ffmpeg:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        for line in iter(proc.stderr.readline, b''):
            line_str = line.decode().strip()
            if line_str.startswith('frame='):
                frame_counter += 1
                pbar_ffmpeg.set_postfix_str(f'Frame: {frame_counter}/{length}')
                pbar_ffmpeg.update(1)

        proc.wait()

    return output_filename
