import os

def make_tsv(metadata, save_path, metadata_header=None):
    if not metadata_header:
        metadata = [str(x) for x in metadata]
    else:
        assert len(metadata_header) == len(metadata[0]), \
            'len of header must be equal to the number of columns in metadata'
        metadata = ['\t'.join(str(e) for e in l) for l in [metadata_header] + metadata]
    import sys
    if sys.version_info[0] == 3:
        with open(os.path.join(save_path, 'metadata.tsv'), 'w', encoding='utf8') as f:
            for x in metadata:
                f.write(x + '\n')
    else:
        with open(os.path.join(save_path, 'metadata.tsv'), 'wb') as f:
            for x in metadata:
                f.write((x + '\n').encode('utf-8'))

def make_sprite(label_img, save_path):
    import math
    import numpy as np
    from .x2num import make_np
    from Code_for_Experiment.Metrics.music_understanding_model.tensorboardX.tensorboardX.utils import make_grid
    from PIL import Image
    assert label_img.shape[2] == label_img.shape[3], 'Image should be square, see tensorflow/tensorboard#670'
    total_pixels = label_img.shape[0] * label_img.shape[2] * label_img.shape[3]
    pixels_one_side = total_pixels ** 0.5
    number_of_images_per_row = int(math.ceil(pixels_one_side / label_img.shape[3]))
    arranged_img_CHW = make_grid(make_np(label_img), ncols=number_of_images_per_row)
    arranged_img_HWC = arranged_img_CHW.transpose(1, 2, 0)  # chw -> hwc
    arranged_augment_square_HWC = np.ndarray((arranged_img_CHW.shape[2], arranged_img_CHW.shape[2], 3))
    arranged_augment_square_HWC[:arranged_img_HWC.shape[0], :, :] = arranged_img_HWC
    im = Image.fromarray(np.uint8((arranged_augment_square_HWC * 255).clip(0, 255)))
    im.save(os.path.join(save_path, 'sprite.png'))

def append_pbtxt(metadata, label_img, save_path, subdir, global_step, tag):
    from posixpath import join
    with open(os.path.join(save_path, 'projector_config.pbtxt'), 'a') as f:
        f.write('embeddings {\n')
        f.write('tensor_name: "{}:{}"\n'.format(
            tag, str(global_step).zfill(5)))
        f.write('tensor_path: "{}"\n'.format(join(subdir, 'tensors.tsv')))
        if metadata is not None:
            f.write('metadata_path: "{}"\n'.format(
                join(subdir, 'metadata.tsv')))
        if label_img is not None:
            f.write('sprite {\n')
            f.write('image_path: "{}"\n'.format(join(subdir, 'sprite.png')))
            f.write('single_image_dim: {}\n'.format(label_img.shape[3]))
            f.write('single_image_dim: {}\n'.format(label_img.shape[2]))
            f.write('}\n')
        f.write('}\n')

def make_mat(matlist, save_path):
    with open(os.path.join(save_path, 'tensors.tsv'), 'w') as f:
        for x in matlist:
            x = [str(i.item()) for i in x]
            f.write('\t'.join(x) + '\n')