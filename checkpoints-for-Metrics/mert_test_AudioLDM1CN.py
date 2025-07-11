import numpy as np
import librosa
import torch
import laion_clap
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
from torch import nn
import torchaudio.transforms as T
import torchaudio
import json
import csv
from tqdm import trange


def compute_cosine_similarity(audio_embed, text_embed):
    # 计算余弦相似度
    audio_embed_norm = F.normalize(audio_embed, p=2, dim=1)
    text_embed_norm = F.normalize(text_embed, p=2, dim=1)
    similarity = torch.mm(audio_embed_norm, text_embed_norm.transpose(0, 1))
    return similarity


# quantization
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)


def clap_test(audio_file, text_data, use_default=True):
    '''
    使用CLAP模型计算音频文件与文本数据之间的余弦相似度。
    
    参数:
    audio_file -- 音频文件路径或文件列表
    text_data -- 文本数据，需为可嵌入的格式
    use_default -- 是否直接使用默认的CLAP模型（不进行微调）
    
    返回:
    无返回值，直接打印余弦相似度矩阵。
    修改为返回余弦相似度矩阵
    '''
    # model = laion_clap.CLAP_Module(enable_fusion=False)
    # model.load_ckpt("D:/wangqi/630k-audioset-best.pt") # 加载默认的预训练检查点。
    # model.load_ckpt()
    audio_embed = model.get_audio_embedding_from_filelist(x=audio_file, use_tensor=True)
    text_embed = model.get_text_embedding(text_data, use_tensor=True)

    similarity = compute_cosine_similarity(audio_embed, text_embed)
    # print(similarity)  # 打印余弦相似度矩阵
    return similarity


def mert_test(generated, original, use_default=True, data_length=80000):
    '''
    使用MERT模型计算生成音频与原始音频之间的平均余弦相似度。
    
    参数:
    generated -- 生成音频的文件路径列表
    original -- 原始音频的文件路径列表
    use_default -- 是否直接使用默认的MERT模型（不进行微调）
    data_length -- 用于计算的数据长度
    
    返回:
    res -- 包含每对音频之间平均余弦相似度的一个列表。
    '''
    # 加载我们的模型权重

    res = []
    for i in trange(len(generated)):
        generated_audio, _ = torchaudio.load(generated[i])
        original_audio, _ = torchaudio.load(original[i])
        generated_audio = generated_audio[:1, :data_length].to(device)
        original_audio = original_audio[:1, :data_length].to(device)
        with torch.no_grad():
            output_generated = model(generated_audio, output_hidden_states=True)
            output_original = model(original_audio, output_hidden_states=True)
        g_all_layer_hidden_states = torch.stack(output_generated.hidden_states).squeeze().mean(-2)
        o_ll_layer_hidden_states = torch.stack(output_original.hidden_states).squeeze().mean(-2)
        similarity = compute_cosine_similarity(g_all_layer_hidden_states, o_ll_layer_hidden_states)
        average_similarity = torch.mean(similarity)
        res.append(average_similarity)

    return res


if __name__ == '__main__':
    data = None
    with open("/home/zhehui/stable-audio-tools/test_input_CN.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    # prompt = data[str(i)]["short"]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).to(device)

    gt_file_list=[]
    audio_file_list=[]
    for i in range(1, 401):
        path = '/home/zhehui/FAD/AudioLDM1CN/' + str(i) + '_0.wav'
        src = data[str(i)]["song_name"]
        gt_file_list.append(path)
        audio_file_list.append('/home/zhehui/muchin_all/'+src[0:-8]+'/'+src)
    res = mert_test(gt_file_list, audio_file_list)
    res_mert=[]
    for i in range(400):
        res_mert.append((gt_file_list[i],res[i].item()))

    with open("/home/zhehui/wangqi/mertresult_AudioLDM1CN.csv","w",encoding="utf-8") as f:
        writer = csv.writer(f)
        for i in range(400):
            writer.writerow(res_mert[i])
