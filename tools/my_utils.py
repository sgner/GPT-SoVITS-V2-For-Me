import platform,os,traceback
import ffmpeg
import numpy as np
import gradio as gr
from tools.i18n.i18n import I18nAuto
import pandas as pd
i18n = I18nAuto(language=os.environ.get('language','Auto'))

def load_audio(file, sr):
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = clean_path(file)  # 防止小白拷路径头尾带了空格和"和回车
        if os.path.exists(file) == False:
            raise RuntimeError(
                "You input a wrong audio path that does not exists, please fix it!"
            )
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(i18n("音频加载失败"))

    return np.frombuffer(out, np.float32).flatten()


def clean_path(path_str:str):
    if path_str.endswith(('\\','/')):
        return clean_path(path_str[0:-1])
    path_str = path_str.replace('/', os.sep).replace('\\', os.sep)
    return path_str.strip(" ").strip('\'').strip("\n").strip('"').strip(" ").strip("\u202a")


def check_for_existance(file_list:list=None,is_train=False,is_dataset_processing=False):
    files_status=[]
    if is_train == True and file_list:
        file_list.append(os.path.join(file_list[0],'2-name2text.txt'))
        file_list.append(os.path.join(file_list[0],'3-bert'))
        file_list.append(os.path.join(file_list[0],'4-cnhubert'))
        file_list.append(os.path.join(file_list[0],'5-wav32k'))
        file_list.append(os.path.join(file_list[0],'6-name2semantic.tsv'))
    for file in file_list:
        if os.path.exists(file):files_status.append(True)
        else:files_status.append(False)
    if sum(files_status)!=len(files_status):
        if is_train:
            for file,status in zip(file_list,files_status):
                if status:pass
                else:print(file)
            print('文件或文件夹不存在')
            return False
        elif is_dataset_processing:
            if files_status[0]:
                return True
            elif not files_status[0]:
                print(file_list[0])
            elif not files_status[1] and file_list[1]:
                print(file_list[1])
            print(i18n('以下文件或文件夹不存在'))
            return False
        else:
            if file_list[0]:
                print(file_list[0])
                print(i18n('以下文件或文件夹不存在'))
            else:
                print(i18n('路径不能为空'))
            return False
    return True

def check_details(path_list=None,is_train=False,is_dataset_processing=False):
    if is_dataset_processing:
        list_path, audio_path = path_list
        if (not list_path.endswith('.list')):
            print('未填入正确的List路径')
            return
        if audio_path:
            if not os.path.isdir(audio_path):
                print('未填入正确的音频文件夹路径')
                return
        with open(list_path,"r",encoding="utf8")as f:
            line=f.readline().strip("\n").split("\n")
        wav_name, _, __, ___ = line[0].split("|")
        wav_name=clean_path(wav_name)
        if (audio_path != "" and audio_path != None):
            wav_name = os.path.basename(wav_name)
            wav_path = "%s/%s"%(audio_path, wav_name)
        else:
            wav_path=wav_name
        if os.path.exists(wav_path):
            ...
        else:
            print(i18n('路径错误'))
        return
    if is_train:
        path_list.append(os.path.join(path_list[0],'2-name2text.txt'))
        path_list.append(os.path.join(path_list[0],'4-cnhubert'))
        path_list.append(os.path.join(path_list[0],'5-wav32k'))
        path_list.append(os.path.join(path_list[0],'6-name2semantic.tsv'))
        phone_path, hubert_path, wav_path, semantic_path = path_list[1:]
        with open(phone_path,'r',encoding='utf-8') as f:
            if f.read(1):...
            else:print(i18n('缺少音素数据集'))
        if os.listdir(hubert_path):...
        else:print(i18n('缺少Hubert数据集'))
        if os.listdir(wav_path):...
        else:print(i18n('缺少音频数据集'))
        df = pd.read_csv(
            semantic_path, delimiter="\t", encoding="utf-8"
        )
        if len(df) >= 1:...
        else:print(i18n('缺少语义数据集'))


def replace_last_two_folders(path, replacement):
    # 找到倒数第二个反斜杠的位置
    last_slash = path.rfind(os.sep)  # 找到最后一个反斜杠的位置
    if last_slash == -1:
        return path  # 如果没有反斜杠，直接返回原路径

    second_last_slash = path.rfind(os.sep, 0, last_slash)  # 找到倒数第二个反斜杠的位置
    if second_last_slash == -1:
        return path  # 如果没有足够的反斜杠，也返回原路径

    # 替换倒数第二个反斜杠后面的内容
    new_path = path[:second_last_slash + 1] + replacement
    return new_path

def replace_last_part_of_path(path, new_part):
    # 查找最后一个反斜杠的位置
    last_backslash_index = path.rfind(os.sep)
    if last_backslash_index != -1:
        # 使用切片替换反斜杠后面的部分
        new_path = path[:last_backslash_index + 1] + new_part
        return new_path
    else:
        # 如果没有反斜杠，返回原路径
        return path

def replace_workspace_folder(path, replacement):
    # 找到 "workspace" 的位置
    workspace_index = path.find("workspace")
    if workspace_index == -1:
        return path  # 如果找不到 "workspace"，返回原路径

    # 找到 "workspace" 后面的第一个反斜杠的位置
    after_workspace_index = path.find(os.sep, workspace_index) + 1
    if after_workspace_index == 0:
        return path  # 如果没有找到，返回原路径

    # 找到 "workspace" 后面的文件夹结束的位置
    next_slash_index = path.find(os.sep, after_workspace_index)
    if next_slash_index == -1:
        next_slash_index = len(path)  # 如果没有更多的反斜杠，取到字符串末尾

    # 构造新的路径
    new_path = path[:after_workspace_index] + replacement + path[next_slash_index:]
    return new_path