import os
import traceback
import logging
import shutil
import ffmpeg
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from typing import Optional
from typing import List
from pydantic import BaseModel
from tools.i18n.i18n import I18nAuto
from tools.my_utils import clean_path
from mdxnet import MDXNetDereverb
from vr import AudioPre, AudioPreDeEcho
from bsroformer import BsRoformer_Loader
import sys

# Initialize i18n and logger
i18n = I18nAuto()
logger = logging.getLogger(__name__)

# Global variables for models and paths
weight_uvr5_root = "tools/uvr5/uvr5_weights"
uvr5_names = [name.replace(".pth", "").replace(".ckpt", "") for name in os.listdir(weight_uvr5_root)
              if name.endswith(".pth") or name.endswith(".ckpt") or "onnx" in name]

device = sys.argv[1]
is_half = eval(sys.argv[2])
webui_port_uvr5 = int(sys.argv[3])
is_share = eval(sys.argv[4])

# FastAPI app initialization
app = FastAPI(title="UVR5 WebUI API")


# Model loading function
def uvr(model_name, inp_root, save_root_vocal, save_root_ins, format0, agg, paths):
    infos = []
    try:
        inp_root = clean_path(inp_root)
        save_root_vocal = clean_path(save_root_vocal)
        save_root_ins = clean_path(save_root_ins)
        is_hp3 = "HP3" in model_name

        # Select the model based on the name
        if model_name == "onnx_dereverb_By_FoxJoy":
            pre_fun = MDXNetDereverb(15)
        elif model_name == "Bs_Roformer" or "bs_roformer" in model_name.lower():
            func = BsRoformer_Loader
            pre_fun = func(
                model_path=os.path.join(weight_uvr5_root, model_name + ".ckpt"),
                device=device,
                is_half=is_half
            )
        else:
            print(model_name)
            func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
            pre_fun = func(
                agg=int(agg),
                model_path=os.path.join(weight_uvr5_root, model_name + ".pth"),
                device=device,
                is_half=is_half,
            )
        # If input root is provided, get file paths
        if inp_root != "":
            paths = [os.path.join(inp_root, name) for name in os.listdir(inp_root)]
        else:
            paths = [path.name for path in paths]

        # Process the audio files
        for path in paths:
            inp_path = os.path.join(inp_root, path)
            if not os.path.isfile(inp_path):
                continue

            need_reformat = 1
            done = 0
            try:
                info = ffmpeg.probe(inp_path, cmd="ffprobe")
                if info["streams"][0]["channels"] == 2 and info["streams"][0]["sample_rate"] == "44100":
                    need_reformat = 0
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0, is_hp3
                    )
                    done = 1
            except Exception as e:
                need_reformat = 1
                logger.error(f"Error processing file {inp_path}: {str(e)}")
                traceback.print_exc()

            if need_reformat == 1:
                tmp_path = "%s/%s.reformatted.wav" % (os.path.join(os.environ["TEMP"]), os.path.basename(inp_path))
                os.system(f'ffmpeg -i "{inp_path}" -vn -acodec pcm_s16le -ac 2 -ar 44100 "{tmp_path}" -y')
                inp_path = tmp_path

            try:
                if done == 0:
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0, is_hp3
                    )
                infos.append(f"{os.path.basename(inp_path)}->Success")
            except Exception as e:
                infos.append(f"{os.path.basename(inp_path)}->{str(e)}")
                traceback.print_exc()

    except Exception as e:
        infos.append(traceback.format_exc())
        logger.error(f"General error: {str(e)}")
    finally:
        try:
            if model_name == "onnx_dereverb_By_FoxJoy":
                del pre_fun.pred.model
                del pre_fun.pred.model_
            else:
                del pre_fun.model
                del pre_fun
        except:
            traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return "\n".join(infos)


# Define the request models
class AudioProcessingRequest(BaseModel):
    model_name: str
    input_root: str
    save_root_vocal: str
    save_root_ins: str
    format0: str
    agg: Optional[int] = 10
    paths: Optional[List[str]] = []


@app.post("/process_audio")
async def process_audio(request: AudioProcessingRequest):
    try:
        print(request)
        # Process audio based on the request
        result = uvr(
            request.model_name,
            request.input_root,
            request.save_root_vocal,
            request.save_root_ins,
            request.format0,
            request.agg,
            request.paths
        )
        return {"status": "success", "message": result}
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/upload_audio/")
async def upload_audio(files: List[UploadFile] = File(...)):
    try:
        # Handle file upload and save them
        uploaded_files = []
        for file in files:
            file_location = f"temp/{file.filename}"
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            uploaded_files.append(file_location)
        return {"status": "success", "files": uploaded_files}
    except Exception as e:
        logger.error(f"Error uploading audio files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    print("Starting server...")
    print("UVR 5 服务启动了")
    uvicorn.run(app, host="0.0.0.0", port=webui_port_uvr5)
