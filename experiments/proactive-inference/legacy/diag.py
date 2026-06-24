import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as R, torch, transformers, subprocess, os
from PIL import Image
print("transformers:", transformers.__version__)
model, proc, tok = R.load_model()
print("model type:", type(model).__name__)
print("hasattr model.visual:", hasattr(model, "visual"))
mm = getattr(model, "model", None)
print("hasattr model.model.visual:", hasattr(mm, "visual") if mm is not None else "no model.model")
FF = str(Path.home()/"miniforge3"/"bin"/"ffmpeg")
os.makedirs("/tmp/diagf", exist_ok=True)
subprocess.run([FF,"-y","-ss","20","-i",R.DEFAULT_VIDEO,"-frames:v","1","/tmp/diagf/a.jpg"],capture_output=True)
fr = Image.open("/tmp/diagf/a.jpg").convert("RGB")
# test the EXISTING Backend.encode_frame
try:
    be = R.Backend(model, proc, tok, "cuda")
    ve,pos,ntok = be.encode_frame(fr, 0)
    print("Backend.encode_frame: OK · ntok", ntok, "· ve type", type(ve).__name__, "tensor", torch.is_tensor(ve))
except Exception as e:
    import traceback; traceback.print_exc(); print("Backend.encode_frame: BROKEN ->", e)
