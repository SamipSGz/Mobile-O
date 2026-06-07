"""
hardware_scheduler.py — Runtime hardware detection and component assignment for Mobile-O.

Detects available compute units at runtime and assigns each model component
(vision encoder, LLM, diffusion) to the optimal available hardware unit.

Supported targets:
  Apple M-series  — ANE (via CoreML) + MPS (via PyTorch)
  NVIDIA GPU      — CUDA (PyTorch)
  NVIDIA Jetson   — DLA (via TensorRT/onnxruntime) + CUDA fallback
  Qualcomm Snap.  — QNN/SNPE NPU (via onnxruntime QNN EP) + CPU fallback
  MediaTek        — APU (via NeuroPilot / ONNX) + CPU fallback
  Raspberry Pi    — CPU only (ARM Cortex)
  Generic Linux   — CPU fallback

Usage:
    from hardware_scheduler import HardwareScheduler
    sched = HardwareScheduler()
    sched.print_report()
    cfg = sched.get_config()
    # cfg.vision_device, cfg.llm_device, cfg.diffusion_device, cfg.dtype, ...
"""
from __future__ import annotations

import os
import sys
import platform
import subprocess
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Hardware probe functions — each returns True/False, never raises
# ══════════════════════════════════════════════════════════════════════════════

def _probe_mps() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available() and torch.backends.mps.is_built()
    except Exception:
        return False


def _probe_cuda() -> tuple[bool, int, float]:
    """Returns (available, device_count, total_vram_gb)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, 0, 0.0
        n = torch.cuda.device_count()
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if n > 0 else 0.0
        return True, n, vram
    except Exception:
        return False, 0, 0.0


def _probe_ane() -> bool:
    """ANE is available on all Apple Silicon Macs (M1+). CoreML routes to it automatically."""
    try:
        import coremltools as ct  # noqa: F401
        return platform.system() == "Darwin" and platform.machine() == "arm64"
    except ImportError:
        return False


def _probe_coreml_model(path: str) -> bool:
    """Check if a pre-exported CoreML model exists and is loadable."""
    try:
        import coremltools as ct
        p = Path(path)
        if not p.exists():
            return False
        ct.models.MLModel(str(p))
        return True
    except Exception:
        return False


def _probe_jetson() -> tuple[bool, bool]:
    """Returns (is_jetson, has_dla)."""
    try:
        model_path = "/proc/device-tree/model"
        if not os.path.exists(model_path):
            return False, False
        with open(model_path, "r", errors="ignore") as f:
            model_str = f.read().lower()
        is_jetson = "jetson" in model_str
        # DLA is available on Jetson Xavier, Orin (not Nano/TX2)
        has_dla = is_jetson and any(x in model_str for x in ["xavier", "orin", "agx"])
        return is_jetson, has_dla
    except Exception:
        return False, False


def _probe_qualcomm_qnn() -> bool:
    """Detect Qualcomm QNN/SNPE via onnxruntime QNN execution provider."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        return "QNNExecutionProvider" in providers
    except ImportError:
        pass
    # Fallback: check for SNPE shared library
    snpe_paths = [
        "/usr/lib/libSNPE.so",
        "/vendor/lib64/libSnpeHtp.so",
        "/usr/local/lib/libQnnHtp.so",
    ]
    return any(Path(p).exists() for p in snpe_paths)


def _probe_mediatek_apu() -> bool:
    """Detect MediaTek APU via cpuinfo or NeuroPilot library."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpu_info = f.read().lower()
        if "mediatek" not in cpu_info and "mt" not in cpu_info:
            return False
        # Check for NeuroPilot or ONNX APU delegate
        np_paths = [
            "/system/lib64/libneuronusdk_adapter.neuron.so",
            "/usr/lib/libneuron_runtime.so",
        ]
        return any(Path(p).exists() for p in np_paths)
    except Exception:
        return False


def _probe_raspberry_pi() -> tuple[bool, str]:
    """Returns (is_rpi, model_string)."""
    try:
        for path in ["/proc/device-tree/model", "/proc/cpuinfo"]:
            if not os.path.exists(path):
                continue
            with open(path, "r", errors="ignore") as f:
                content = f.read().lower()
            if "raspberry" in content:
                # Extract model number
                for line in content.split("\n"):
                    if "raspberry" in line:
                        return True, line.strip()
        return False, ""
    except Exception:
        return False, ""


def _probe_ram_gb() -> tuple[float, float]:
    """Returns (total_gb, available_gb)."""
    try:
        import psutil
        m = psutil.virtual_memory()
        return m.total / 1e9, m.available / 1e9
    except ImportError:
        pass
    # Fallback: /proc/meminfo
    try:
        meminfo = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":")
                meminfo[k.strip()] = int(v.strip().split()[0]) / 1e6  # KB→GB
        return meminfo.get("MemTotal", 0), meminfo.get("MemAvailable", 0)
    except Exception:
        return 0.0, 0.0


def _probe_gguf_llm(path: str) -> bool:
    """Check if INT4 GGUF model exists and llama_cpp is available."""
    try:
        from llama_cpp import Llama  # noqa: F401
        return Path(path).exists()
    except ImportError:
        return False


def _get_chip_name() -> str:
    """Best-effort chip/SoC name."""
    system = platform.system()
    if system == "Darwin":
        try:
            r = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2
            )
            name = r.stdout.strip()
            if not name:
                r2 = subprocess.run(
                    ["system_profiler", "SPHardwareDataType"],
                    capture_output=True, text=True, timeout=5
                )
                for line in r2.stdout.split("\n"):
                    if "Chip:" in line:
                        return line.split("Chip:")[-1].strip()
            return name
        except Exception:
            return "Apple Silicon"
    if system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line.lower() or "Hardware" in line:
                        return line.split(":")[-1].strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


# ══════════════════════════════════════════════════════════════════════════════
# Hardware profile dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HardwareProfile:
    # Platform
    os_name:        str = ""
    chip_name:      str = ""
    # Compute units
    has_ane:        bool = False   # Apple Neural Engine
    has_mps:        bool = False   # Apple Metal Performance Shaders (GPU)
    has_cuda:       bool = False   # NVIDIA CUDA
    cuda_devices:   int  = 0
    cuda_vram_gb:   float = 0.0
    has_dla:        bool = False   # NVIDIA Jetson DLA
    has_qnn:        bool = False   # Qualcomm QNN/SNPE NPU
    has_apu:        bool = False   # MediaTek APU
    is_jetson:      bool = False
    is_rpi:         bool = False
    rpi_model:      str  = ""
    # Memory
    ram_total_gb:   float = 0.0
    ram_avail_gb:   float = 0.0
    # Pre-exported models
    coreml_vision_path:  str = "vision_encoder.mlpackage"
    gguf_llm_path:       str = "llm_export/model-q4.gguf"
    has_coreml_vision:   bool = False
    has_gguf_llm:        bool = False


# ══════════════════════════════════════════════════════════════════════════════
# Assigned configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ComponentConfig:
    """
    The scheduler's output: where each model component should run and
    at what precision. All downstream code reads this — never hardcodes devices.
    """
    # Devices
    vision_device:     str = "cpu"   # "ane", "mps", "cuda", "npu_qnn", "apu", "cpu"
    llm_device:        str = "cpu"   # "mps", "cuda_int4", "cuda", "cpu_int4", "cpu"
    diffusion_device:  str = "cpu"   # "mps", "cuda", "cpu"
    # Precision
    dtype_str:         str = "float32"   # "float16", "bfloat16", "int4", "float32"
    # Parallelism
    use_async_pipeline: bool = False     # can vision + LLM run truly in parallel?
    # Loader hints
    use_coreml_vision:  bool = False     # load CoreML .mlpackage for vision
    use_gguf_llm:       bool = False     # load GGUF INT4 for LLM
    use_cuda_llm_int4:  bool = False     # use bitsandbytes INT4 on CUDA
    # Extra context
    reason:            dict = field(default_factory=dict)

    @property
    def torch_dtype(self):
        import torch
        return {
            "float16":  torch.float16,
            "bfloat16": torch.bfloat16,
            "float32":  torch.float32,
            "int4":     torch.float16,   # INT4 quantized, store as fp16 container
        }.get(self.dtype_str, torch.float32)

    @property
    def torch_device(self) -> str:
        """Primary torch device for the PyTorch full model."""
        if "cuda" in self.llm_device:
            return "cuda"
        if "mps" in self.llm_device:   # handles "mps" and "mps_int4"
            return "mps"
        return "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# Scheduler
# ══════════════════════════════════════════════════════════════════════════════

class HardwareScheduler:
    """
    Detects hardware at runtime and assigns Mobile-O components to optimal units.

    Example:
        sched = HardwareScheduler()
        sched.print_report()
        cfg = sched.get_config()

        # Load vision encoder accordingly
        if cfg.use_coreml_vision:
            import coremltools as ct
            vision_model = ct.models.MLModel(sched.hw.coreml_vision_path,
                                              compute_units=ct.ComputeUnit.ALL)
        else:
            vision_model = pytorch_vision_tower.to(cfg.torch_device)
    """

    def __init__(
        self,
        coreml_vision_path: str = "vision_encoder.mlpackage",
        gguf_llm_path:       str = "llm_export/model-q4.gguf",
    ):
        self.hw = self._detect(coreml_vision_path, gguf_llm_path)
        self._cfg: Optional[ComponentConfig] = None

    # ── Detection ─────────────────────────────────────────────────────────────
    @staticmethod
    def _detect(coreml_path: str, gguf_path: str) -> HardwareProfile:
        hw = HardwareProfile()
        hw.os_name   = platform.system()
        hw.chip_name = _get_chip_name()

        hw.has_mps  = _probe_mps()
        hw.has_ane  = _probe_ane()
        hw.has_cuda, hw.cuda_devices, hw.cuda_vram_gb = _probe_cuda()
        hw.is_jetson, hw.has_dla = _probe_jetson()
        hw.has_qnn  = _probe_qualcomm_qnn()
        hw.has_apu  = _probe_mediatek_apu()
        hw.is_rpi, hw.rpi_model = _probe_raspberry_pi()
        hw.ram_total_gb, hw.ram_avail_gb = _probe_ram_gb()

        hw.coreml_vision_path = coreml_path
        hw.gguf_llm_path      = gguf_path
        hw.has_coreml_vision  = _probe_coreml_model(coreml_path)
        hw.has_gguf_llm       = _probe_gguf_llm(gguf_path)

        return hw

    # ── Assignment logic ──────────────────────────────────────────────────────
    def get_config(self) -> ComponentConfig:
        if self._cfg is not None:
            return self._cfg

        hw  = self.hw
        cfg = ComponentConfig()
        r   = cfg.reason   # human-readable explanation for each decision

        # ── Precision ─────────────────────────────────────────────────────────
        if hw.ram_total_gb < 4:
            cfg.dtype_str = "int4"
            r["dtype"] = f"RAM={hw.ram_total_gb:.1f}GB < 4GB → INT4 everywhere"
        elif hw.ram_total_gb < 8:
            cfg.dtype_str = "float16"
            r["dtype"] = f"RAM={hw.ram_total_gb:.1f}GB 4–8GB → float16"
        elif hw.has_mps or hw.has_cuda:
            cfg.dtype_str = "float16"
            r["dtype"] = f"GPU available → float16"
        else:
            cfg.dtype_str = "float32"
            r["dtype"] = f"CPU-only → float32 (float16 unstable on some CPUs)"

        # ── Vision encoder ────────────────────────────────────────────────────
        # Priority: ANE (CoreML, 11.5× faster) > QNN > CUDA > MPS > CPU
        if hw.has_ane and hw.has_coreml_vision:
            cfg.vision_device    = "ane"
            cfg.use_coreml_vision = True
            r["vision"] = "Apple ANE via CoreML (.mlpackage) — 11.5× faster than MPS"
        elif hw.has_ane and not hw.has_coreml_vision:
            cfg.vision_device    = "mps"
            cfg.use_coreml_vision = False
            r["vision"] = "ANE available but vision_encoder.mlpackage not found → MPS fallback. Run: python export_vision_coreml.py"
        elif hw.has_qnn:
            cfg.vision_device = "npu_qnn"
            r["vision"] = "Qualcomm QNN NPU via onnxruntime QNNExecutionProvider"
        elif hw.has_apu:
            cfg.vision_device = "apu"
            r["vision"] = "MediaTek APU via NeuroPilot delegate"
        elif hw.has_dla and hw.is_jetson:
            cfg.vision_device = "dla"
            r["vision"] = "NVIDIA Jetson DLA — efficient for conv-heavy vision"
        elif hw.has_cuda:
            cfg.vision_device = "cuda"
            r["vision"] = f"NVIDIA CUDA — {hw.cuda_devices} GPU(s), {hw.cuda_vram_gb:.1f}GB VRAM"
        elif hw.has_mps:
            cfg.vision_device = "mps"
            r["vision"] = "Apple MPS (ANE unavailable or CoreML not exported)"
        else:
            cfg.vision_device = "cpu"
            r["vision"] = "CPU fallback — no GPU/NPU detected"

        # ── LLM ───────────────────────────────────────────────────────────────
        # Priority: INT4 Metal > INT4 CUDA > MPS fp16 > CUDA fp16 > CPU INT4 > CPU fp32
        if hw.has_mps and hw.has_gguf_llm:
            cfg.llm_device    = "mps_int4"
            cfg.use_gguf_llm  = True
            r["llm"] = "llama.cpp INT4 Q4_K_M on Metal — 22× faster decode than MPS fp16"
        elif hw.has_cuda and hw.has_gguf_llm:
            cfg.llm_device      = "cuda_int4"
            cfg.use_gguf_llm    = True
            cfg.use_cuda_llm_int4 = True
            r["llm"] = "llama.cpp INT4 Q4_K_M on CUDA — fastest decode"
        elif hw.has_cuda and hw.cuda_vram_gb >= 4:
            cfg.llm_device = "cuda"
            r["llm"] = f"CUDA fp16 — {hw.cuda_vram_gb:.1f}GB VRAM sufficient"
        elif hw.has_mps:
            cfg.llm_device = "mps"
            r["llm"] = "MPS fp16 — GGUF not found, using PyTorch. Run: python export_llm_gguf.py"
        elif hw.ram_total_gb >= 4 and hw.has_gguf_llm:
            cfg.llm_device   = "cpu_int4"
            cfg.use_gguf_llm = True
            r["llm"] = "CPU INT4 via llama.cpp — no GPU but enough RAM"
        else:
            cfg.llm_device = "cpu"
            r["llm"] = "CPU fp32 fallback — insufficient RAM/GPU for INT4"

        # ── Diffusion (DiT + VAE) ─────────────────────────────────────────────
        # DiT is the bottleneck (80–83% of gen/edit time). No INT4/ANE export yet.
        # Priority: CUDA > MPS > CPU
        if hw.has_cuda:
            cfg.diffusion_device = "cuda"
            r["diffusion"] = f"CUDA — best for DiT transformer matrix ops, {hw.cuda_vram_gb:.1f}GB VRAM"
        elif hw.has_mps:
            cfg.diffusion_device = "mps"
            r["diffusion"] = "MPS — DiT on GPU; CoreML DiT export would give further speedup (future work)"
        else:
            cfg.diffusion_device = "cpu"
            r["diffusion"] = "CPU — slow but functional; consider reducing DiT steps 20→10"

        # ── Async pipeline ────────────────────────────────────────────────────
        # Only beneficial when vision and LLM are on physically separate silicon
        vision_is_ane = cfg.vision_device == "ane"
        llm_is_gpu    = cfg.llm_device in ("mps", "mps_int4", "cuda", "cuda_int4")
        cfg.use_async_pipeline = vision_is_ane and llm_is_gpu
        r["async"] = (
            "ANE + Metal run on separate silicon → async pipeline overlaps them"
            if cfg.use_async_pipeline
            else "Vision and LLM share same compute unit → async pipeline would not help"
        )

        self._cfg = cfg
        return cfg

    # ── Utilities ─────────────────────────────────────────────────────────────
    def print_report(self):
        hw  = self.hw
        cfg = self.get_config()

        RESET = "\033[0m"
        BOLD  = "\033[1m"
        GREEN = "\033[92m"
        YELLOW= "\033[93m"
        RED   = "\033[91m"
        CYAN  = "\033[96m"
        DIM   = "\033[2m"

        def tick(v): return f"{GREEN}✓{RESET}" if v else f"{RED}✗{RESET}"

        print(f"\n{BOLD}{'═'*60}{RESET}")
        print(f"{BOLD}  Mobile-O Hardware Scheduler{RESET}")
        print(f"{'═'*60}")
        print(f"  {BOLD}Platform{RESET}")
        print(f"    OS    : {hw.os_name} {platform.release()}")
        print(f"    Chip  : {hw.chip_name}")
        print(f"    RAM   : {hw.ram_total_gb:.1f} GB total  /  {hw.ram_avail_gb:.1f} GB available")
        print()
        print(f"  {BOLD}Compute Units Detected{RESET}")
        print(f"    {tick(hw.has_ane)}  Apple ANE (Neural Engine, CoreML)")
        print(f"    {tick(hw.has_mps)}  Apple MPS (Metal Performance Shaders)")
        print(f"    {tick(hw.has_cuda)}  NVIDIA CUDA  " +
              (f"[{hw.cuda_devices} GPU(s), {hw.cuda_vram_gb:.1f}GB VRAM]" if hw.has_cuda else ""))
        print(f"    {tick(hw.is_jetson)}  NVIDIA Jetson  " +
              (f"[DLA: {tick(hw.has_dla)}]" if hw.is_jetson else ""))
        print(f"    {tick(hw.has_qnn)}  Qualcomm QNN/SNPE NPU")
        print(f"    {tick(hw.has_apu)}  MediaTek APU")
        print(f"    {tick(hw.is_rpi)}  Raspberry Pi  " +
              (f"[{hw.rpi_model}]" if hw.is_rpi else ""))
        print()
        print(f"  {BOLD}Pre-exported Models{RESET}")
        print(f"    {tick(hw.has_coreml_vision)}  CoreML vision encoder  [{hw.coreml_vision_path}]")
        print(f"    {tick(hw.has_gguf_llm)}  INT4 GGUF LLM           [{hw.gguf_llm_path}]")
        if not hw.has_coreml_vision:
            print(f"    {DIM}    → run: python export_vision_coreml.py{RESET}")
        if not hw.has_gguf_llm:
            print(f"    {DIM}    → run: python export_llm_gguf.py{RESET}")
        print()
        print(f"  {BOLD}{'─'*56}{RESET}")
        print(f"  {BOLD}Component Assignments{RESET}")
        print(f"  {BOLD}{'─'*56}{RESET}")

        def dev_color(d):
            if d in ("ane", "mps_int4", "cuda_int4"): return GREEN
            if d in ("mps", "cuda"):                   return CYAN
            if d in ("cpu_int4",):                     return YELLOW
            return DIM

        def fmt(label, device, reason):
            col = dev_color(device)
            print(f"  {BOLD}{label:<22}{RESET} {col}{device:<14}{RESET}  {DIM}{reason}{RESET}")

        fmt("Vision Encoder",    cfg.vision_device,    cfg.reason.get("vision",""))
        fmt("LLM Decode",        cfg.llm_device,       cfg.reason.get("llm",""))
        fmt("DiT Diffusion",     cfg.diffusion_device, cfg.reason.get("diffusion",""))
        fmt("Precision",         cfg.dtype_str,        cfg.reason.get("dtype",""))
        fmt("Async pipeline",    str(cfg.use_async_pipeline), cfg.reason.get("async",""))
        print(f"{'═'*60}\n")

    def summary_dict(self) -> dict:
        """Machine-readable summary for logging / paper tables."""
        hw  = self.hw
        cfg = self.get_config()
        return {
            "chip":             hw.chip_name,
            "ram_gb":           round(hw.ram_total_gb, 1),
            "has_ane":          hw.has_ane,
            "has_mps":          hw.has_mps,
            "has_cuda":         hw.has_cuda,
            "has_qnn":          hw.has_qnn,
            "has_dla":          hw.has_dla,
            "has_apu":          hw.has_apu,
            "is_rpi":           hw.is_rpi,
            "vision_device":    cfg.vision_device,
            "llm_device":       cfg.llm_device,
            "diffusion_device": cfg.diffusion_device,
            "dtype":            cfg.dtype_str,
            "async_pipeline":   cfg.use_async_pipeline,
            "use_coreml":       cfg.use_coreml_vision,
            "use_gguf":         cfg.use_gguf_llm,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Model loader — uses config to load each component correctly
# ══════════════════════════════════════════════════════════════════════════════

class ScheduledModelLoader:
    """
    Loads Mobile-O components according to HardwareScheduler decisions.
    Replaces hardcoded device strings everywhere.

    Usage:
        sched  = HardwareScheduler()
        loader = ScheduledModelLoader(sched, model_path="checkpoints/")
        loader.load()
        # loader.vision_model, loader.llm_model, loader.tokenizer, loader.image_processor
    """

    def __init__(self, scheduler: HardwareScheduler, model_path: str = "checkpoints/"):
        self.sched      = scheduler
        self.model_path = model_path
        self.cfg        = scheduler.get_config()
        self.hw         = scheduler.hw

        self.vision_model    = None
        self.llm_model       = None   # PyTorch full model OR llama_cpp.Llama
        self.tokenizer       = None
        self.image_processor = None
        self._full_model     = None   # full PyTorch model (kept for diffusion)

    def load(self):
        self._load_base_model()
        self._load_vision()
        self._load_llm()
        logger.info("All components loaded per hardware schedule.")
        return self

    # ── Base model (always needed for diffusion + tokenizer) ──────────────────
    def _load_base_model(self):
        import warnings; warnings.filterwarnings("ignore")
        from mobileo.utils import disable_torch_init
        from mobileo.model.builder import load_pretrained_model
        import torch

        disable_torch_init()
        tokenizer, model, _ = load_pretrained_model(self.model_path)
        device = self.cfg.torch_device
        dtype  = self.cfg.torch_dtype
        model.to(dtype).to(device).eval()

        self.tokenizer       = tokenizer
        self.image_processor = model.get_vision_tower().image_processor
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        self._full_model = model

    # ── Vision encoder ────────────────────────────────────────────────────────
    def _load_vision(self):
        vd = self.cfg.vision_device

        if vd == "ane" and self.cfg.use_coreml_vision:
            import coremltools as ct
            self.vision_model = ct.models.MLModel(
                self.hw.coreml_vision_path,
                compute_units=ct.ComputeUnit.ALL
            )
            logger.info("Vision encoder loaded on ANE (CoreML)")

        elif vd in ("mps", "cpu", "cuda"):
            # Use PyTorch vision tower already inside the full model
            self.vision_model = self._full_model  # generate() handles vision internally
            logger.info(f"Vision encoder using PyTorch on {vd}")

        elif vd == "npu_qnn":
            try:
                import onnxruntime as ort
                so = ort.SessionOptions()
                self.vision_model = ort.InferenceSession(
                    "vision_encoder.onnx",
                    providers=["QNNExecutionProvider"],
                    sess_options=so
                )
                logger.info("Vision encoder loaded on Qualcomm QNN NPU")
            except Exception as e:
                logger.warning(f"QNN load failed ({e}), falling back to CPU")
                self.vision_model = self._full_model

        else:
            # DLA, APU, unknown — fall back to PyTorch on detected device
            logger.warning(f"Vision device '{vd}' loader not fully implemented, using PyTorch")
            self.vision_model = self._full_model

    # ── LLM ───────────────────────────────────────────────────────────────────
    def _load_llm(self):
        ld = self.cfg.llm_device

        if ld in ("mps_int4", "cuda_int4", "cpu_int4") and self.cfg.use_gguf_llm:
            try:
                from llama_cpp import Llama
                n_gpu = 99 if ld in ("mps_int4", "cuda_int4") else 0
                self.llm_model = Llama(
                    model_path=self.hw.gguf_llm_path,
                    n_gpu_layers=n_gpu,
                    n_ctx=1024,
                    verbose=False,
                )
                logger.info(f"LLM loaded as INT4 GGUF via llama_cpp (gpu_layers={n_gpu})")
            except ImportError:
                logger.warning("llama_cpp not installed, falling back to PyTorch LLM")
                self.llm_model = self._full_model

        elif ld in ("mps", "cuda", "cpu"):
            self.llm_model = self._full_model
            logger.info(f"LLM using full PyTorch model on {ld}")

        else:
            logger.warning(f"LLM device '{ld}' not implemented, using PyTorch")
            self.llm_model = self._full_model

    # ── Convenience: run understanding inference with correct path ─────────────
    def run_understanding(self, image_path: str, prompt: str) -> str:
        """
        Run image understanding using scheduled hardware.

        Vision encoding: ANE (CoreML) if available, otherwise MPS/CUDA.
        LLM decode: always uses full PyTorch multimodal model — GGUF is
        text-only (cannot receive image features) so it is not used here.
        The GGUF speedup applies to text-only tasks; multimodal understanding
        requires the full model until llama.cpp adds multimodal GGUF support.
        """
        import torch
        from PIL import Image
        from mobileo.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from mobileo.mm_utils import tokenizer_image_token, process_images
        from mobileo.conversation import conv_templates
        import time

        device = self.cfg.torch_device
        dtype  = self.cfg.torch_dtype

        raw    = Image.open(image_path).convert("RGB")
        tensor = process_images([raw], self.image_processor, self._full_model.config)[0]
        image_batch = tensor.unsqueeze(0).to(dtype).to(device)

        # ── Vision: ANE via CoreML (fast), MPS fallback ───────────────────────
        t0 = time.perf_counter()
        if self.cfg.vision_device == "ane" and self.cfg.use_coreml_vision:
            pixel_np = tensor.unsqueeze(0).float().numpy()
            self.vision_model.predict({"pixel_values": pixel_np})
            vision_backend = "ANE (CoreML)"
        else:
            with torch.inference_mode():
                self._full_model.visual(image_batch)
            vision_backend = self.cfg.vision_device
        t_vision = (time.perf_counter() - t0) * 1000

        # ── LLM: full PyTorch multimodal (GGUF can't take image features) ─────
        qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv = conv_templates["qwen_2"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        input_ids = tokenizer_image_token(
            conv.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(device)

        t0 = time.perf_counter()
        with torch.inference_mode():
            output_ids = self._full_model.generate(
                input_ids, images=image_batch,
                do_sample=True, temperature=0.8, top_p=None,
                num_beams=1, max_new_tokens=128, use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        t_llm = (time.perf_counter() - t0) * 1000

        answer = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(f"  [scheduler] vision={vision_backend} {t_vision:.0f}ms  |  LLM=PyTorch({device}) {t_llm:.0f}ms")
        return answer


# ══════════════════════════════════════════════════════════════════════════════
# CLI — run this file directly to see hardware report
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.WARNING)

    print("Probing hardware...")
    sched = HardwareScheduler()
    sched.print_report()

    summary = sched.summary_dict()
    print("Machine-readable summary:")
    print(json.dumps(summary, indent=2))

    # Optional: run a quick understanding inference
    if "--run" in sys.argv:
        from pathlib import Path
        img = "assets/cute_cat.png"
        if not Path(img).exists():
            print(f"Image {img} not found, skipping inference test.")
        else:
            print(f"\nRunning scheduled inference on {img}...")
            loader = ScheduledModelLoader(sched, model_path="checkpoints/")
            loader.load()
            answer = loader.run_understanding(img, "What is in the image?")
            print(f"Answer: {answer}")
