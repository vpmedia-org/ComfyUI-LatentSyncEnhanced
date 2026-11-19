# ComfyUI-LatentSyncEnhanced

An enhanced lip-sync node for ComfyUI, built on top of [ByteDance LatentSync 1.6](https://github.com/bytedance/LatentSync).

Improvements over the original wrapper:

| Feature | Original | Enhanced |
|---------|----------|----------|
| Frame with no face | ❌ Crashes | ✅ Warning + pass-through |
| All frames have no face | ❌ Crashes | ✅ Warning + return original video |
| Model path | Node's own `checkpoints/` symlink | ✅ One plain file per weight in a standard ComfyUI model folder |
| OOM on long videos | ❌ Possible | ✅ Configurable `chunk_frames` (default 80, safe for 24 GB VRAM) |

---

## Prerequisites

This node is **fully self-contained** — no other custom nodes are required.
The `latentsync` inference library is bundled directly inside this package.

---

## Model Download & Storage

### Required model files

Each weight is one plain file in a ComfyUI model folder (`MODEL_FILES` in `nodes.py`). Nothing downloads
at run time. The face models go in `models/latentsync/`, a folder that the node registers: ComfyUI lists a
core folder by file extension, and no core folder takes `.onnx`.

| Folder and file | Source | Size | sha256 |
|---|---|---|---|
| `diffusion_models/latentsync_1.6_unet.pt` | [ByteDance/LatentSync-1.6](https://huggingface.co/ByteDance/LatentSync-1.6) `latentsync_unet.pt` | 5.07 GB | `0a478e89eb660f82da4c35dbdde8a5adfb27f99d1b4e50edd03729e1e98316d3` |
| `audio_encoders/latentsync_whisper_tiny.pt` | ByteDance/LatentSync-1.6 `whisper/tiny.pt` | 75.6 MB | `65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9` |
| `vae/sd_vae_ft_mse.safetensors` | [stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse) `diffusion_pytorch_model.safetensors` | 335 MB | `a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815` |
| `latentsync/insightface_buffalo_l_det_10g.onnx` | insightface `buffalo_l.zip` (v0.7) `det_10g.onnx` | 16.9 MB | `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91` |
| `latentsync/insightface_buffalo_l_2d106det.onnx` | insightface `buffalo_l.zip` (v0.7) `2d106det.onnx` | 5.0 MB | `f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf` |

The VAE config ships in this repository (`configs/vae/sd_vae_ft_mse.json`). The two face models run with
onnxruntime (`latentsync/utils/onnx_face.py`), so the insightface package is not needed. The buffalo_l
models are for non-commercial research use (insightface licence).

### Download

```bash
M=ComfyUI/models
hf download ByteDance/LatentSync-1.6 latentsync_unet.pt whisper/tiny.pt --local-dir /tmp/ls
mv /tmp/ls/latentsync_unet.pt $M/diffusion_models/latentsync_1.6_unet.pt
mv /tmp/ls/whisper/tiny.pt $M/audio_encoders/latentsync_whisper_tiny.pt
hf download stabilityai/sd-vae-ft-mse diffusion_pytorch_model.safetensors --local-dir /tmp/vae
mv /tmp/vae/diffusion_pytorch_model.safetensors $M/vae/sd_vae_ft_mse.safetensors
hf download public-data/insightface models/buffalo_l/det_10g.onnx models/buffalo_l/2d106det.onnx --local-dir /tmp/if
mv /tmp/if/models/buffalo_l/det_10g.onnx $M/latentsync/insightface_buffalo_l_det_10g.onnx
mv /tmp/if/models/buffalo_l/2d106det.onnx $M/latentsync/insightface_buffalo_l_2d106det.onnx
```

---

## Node Inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `images` | IMAGE | — | Input video frames |
| `audio` | AUDIO | — | Input audio (any sample rate, auto-resampled to 16 kHz) |
| `seed` | INT | 1247 | Random seed for reproducibility |
| `lips_expression` | FLOAT | 1.5 | Lip movement strength (guidance scale). Range: 1.0–3.0 |
| `inference_steps` | INT | 20 | Diffusion denoising steps. More = slower but higher quality |
| `chunk_frames` | INT | 80 | Frames per processing segment. **Reduce if you hit OOM.** |

### chunk_frames guide (VRAM)

| VRAM | Recommended `chunk_frames` |
|------|---------------------------|
| 24 GB | 80 (default) |
| 16 GB | 48 |
| 12 GB | 32 |
| 8 GB | 16 |

---

## Node Outputs

| Output | Type | Description |
|--------|------|-------------|
| `images` | IMAGE | Lip-synced video frames |
| `audio` | AUDIO | Input audio resampled to 16 kHz |

---

## How no-face frames are handled

- **Some frames have no face**: those frames are passed through unchanged (original pixels). Surrounding frames' face-alignment data is used as nearest-neighbour fallback for the diffusion input so the audio sequence stays intact.
- **All frames have no face**: a warning is printed and the original video is returned as-is (no lip-sync applied, no crash).

---

## Video length vs audio length

Handled automatically inside the pipeline:

| Situation | Behaviour |
|-----------|-----------|
| Audio longer than video | Video is **ping-pong looped** to match audio length |
| Video longer than audio | Video is **trimmed** to audio length |

---

## License

Apache 2.0 — same as the underlying LatentSync model.
