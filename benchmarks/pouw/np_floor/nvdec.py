"""The video decode engines' (NVDEC) output rate on the card, through PyNvVideoCodec and the driver's libnvcuvid.

ffmpeg on the host encodes a smooth and a noise stream in H.264 and HEVC. On each stream, P processes each decode the
whole stream R times with a decoder of their own, so the engines run side by side, and the rate is every process's
NV12 output bytes over the span from the first start to the last end. The first K frames, decoded to host memory, are
compared byte for byte with ffmpeg's CPU decode. Prints one JSON line a stream and process count.

  python nvdec.py make WORKDIR [W H FRAMES]          # encode the streams (no GPU)
  python nvdec.py WORKDIR [W H FRAMES R K PROCS...]
"""

import ctypes
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time

import imageio_ffmpeg
import numpy as np
import PyNvVideoCodec as nvc

FF = imageio_ffmpeg.get_ffmpeg_exe()
SOURCES = {"smooth": "testsrc2=s={w}x{h}:r=30", "noise": "testsrc2=s={w}x{h}:r=30,noise=alls=100:allf=t+u"}
CODECS = {"h264": ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "18"],
          "hevc": ["-c:v", "libx265", "-preset", "ultrafast", "-crf", "18", "-x265-params", "log-level=error"]}


def make(work, src, codec, w, h, frames):
    path = os.path.join(work, f"{src}-{codec}-{w}x{h}.mp4")
    if not os.path.exists(path):
        subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", SOURCES[src].format(w=w, h=h),
                        "-frames:v", str(frames), "-pix_fmt", "yuv420p", *CODECS[codec], path], check=True)
    return path


def decode(path, reps, keep=0):
    frames = out = 0
    kept = []
    for _ in range(reps):
        dmx = nvc.CreateDemuxer(filename=path)
        dec = nvc.CreateDecoder(gpuid=0, codec=dmx.GetNvCodecId(), usedevicememory=keep == 0)
        for pkt in dmx:
            for fr in dec.Decode(pkt):
                frames += 1
                out += fr.framesize()
                if len(kept) < keep:
                    kept.append(bytes((ctypes.c_uint8 * fr.framesize()).from_address(fr.GetPtrToPlane(0))))
        del dec, dmx
    return frames, out, kept


def worker(path, reps, barrier, q):
    decode(path, 1)
    barrier.wait()
    t0 = time.perf_counter()
    frames, out, _ = decode(path, reps)
    q.put((t0, time.perf_counter(), frames, out))


def run(path, reps, procs):
    ctx = mp.get_context("spawn")
    barrier, q = ctx.Barrier(procs), ctx.Queue()
    ps = [ctx.Process(target=worker, args=(path, reps, barrier, q)) for _ in range(procs)]
    for p in ps:
        p.start()
    rs = [q.get() for _ in ps]
    for p in ps:
        p.join()
    span = max(r[1] for r in rs) - min(r[0] for r in rs)
    return span, sum(r[2] for r in rs), sum(r[3] for r in rs)


def main():
    if sys.argv[1] == "make":
        work, args = sys.argv[2], sys.argv[3:6] or ["1920", "1080", "240"]
        for src in SOURCES:
            for codec in CODECS:
                print(make(work, src, codec, *(int(a) for a in args)), flush=True)
        return
    work = sys.argv[1]
    w, h, nframes, reps, keep = (int(a) for a in (sys.argv[2:7] or ["1920", "1080", "240", "4", "8"]))
    procs = [int(a) for a in sys.argv[7:]] or [1, 7, 14]
    for src in SOURCES:
        for codec in CODECS:
            path = make(work, src, codec, w, h, nframes)
            dmx = nvc.CreateDemuxer(filename=path)
            caps = nvc.GetDecoderCaps(gpuid=0, codec=dmx.GetNvCodecId(), chromaformat=dmx.ChromaFormat(),
                                      bitdepth=dmx.BitDepth())
            n, _, kept = decode(path, 1, keep)
            ref = subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-i", path, "-frames:v", str(keep), "-f",
                                  "rawvideo", "-pix_fmt", "nv12", "-"], check=True, capture_output=True).stdout
            got = b"".join(kept)
            row = {"stream": src, "codec": codec, "w": w, "h": h, "frames": n, "frames_expected": nframes,
                   "stream_bytes": os.path.getsize(path), "num_decoder_engines": caps.get("num_decoder_engines"),
                   "frame_bytes": len(kept[0]) if kept else None, "checked_frames": len(kept),
                   "bytes_differ_vs_cpu": int(np.count_nonzero(np.frombuffer(got, np.uint8) != np.frombuffer(ref, np.uint8)))
                   if len(got) == len(ref) else None,
                   "cpu_ref_bytes": len(ref), "nvdec_bytes": len(got)}
            for p in procs:
                span, frames, out = run(path, reps, p)
                row[f"p{p}"] = {"seconds": span, "frames": frames, "pixels_per_s": frames * w * h / span,
                                "out_bytes_per_s": out / span}
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
