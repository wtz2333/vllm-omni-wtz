#!/usr/bin/env bash
sleep 1

SERVER_LOG="/home/mumura/omni/vllm-omni/scripts/results/server.log"
CURL_LOG="/home/mumura/omni/vllm-omni/scripts/results/curl.log"
echo "starting"
vllm serve /data2/group_谈海生/mumura/models/Wan2.2-T2V-A14B-Diffusers \
    --omni --port 8091 \
    --cfg-parallel-size 2 --ulysses-degree 4 --use-hsdp \
    >> "$SERVER_LOG" 2>&1 &

sleep 600

echo "开始发送请求..."
curl -X POST http://localhost:8091/v1/videos \
  -F "prompt=A cinematic view of a futuristic city at sunset" \
  -F "width=832" \
  -F "height=480" \
  -F "num_frames=33" \
  -F "negative_prompt=low quality, blurry, static" \
  -F "fps=16" \
  -F "num_inference_steps=4" \
  -F "guidance_scale=4.0" \
  -F "guidance_scale_2=4.0" \
  -F "boundary_ratio=0.875" \
  -F "flow_shift=5.0" \
  -F "seed=42" | tee -a "$CURL_LOG"

echo -e "\n请求完成。"
