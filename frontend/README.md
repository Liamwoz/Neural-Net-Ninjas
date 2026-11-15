# Touchline frontend

A local dashboard for reviewing footage alongside projected player and ball positions. The sample positions are illustrative and do not run inference. To start it from the repository root:

```powershell
python -m pip install -r requirements-yolo.txt
python frontend/server.py
```

Open http://localhost:8000. Choose a browser-supported video and select **Analyze video** to run `model/weights_better_model.pt` locally. Analysis processes every frame. The progress bar shows processed frames and estimates the remaining time from the observed speed; the estimate can change as processing continues. Restart the local server after updating the app. Once finished, detections appear as boxes during playback. You can also load a JSON file containing homography output. The video controls drive both views. Without a video, the same controls play the loaded timeline. Video analysis sends the selected file to the local server on your computer; no remote service is used. The server runs YOLO; it does not calculate homography. The pitch remains empty for YOLO-only results because image coordinates are not projected pitch coordinates. The model is loaded on first analysis and reused. Videos up to 512 MB are supported and temporary uploads are removed after analysis.

## Data contract

Provide a JSON object in this form:

```json
{
  "name": "Match clip",
  "duration": 20,
  "frames": [
    {
      "time": 0.0,
      "objects": [
        {"type": "player", "id": 7, "team": "team_a", "x": 35.2, "y": 48.1, "bbox": [0.21, 0.28, 0.06, 0.24]},
        {"type": "player", "id": 4, "team": "team_b", "x": 62.0, "y": 53.0},
        {"type": "ball", "x": 50.0, "y": 50.0}
      ]
    },
    {"time": 0.1, "objects": []}
  ]
}
```

- `time` is elapsed seconds from the start of the video. Frames may be unordered; the UI sorts them and displays the latest frame at or before the current time. An empty `objects` array clears the map for that frame. Include frames throughout the clip so stale detections are not retained.
- `x` and `y` are projected pitch coordinates as percentages from 0 to 100. `(0, 0)` is the top-left corner and `(100, 100)` is the bottom-right corner. Convert your homography's coordinate system to this convention before export.
- `type` is `player`, `ball`, or `other`. `team` is optional: `team_a`, `team_b`, or omitted for other/unknown. `id` is optional and shown on the map.
- Optional `bbox` is `[x, y, width, height]` in normalized video coordinates (0–1), measured from the top-left of the original video. These boxes are drawn on the footage. Use an already annotated video if your pipeline renders its own boxes.
- `duration` is optional and is used when no video is loaded. With a video, the video duration takes precedence. The pitch dimensions shown are a visual reference; normalized coordinates do not require a particular physical pitch size.

If you already have a CSV of image-space detections, it must first be transformed and exported to this JSON format by the model/homography pipeline. No integration with that pipeline is assumed here.
