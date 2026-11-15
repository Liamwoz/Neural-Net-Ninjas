# Touchline frontend

A local dashboard for reviewing footage alongside projected player and ball positions. The sample positions are illustrative and do not run inference. To start it from the repository root:

```powershell
python -m pip install -r requirements-yolo.txt
python frontend/server.py
```

Open http://localhost:8000. Choose a browser-supported video and select **Analyze video**. The server first uses `model/pitch_keypoints.pt` and camera motion to estimate and smooth the pitch homography, then uses `model/weights_better_model.pt` to detect players and the ball. Both run locally, without Roboflow credits. Player bottom-centres and ball centres are projected onto the pitch and synchronized with the video; boxes remain visible even when projection is unavailable. The progress display shows the two stages. Models are loaded on first use and reused. Videos up to 512 MB are supported and temporary uploads are removed after analysis. Restart the server after updating the app.

The pitch keypoint model was trained on this project's clip and may not work on other camera angles or matches. The interface reports frames without a homography. Detections outside the pitch are not displayed on the map. The local analysis matches player detections across frames and uses the project’s jersey classifier to estimate red and green teams, the referee and the goalkeeper. Unknown labels are shown until there is enough evidence; early frames can be filled from later confirmed assignments. The goalkeeper rules are specific to this match, and IDs and teams can still be wrong during occlusions. You can also load projected JSON, and the video controls synchronize either source with playback.

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
