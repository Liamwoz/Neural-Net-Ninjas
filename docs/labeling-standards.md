# Labeling standards (Week 1)

## Target format
Use YOLO detection labels (`class x_center y_center width height`), normalized to frame width/height.

## Class definitions
- `0` – player_team_a
- `1` – player_team_b
- `2` – goalkeeper_team_a
- `3` – goalkeeper_team_b
- `4` – referee
- `5` – ball

## Box rules
- Draw tight boxes around visible object boundaries
- Include partially visible objects if at least ~30% of the object is visible
- Ignore crowd, coaches, and non-field objects unless they overlap heavily with a target class

## Occlusion rules
- If object identity is clear, keep labeling through moderate occlusion
- If identity/class is uncertain, do not label the object for that frame

## Quality checks
- Class IDs match this document
- Box stays on the same tracked entity across adjacent frames
- Ball labels are reviewed a second time because they are high-impact for downstream mapping
