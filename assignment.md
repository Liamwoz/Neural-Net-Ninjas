# Computer Vision Challenge: Sports Vision & Tactical Mapping

## Why this Challenge?

Real-world computer vision engineering is not about running `model.predict()` or copying boilerplate pipeline scripts. In modern industry setups—such as sports broadcasting, automated refereeing, and performance scouting—engineers must combine multiple vision techniques into a single cohesive pipeline: data labeling, transfer learning, coordinate transformation, temporal tracking, and deploying a usable interface.

This challenge forces you to move away from isolated model training and build an applied, end-to-end computer vision product, iteratively, and under time constraints.

The goal is mastering the trade-offs, debugging edge cases, and reasoning critically about model behavior. Your output (the functional dashboard) is merely the artifact that demonstrates your engineering process; your true evaluation lies in your conceptual understanding, architectural decisions, and the transparency of your development journey.

## Eligible Sports & Video Scope

- **Allowed Sports:** Soccer, Basketball, Tennis, Ice/Field Hockey (sports where action is predominantly planar, avoiding extreme 3D height variations).
- **Footage Constraints:** Single static broadcast or tactical camera angle (no rapid cuts or extreme panning for your MVP).

**GenAI Disclosure** → check GenAI policy on how to use GenAI tools

## The Core (Minimal Viable Product)

Your MVP demonstrates a sound foundational pipeline taking an .mp4 clip and generating a synchronized 2D tactical map.

### 1. Targeted Data Annotation (Incremental Delta)

- Use an existing pre-trained detector (e.g. standard YOLO weights) as your base for players/referees.
- Identify a high-value data gap: the fast-moving or small object (e.g., puck, tennis ball, soccer ball) or sport-specific equipment.
- Curate and annotate a targeted dataset (40–80 challenging frames) using an assisted labeling workflow (e.g. SAM-assisted annotation or CVAT/Roboflow).
- **Focus:** Document label consistency (how did you handle labelling, because you will be labelling as a team), occlusions (how did you handle or treat occlusions when labelling), and train/val split strategy.

### 2. Model Fine-Tuning & Error Analysis

- Fine-tune a YOLO detector or segmenter on your annotated dataset.
- Quantitatively evaluate with mAP@50 and mAP@50:95.
- **Focus:** Document failure modes. What does the model confuse? Why did specific false positives/negatives occur (link with the annotation part above)?

### 3. Homography & 2D Minimap Projection

- Identify 4 or more known planar field/court keypoints (e.g., penalty box corners, half-court line, baseline intersections).
- Compute a homography matrix using OpenCV (`cv2.findHomography` for instance).
- Project detected player coordinates onto a static 2D top-down template of the pitch/court. The projection must make spatial sense visually.
- **Focus:** Demonstrate conceptual understanding of perspective transformation matrices and planar assumptions.

### 4. Local Deployment

- Build a lightweight local interface (Streamlit, Gradio) to run inference on input .mp4 files and display detections alongside the 2D pitch (the application processes the clip and displays the video with bounding boxes alongside the synchronized 2D tactical pitch).

## The Extra Mile – extra credits (extensions)

Once the MVP is solid, you can choose one or more technical extensions to deepen your investigation:

- **Tracking & Trajectories:** Implement ByteTrack/BoT-SORT or ball trajectory smoothing (Kalman filters).
- **Automated Keypoints:** Train a lightweight keypoint model (e.g., YOLO Pose) to automate homography line detection.
- **Tactical Analytics:** Dynamic team color clustering (HSV/K-Means on jerseys) or spatial occupancy heatmaps (KDE).

## Evaluation & Grading Rubric

The final score prioritizes engineering methodology and conceptual depth over raw dashboard aesthetics.

| Criterion | % | Formative Checkpoints (Process & Iteration) | Final Oral Defense (Mastery & Ownership) |
|---|---:|---|---|
| Data & Annotation Strategy | 25% | Consistent labeling rules, handling edge cases/blur, and solid dataset hygiene (train/val split). | Justifies data trade-offs and explains how annotation quality directly influenced model convergence and performance. |
| Modeling & Fine-Tuning Core | 35% | Structured experimentation beyond defaults, justified augmentations, and rigorous validation metric tracking (mAP). | Explains the underlying vision architecture, training mechanics, loss dynamics, and the rationale behind tuning choices. |
| Failure & Error Analysis | 20% | Proactive auditing of mistakes between milestones and formulating hypotheses to fix recurring errors. | Diagnoses concrete false positives/negatives and traces root causes to data or parameters rather than treating the model as a black box. |
| End-to-End Pipeline & Homography | 10% | Working flow from raw video --> inference --> minimap projection --> Streamlit dashboard | Explains high-level data flow through the pipeline and the practical role of coordinate transformation without needing the underlying matrix math. |
| Process & GenAI Ownership | 10% | Responsiveness to checkpoint feedback; critical, transparent, and documented use of AI tools. | Demonstrates individual code accountability; capable of walking through and defending implementation choices live without AI assistance. |

**Note on Grading Scale:** A fully functional MVP with clean analysis earns up to 85% (Solid/Good). Tackling and successfully defending one of the "Extra Mile" tracks is required to access higher distinction grades (85%+).
