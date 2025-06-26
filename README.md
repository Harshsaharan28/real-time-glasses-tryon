#  Real time glasses try-on 

A real-time virtual try-on system for glasses that lets users scroll through a catalog of eyewear and see how each style fits their face using their webcam. It automatically aligns glasses to your eyes and rotates them based on your head tilt for a realistic experience.

**Recommendation system coming soon!** Glasses will be auto-suggested based on face shape and preferences.

---

## Features

- Scroll through multiple glasses styles (`N`)
- Live webcam-based face tracking
- Glasses auto-align to your eyes
- Rotates glasses based on face tilt
- Manual adjustments with keyboard:
  - `+ / -` to resize
  - `W A S D` to move
  - `R` to reset
  - `S` to save snapshot
- Catalog of `.png` glasses overlays
- Saves snapshots in `captures/` folder

---

## Tech Stack

- Python 3.x
- OpenCV
- MediaPipe (FaceMesh)
- NumPy

---

## Coming Soon

  - Face shape detection
  - Glasses recommendation engine
  - tyle filters by frame type (round, square, etc.)
  - Streamlit web interface

