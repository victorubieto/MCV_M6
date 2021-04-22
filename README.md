# M6 Video Analysis - MCV

## Video Surveillance for Road Traffic Monitoring

### Team 7
| Members        |  Mail                           | Github |
| :---           | ---:                            | ---: |
| Alex Tempelaar | alexander.tempelaar@e-campus.uab.cat | Tempe25 |
| Víctor ubieto  | victor.ubieto@e-campus.uab.cat   | victorubieto |
| Mar Ferrer     | mar.ferrerf@e-campus.uab.cat  | namarsu |
| Antoni Rodríquez| antoni.rodriguez@e-campus.uab.cat  | antoniRodriguez |


## Project Schedule
- Presentation Slides [here](https://docs.google.com/presentation/d/1urabVFpes0Lc_ao0FNEmwvkLzny8GpsogwU15xpdJy4/edit?usp=sharing)
- Report in LateX [here]()
- Tasks
   - Multi-target Single-camera Tracking: This tasks consists on connecting the detected objects between frames taken by a single camera. In our case, we have some videos of a street where we are detecting the moving cars. Our goal is to track the car along all its appearance in the video.
   - Multi-target Multi-camera Tracking: This tasks consists on connecting the detected objects between frames taken by a set of cameras. Each camera is located at a different location, therefore, our job is to track each car along all the frames of all the cameras (it can be seen by two cameras at the same time).
- Dataset used: In this project we used the dataset from [CVPR 2020 AI City Challenge](https://www.aicitychallenge.org/), specifically the sequence 03 for testing, and sequences 01 and 04 for training. 


## Process to run the code
1. Add paths to data files:
   - Task 1
     - Detections: --Line 16 
     - Ground truth: --Line 17
     - Video: --Line 18
   - Task 2
     - Ground truth: --Line 335
     - Sequence 03 paths: --Line 336
     - Pickles path (optional): --Lines 375-376
2. (Optional) Tweak parameters and flags as desired:
     - Task 1: --Lines 16 and 21-35
     - Task 2: --Lines 379-382
3. Select the task to run leaving it uncommented: --Lines 496-497
4. Run: >> python lab5.py
