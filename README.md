# Depth Estimation Using Stereo Cameras
## Overview
Using a stereo camera, depth is determined by the concept of triangulation and stereo matching. By constraining the problem on a 2D plane known as the epipolar plane, we can simplify the formal search to a line search along the epipolar line.

## Approach
1. Identify similar points from feature descriptors.
2. Match feature correspondence using a matching cost function.
3. Using epipolar geometry, find and match correspondence in one picture frame to the other.
4. Compute disparity from known correspondence.
5. Compute depth from known disparity.

## To run the code
```bash
git clone https://github.com/Prat33k-dev/Stereo_Depth_Estimation.git
cd Stereo_Depth_Estimation/code
python3 stereo.py --filePath ../data_files/ --dataset 1
```
### Parameters
- FilePath -  Dataset Path file path. *Default :- '../data_files/'*
- dataset - Dataset Number . *Default :- '1'*

## Results
### CALIBRATION
Matches             |
:-------------------------:|
![](./results/DS1_matches.png) |
![](./results/DS2_matches.png) |
![](./results/DS3_matches.png) |

### RECTIFICATION
Epipolar lines for unrectified images  |
:-------------------------:|
![](./results/DS1_epipolarlines.png) |
![](./results/DS2_epipolarlines.png) |
![](./results/DS3_epipolarlines.png) |


Epipolar lines for rectified images  |
:-------------------------:|
![](./results/DS1_epipolarRectified.png) |
![](./results/DS2_epipolarRectified.png) |
![](./results/DS3_epipolarRectified.png) |

### CORRESPONDENCE
Disparity Map |
:-------------------------:|
![](./results/DS1_disparitygraymap.png) |
![](./results/DS2_disparitygraymap.png) |
![](./results/DS3_disparitygraymap.png) |

### DEPTH
Depth Map |
:-------------------------:|
![](./results/DS1_depthheatmap.png) |
![](./results/DS2_depthheatmap.png) |
![](./results/DS3_depthheatmap.png) |
