# Infant Movement Analysis and Tracking – Summer 2025 Research

This repository contains code and experiments for analyzing and tracking infant movement patterns.  
The project explores two main research directions:

- **Filtering methods** for optimal trajectory smoothing  
- **Optimal control approaches** for identifying divergence points in infant development  

---

## Author
**Ioana Gidiuta**  
BSE Electrical Engineering, MSE Data Science  
University of Pennsylvania  

---

## Repository Structure
- **smoothing.ipynb** – Main notebook for exploring filtering methods (shifts in feature distribution; frame-wise and window-wise displacements of keypoints)
- **feature_extraction.ipynb** - Notebook for computing 6 sets of features: posture, symmetry, smoothness, range, variability, effort (for different filtering methods)
- **ioc_divergence.ipynb** - Notebook used for quantifying the effect of feature sets (obtained via different fitering methods) on donwnstream clasiffication  
- **constants.py** – Project constants used across scripts  
- **feature_computation.py** – Functions for computing movement features (velocity, acceleration, etc.)  
- **feature_extraction.py** – Extracts features for 4 keypoint sets: wrists, elbows, knees, ankles  
- **filters.py** – Implementation of filters and filter dictionary  
- **preprocessing_utils.py** – Interpolation utilities  
- **utils.py** – Miscellaneous helper functions  
