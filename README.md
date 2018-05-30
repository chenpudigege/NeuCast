# Datasets
- SC data: the dataset contains power grid series of 133 locations, and the location index, date, hour, temperature, precipitation, active power and reactive power is reported in the dataset.

# Usage
- For CMU data:  
  > python NeuCast_CMU.py --method sar --day_pred 5 --end 23    
  - method: the smoothing method used in the algorithm, including 'sar','ar','hw'
  - day_pred: control the predict length of the algorithm.
  - end: control the end day of the dataset.

- For SC data:  
  > python NeuCast_SC.py --loc_id 0 --method hw
  - loc_id: the location id, from 0~132
  - method: the smoothing method used in the algorithm, including 'sar','ar','hw'
  

# Require
- Keras (2.0.9) with tensorflow backend
- rpy2 (2.9.1)


# Reference
If you use the data or the NeuCast algorithom, please cite our work.
>@inproceedings{neu2018,  
>&emsp;&emsp;title={NeuCast: Seasonal Neural Forecast of Power Grid Time Series},  
>&emsp;&emsp;author={Chen, Pudi and Liu, Shenghua and Shi, Chuan and Bryan Hooi and Wang, bai and Cheng, Xueqi},  
>&emsp;&emsp;booktitle={International Joint Conference on Artificial Intelligence},  
>&emsp;&emsp;year={2018},  
>}