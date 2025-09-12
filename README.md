# Vehicle State of Health (SoH) using AI/ML

## Problem Statement
Vehicle State of Health (SoH) is an emerging application of AI/ML for automotive. Given the sheer number of Electronic Control Units (ECUs) in a modern vehicle and the network communication that occurs between these ECUs and now the cloud for connected vehicles, it is an obvious use case to apply Natural Language Processing (NLP) techniques to understand the basis of these communications. Furthermore, it seems reasonable to gather information on vehicle SoH beyond the traditional Diagnostic Trouble Codes (DTCs). 

This project aims to just scratch the surface of SoH possibilities by focusing on one specific yet important use case—**12V battery SoH**. For both EVs and ICE vehicles it is the 12V battery (“LV battery") that drives the main ECUs that control the vehicle. For EVs to the large battery (“HV battery") that drives the powertrain is also used to periodically recharge the LV battery. It is this cycling that reduces the life of the 12V battery. Therefore, monitoring the LV battery SoH is a critical aspect in understanding vehicle SoH, predictive maintenance, and even OTA adjustments that may be used to extend the LV battery life. 

Therefore, we will analyze these LV battery charging cycles under various loading conditions to calculate and model LV battery SoH. 

## Approach
Due to the critical nature of LV battery SoH, most modern vehicles are equipped with an Intelligent Battery Sensor (IBS). The IBS monitors all the critical battery parameters including current draw, internal resistance, temperature to determine its own SoH indicator that is then used by the main vehicle controllers. This is done via a standardized algorithmic approach. 

The aim of this project is to compare the IBS SoH with a ML approach that uses the (1) IBS parameters alone to develop a comparative SoH indicator and (2) expand the parameters to the ~1000 vehicle signals with possibly identify an even more accurate 12V battery SoH metric.

## Data Acquisition
Acquisitions of vehicle network traffic was done using Vector CANalyzer tools to log and record data. These data files contain  massive amounts of data and are stored in Vectors proprietary blf binary file format. 

## Data Pre-Processing
The first step was to convert the Vector blf files to a readable '.csv' that then can be more easily digested by ML models. Again, due to the size of the files as they are recorded at ms intervals it is necessary to convert and down sample at the same time. While there python libraries developed to convert the blf format, I found them difficult and buggy. Therefore, I used Python BLFReader from the python-can library as the basis for developing my own converter and downsampling script that I could use. I do not provide the code here for this converter but may provide it in a separate GitHub repository at a later time. 

After converting and down sampling the data to a 1 second rate, I then began the process of pre-processing and cleaning the data. 
First approach was to only use IBS signals itself. Therefore, all data other than timestamp and IBS vehicle signals were removed. Followed by converting not numeric signals to numeric. 

Second approach was to do feature engineering on the complete set of vehicle signals to find the imost mportant features that correlate to LV battery SoH. From there the top features were cleaned for only numeric. 

## Modeling Approach
Initially I used the only the IBS signals themselves with a standard algorithm to compute a SoH that could be compared against that reported by the IBS **(SOH1)**.

<p align="center">

$$
SOH_{1} = w_C \cdot SOH_{capacity} + (1 - w_C) \cdot SOH_{resistance}
$$

</p>

**SOH_capacity** ≈ ratio of measured capacity (Ah) vs nominal capacity.\
**SOH_resistance** ≈ normalized inverse of internal resistance.\
$$w_C$$ is s a weighting factor (typically favoring capacity when discharge cycles are available, and resistance when only charge cycles or current pulses are available).\

Applied **penalty factors from IBS diagnostic flags:**\
**Sulfation flag** → apply a 10% reduction (×0.9).\
**Defect flag** → apply a harsher reduction.

<p align="center">

$$
SOH_{final} = SOH_{1} \times P_{flags}
$$

</p>

<p align="center">

$$
P_{flags} =
\begin{cases} 
0.90 & \text{if sulfation = 1} \\
0.80 & \text{if defect = 1} \\
1.00 & \text{otherwise}
\end{cases}
$$

</p>


Second, I used ML models on the same IBS data to determine a SoH **(SOH2)**. Third, I used the top vehicle network features discovered along with various ML models (from Linear Regression to DNNs) to attempt to improve the LV SoH even further **(SOH3)**. 

## Modeling Evaluation
ML models used for this project were as follow:
- Linear Regresssion
- Random Forest Decision Trees
- SVMs (Support Vector Machines)
- ARIMA (AutoRegressive Integrated Moving Average)
- Keras CNNs (Convolutional Neural Networks)
- RNN (Regressive Neural Network)

## Results & Insights
tbd

## Next Steps
tbd
