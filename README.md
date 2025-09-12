# Vehicle State of Health (SoH) using AI/ML

## Problem Statement
Vehicle State of Health (SoH) is an emerging application of AI/ML for automotive. Given the sheer number of Electronic Control Units (ECUs) in a modern vehicle and the network communication that occurs between these ECUs and now the cloud for connected vehicles, it is an obvious use case to apply Natural Language Processing (NLP) techniques to understand the basis of these communications. Furthermore, it seems reasonable to gather information on vehicle SoH beyond the traditional Diagnostic Trouble Codes (DTCs). 

This project aims to just scratch the surface of SoH possibilities by focusing on one specific yet important use case—**12V battery SoH**. For both EVs and ICE vehicles it is the 12V battery (“LV battery") that drives the main ECUs that control the vehicle. For EVs to the large battery (“HV battery") that drives the powertrain is also used to periodically recharge the LV battery. It is this cycling that reduces the life of the 12V battery. Therefore, monitoring the LV battery SoH is a critical aspect in understanding vehicle SoH, predictive maintenance, and even OTA adjustments that may be used to extend the LV battery life. 

Therefore, we will analyze these LV battery charging cycles under various loading conditions to calculate and model LV battery SoH. 

## Approach
