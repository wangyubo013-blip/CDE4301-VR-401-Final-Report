# CDE4301-VR-401-Final-Report
This repository contains the project website for the CDE4301 Final Report.

**Team Members** 
- Liu Yuhui (Ula) 		(A0266685L)
- Wang Yubo 			(A0262295Y)
- Zhang Yining 			(A0258906R)
- Gao Jiquan 			(A0258910B)

**Affiliation:** Singapore Airlines – NUS Digital Aviation Corp Lab  
**Semester:** AY2025/26 Sem 2

## Project Website

The complete final report is published here: [https://wangyubo013-blip.github.io/CDE4301-VR-401-Final-Report/](https://wangyubo013-blip.github.io/CDE4301-VR-401-Final-Report/)

## Repository Structure

```
.
├── Scripts/                                # All processing scripts and data pipelines
│   ├── Workstream1_SpeechRecognition/      # Speech recognition workstream
│   │   ├── 1. EventlogPreprocess/          # Event log preprocessing scripts
│   │   ├── 2. DatasetPreparation/          # Dataset preparation pipeline
│   │   ├── 3. WERAnalysis/                 # Word Error Rate analysis
│   │   └── 4. Results/                     # Speech recognition results
│   └── Workstream2_PhysiologicalSensors/   # Physiological sensor workstream
│       ├── MIST_test/                      # MIST stress test scripts
│       ├── data_cleaning/                  # Sensor data cleaning pipeline
│       ├── raw_data_of_subjects/           # Raw physiological data from subjects
│       ├── real-world_VR_training_data/    # Real-world and VR training datasets
│       └── sensor_data_collection/         # Sensor data collection tools
├── assets/                                 # Website assets
│   ├── images/                             # Project images and figures
│   └── samples/                            # Project event log samples
├── faster_whisper/                         # Faster Whisper model integration
│   ├── whisper_training/                   # Whisper model training scripts
│   └── add_model_here.txt                  # Placeholder for model weights
├── index.html                              # Main report webpage
└── README.md                               # This file
```

## Supplementary Materials

In addition to the website version of the report, the repository includes:

Workstream 1:

1. Fine-tuned model
2. Data Preparation Pipeline
3. Test Results

Workstream 2:

1. MIST stress test scripts
2. Raw physiological data of subjects
3. Sensor data collection tools
4. Data cleaning pipeline
5. Real-world and VR training data

## Notes for Reviewers

You may view the full report directly via the website link above.  
For detailed scripts and data, please refer to the `Scripts/` directory, which is organised by workstream.
