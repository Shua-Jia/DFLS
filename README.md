# Domian Fusion in Latent Space (DFLS) for Face Video Super-Resolution
This is the program page of our latest DFLS approach for Face Video Super-Resolution(FVSR).   Version 1


[![Face Video Super-Resolution Demo 01](https://img.youtube.com/vi/DVElNkSY9WM/0.jpg)](https://youtu.be/DVElNkSY9WM)


This is our 8X super-resolution demo, you can click this frame, and then will auto-redirect to YouTube playback.

[![4X Face Video Super-Resolution](https://img.youtube.com/vi/fY6T_epeDSQ/0.jpg)](https://youtu.be/fY6T_epeDSQ)

[![8X super-resolution demo](https://img.youtube.com/vi/JLFu-V0Dk_E/0.jpg)](https://youtu.be/JLFu-V0Dk_E)

This is our 4X super-resolution demo, you can click this frame, and then will auto-redirect to YouTube playback.

### Dataset
you can download training data from https://drive.google.com/drive/folders/19DLr27P9xMOTn_W6hxpxxm8_5jJoX-nR

### Training
Train the model by following the command lines below
python train.py

Details:
--data_path: [should be filled in the directory to the training dataset]
--ckpt_path: [the location of your training model]

### Inference
After the training you can run the following command to FVSR for evaluation.

python eval.py


### Related Code
https://github.com/GreyCC/DTLS_1024
https://github.com/yangxy/GPEN
https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement
https://github.com/LabShuHangGU/MIA-VSR
