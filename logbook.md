### Dino pretrained weights storing dir . 

  - /root/.cache/torch/hub/checkpoints/dino_vitbase16_pretrain.pth


```
 2023-08-21 08:55:56.319 | INFO     | __main__:<module>:234 - Using evaluation function v2 to print results
Downloading: "https://github.com/facebookresearch/dino/zipball/main" to /root/.cache/torch/hub/main.zip
Downloading: "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth" to /root/.cache/torch/hub/checkpoints/dino_vitbase16_pretrain.pth
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 327M/327M [00:12<00:00, 28.0MB/s]
2023-08-21 08:56:14.010 | INFO     | __main__:<module>:276 - model build
Traceback (most recent call last):
```


### Dataset dir : ./dataset

- How to change : `~/.ssb/ssb_config.json` or `SSB/SSB/utils.py`

### Model Checkpoints 
- model saved to `dev_outputs/simgcd/log/aircraft_simgcd_(21.08.2023_|_38.860)/checkpoints/model.pt`


### Submission
- 그냥, Submission 파일에 있는 것들 직접 읽고, 
- Image load하고 
- Transform 함수는 그대로 쓰고, 
- 이미지 in => logit out 으로 모델 쓰고 
- Output logit을 argmax 해서, Classification 


=> 아직 Classification 과정에 대한 코드와 이해가 부족하다고 생각하고 있음. [8/30, 2023]