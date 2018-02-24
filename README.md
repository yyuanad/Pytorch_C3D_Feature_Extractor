# Pytorch_C3D_Feature_Extractor

pre-trained model (on sport1M) is available:

[C3D_sport.pkl](http://imagelab.ing.unimore.it/files/c3d_pytorch/c3d.pickle)

### input: video
```
python feature_extractor_vid.py -l 6 -i /data/miayuan/videos/ -o /data/miayuan/c3d_features -gpu -id 0 -p /data/miayuan/video_list.txt --OUTPUT_NAME c3d_fc6_features.hdf5
```

### input: frames
```
python feature_extractor_frm.py -l 6 -i /data/miayuan/frames/ -o /data/miayuan/c3d_features -gpu -id 0 -p /data/miayuan/video_list.txt --OUTPUT_NAME c3d_fc6_features.hdf5
```

