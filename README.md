### This repo is for Paper *Contrastive Learning Based Visual Representation Enhancement for Multimodal Machine Translation*
---


#### In simple, run 'process_imgdata.sh' first and next 'run.sh' for training and inference

1. First of all, the image data needs to be downloaded by person.
- With a request to https://forms.illinois.edu/sec/229675, you will get the Flickr30k image dataset, which contains images for train, valid and test_2016_flickr.
- Test_2017_flickr is in https://drive.google.com/drive/folders/1kfgmYFL5kup51ET7WQNxYmKCvwz_Hjkt
- MSCOCO is in http://images.cocodataset.org/#download
- Test_2018_flickr is in https://drive.google.com/drive/folders/1kfgmYFL5kup51ET7WQNxYmKCvwz_Hjkt

The way of processing image data is written in 'process_imgdata.sh'
Run the shell file, all image data will be the same as we used in our work

2. 'run.sh' is the code for running our code.
  It contains how to preprocess text data, train, inference and evaluation
