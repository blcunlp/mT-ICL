#tables=3.6.1
#torch==1.4.0
#torchtext=0.9.1
#torchvision==0.5.0

pip install --editable ./
pip install fastBPE sacremoses subword_nmt

#download text data
git clone --recursive https://github.com/multi30k/dataset.git multi30k-dataset
cd /path/to/multi30k-dataset
bash script/task1-bpe.sh -m 10000
#this will automatically generate bpe tokens
cd ..

#the way of accessing and processing image data is written in process_imgdata.sh

#preprocess text
TEXT=/path/to/multi30k-dataset/data/task1/bpe10000/en-de
python3 preprocess.py --source-lang en --target-lang de \
        --trainpref $TEXT/train.lc.norm.tok.bpe --validpref $TEXT/val.lc.norm.tok.bpe --testpref $TEXT/test_2016_flickr.lc.norm.tok.bpe,$TEXT/test_2017_flickr.lc.norm.tok.bpe,$TEXT/test_2017_mscoco.lc.norm.tok.bpe,$TEXT/test_2018_flickr.lc.norm.tok.bpe\
        --destdir data-bin --workers 20 --joined-dictionary


#train the model

DATA_DIR=/path/to/data-bin
imgs_path=/path/to/multi30k-dataset/images
imgs_feat_path=/path/to/visual/features.pth    

python3 train.py ${DATA_DIR} --task translation \
      --arch transformer_wmt_en_de --share-all-embeddings \
      --save-dir path/to/checkpoints \
      --source-lang en --target-lang de \
      --encoder-layers 4 --decoder-layers 4 \ 
      --encoder-attention-heads 4 --decoder-attention-heads 4 \ 
      --encoder-embed-dim 128 --decoder-embed-dim 128 \
      --encoder-ffn-embed-dim 256 --decoder-ffn-embed-dim 256 \
      --dropout 0.3 --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
      --lr-scheduler new_inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 2000 \
      --lr 0.005 \
      --temperature 55 --use-awl --criterion margin_label_smoothed_cross_entropy \
      --label-smoothing 0.1 --weight-decay 0.0 \
      --max-tokens 2048 --update-freq 4 --max-update 7500 \
      --no-progress-bar --log-format simple \
      --image-path ${imgs_path} --image-resize 380 --image-color 0.5 \
      --image-hidden 1024 \
      --pretrained-img-model-path ${img_feat_path} \
      --distributed-world-size 1 

#test
TEXT=/path/to/multi30k-dataset/data/task1/bpe16000/en-de
MODEL_DIR=path/to/checkpoints
ref=/path/to/multi30k-dataset/data/task1/tok/test_2016_flickr.lc.norm.tok.de

#average last 10 epochs
model=${MODEL_DIR}/checkpoint_avg10.pt
python3 scripts/average_checkpoints.py \
        --inputs ${MODEL_DIR} --num-epoch-checkpoints 10 \
        --output ${model}

python3 interactive.py ${DATA_DIR} \
        --input ${TEXT}/test_2016_flickr.lc.norm.tok.bpe.en \
        --path ${model} --beam 5 --remove-bpe --lenpen 0.6 \
        --image-path ${imgs_path} \
        --source-lang en --target-lang de --batch-size 64 --buffer-size 2000 \
        > en2de_2016.log

grep ^H en2de_2016.log | cut -f3  > 2016.de.pred
perl multi-bleu.perl ${ref} < 2016.de.pred 
python3 meteor_score.py --hypo 2016.de.pred --ref ${ref} 
