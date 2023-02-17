#the image data needs to download online,requested to https://forms.illinois.edu/sec/229675
#after downloading the images, unzip it and run this shell file to seperate into train, valid and test_2016_flickr

images=flickr30k-images
splits=/path/to/image_splits
files=$(ls $splits)
for path in $files
do
  mkdir -p images/${path%\.*}
  cp $splits/$path  images/${path%\.*}/index.txt
  cat $splits/$path | while read line
  do 
    if [ -f "$images/$line" ]; then
      cp $images/$line images/${path%\.*}
    fi
  done
  if [ ${path%\.*} == 'val' ];then
    mv images/${path%\.*} images/valid
  fi
done

#test_2017_flickr & test_2018_flickr
#https://drive.google.com/drive/folders/1kfgmYFL5kup51ET7WQNxYmKCvwz_Hjkt
#download and move all images corresponding document 

#download mscoco test
#wget http://images.cocodataset.org/zips/val2014.zip
#wget http://images.cocodataset.org/zips/train2014.zip
#unzip and rename into test_2017_mscoco
#cd test_2017_mscoco
#for line in $(cat index.txt)
#do
#  if [ -f path/to/train2014/$line ];then
#    cp train2014/$line test_2017_mscoco
#    else if [ -f path/to/val2014/$line ];then
#      cp val2014/$line test_2017_mscoco
#    fi  
#  fi  
#done
