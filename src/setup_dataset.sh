# make dir
mkdir data
mkdir data/test
mkdir data/train
mkdir raw_data

# download data
wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz -O ./raw_data/food-101.tar.gz

# unzip
tar zxvf ./raw_data/food-101.tar.gz -C ./raw_data

# cp data and split train and test
cp raw_data/food-101/images/* data/train/ -r
cp raw_data/food-101/images/* data/test/ -r
cat raw_data/food-101/meta/test.txt | xargs -I @ rm data/train/@.jpg
cat raw_data/food-101/meta/train.txt | xargs -I @ rm data/test/@.jpg

