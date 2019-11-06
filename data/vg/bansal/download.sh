#! /bin/sh
wget http://ankan.umiacs.io/files/vg_test_list.txt
wget http://ankan.umiacs.io/files/vg_seen_classes.json
wget http://ankan.umiacs.io/files/vg_unseen_classes.json
wget http://ankan.umiacs.io/files/vg_synset_word_dict.json
wget https://obj.umiacs.umd.edu/zsd_files/vg_train_list.json.tar.gz
tar -xvf vg_train_list.json.tar.gz
rm vg_train_list.json.tar.gz
