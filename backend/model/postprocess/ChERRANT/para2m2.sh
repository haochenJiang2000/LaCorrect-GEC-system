DATA_FILE=../../data/MuCGEC2.0/MuCGEC2.0/paper
python parallel_to_m2.py -f $DATA_FILE"/train/train_notag.para" -o $DATA_FILE"/train/train.m2.char" -g char
python parallel_to_m2.py -f $DATA_FILE"/train/train_notag.para" -o $DATA_FILE"/train/train.m2.word" -g word
python parallel_to_m2.py -f $DATA_FILE"/valid/valid_notag.para" -o $DATA_FILE"/valid/valid.m2.char" -g char
python parallel_to_m2.py -f $DATA_FILE"/valid/valid_notag.para" -o $DATA_FILE"/valid/valid.m2.word" -g word
python parallel_to_m2.py -f $DATA_FILE"/test/test_notag.para" -o $DATA_FILE"/test/test.m2.char" -g char
python parallel_to_m2.py -f $DATA_FILE"/test/test_notag.para" -o $DATA_FILE"/test/test.m2.word" -g word


DATA_FILE=../../data/MuCGEC2.0/MuCGEC2.0/bingju
python parallel_to_m2.py -f $DATA_FILE"/train/train_notag.para" -o $DATA_FILE"/train/train.m2.char" -g char
python parallel_to_m2.py -f $DATA_FILE"/train/train_notag.para" -o $DATA_FILE"/train/train.m2.word" -g word
python parallel_to_m2.py -f $DATA_FILE"/valid/valid_notag.para" -o $DATA_FILE"/valid/valid.m2.char" -g char
python parallel_to_m2.py -f $DATA_FILE"/valid/valid_notag.para" -o $DATA_FILE"/valid/valid.m2.word" -g word
python parallel_to_m2.py -f $DATA_FILE"/test/test_notag.para" -o $DATA_FILE"/test/test.m2.char" -g char
python parallel_to_m2.py -f $DATA_FILE"/test/test_notag.para" -o $DATA_FILE"/test/test.m2.word" -g word


#DATA_FILE=../../data/MuCGEC2.0/MuCGEC2.0/weixin
#python parallel_to_m2.py -f $DATA_FILE"/train/train_notag.para" -o $DATA_FILE"/train/train.m2.char" -g char
#python parallel_to_m2.py -f $DATA_FILE"/train/train_notag.para" -o $DATA_FILE"/train/train.m2.word" -g word
#python parallel_to_m2.py -f $DATA_FILE"/valid/valid_notag.para" -o $DATA_FILE"/valid/valid.m2.char" -g char
#python parallel_to_m2.py -f $DATA_FILE"/valid/valid_notag.para" -o $DATA_FILE"/valid/valid.m2.word" -g word
#python parallel_to_m2.py -f $DATA_FILE"/test/test_notag.para" -o $DATA_FILE"/test/test.m2.char" -g char
#python parallel_to_m2.py -f $DATA_FILE"/test/test_notag.para" -o $DATA_FILE"/test/test.m2.word" -g word

#python parallel_to_m2.py -f total_notag.para -o total.m2.word -g word
#python parallel_to_m2.py -f total_notag.para -o total.m2.char -g char