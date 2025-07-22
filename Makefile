VIDFILES = dance.mp4
CHECK_FILES = dance.mp4
LABEL = dance
allclean:
	-rm *.npy
	-rm data/*
	-rm *.h5
collect:
	python collect.py $(VIDFILES) $(LABEL)
	python train.py 
	python recognize_gesture $(CHECK_FILES)
collect2d:
	python collect.py $(VIDFILES) $(LABEL)
	python train2d.py
	python recog2d.py $(CHECK_FILES)
