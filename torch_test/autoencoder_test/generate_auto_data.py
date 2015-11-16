import csv
import sys
import string
import numpy as np
import cv2







def write_frames(video_data, output_dir):
	for i in range(len(video_data)):
		my_vid = VideoCapture(video_data[0])
		my_vid.set(cv2.cv.CAP_PROP_POS_FRAMES, video_data[1])
		
		for j in range(len(120)):
			stuff, im = my_vid.read()
			output_name = output_dir + str(i) + "_" + str(j) + ".png"
			cv2.imwrite(output_name, im)
		
		my_vid.release()
		
	



def read_data(file_name, video_data):
	
	with open(file_name, 'r') as csvfile:
		my_file = csv.reader(csvfile, delimiter='\t')
		for row in my_file:
			video_data.append([row[0], int(row[1])])
		


def main(argc, argv):
	video_data = []
	read_data(argv[1], video_data)

	write_frames(video_data, argv[2])

if __name__ == "__main__":
	main(len(sys.argv), sys.argv)

