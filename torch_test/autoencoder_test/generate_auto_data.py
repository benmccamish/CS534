import csv
import sys
import string
import numpy as np
import cv2







def write_frames(video_data, data_dir, data_file):
	
	



def read_data(file_name, video_data):
	
	with open(file_name, 'r') as csvfile:
		my_file = csv.reader(csvfile, delimiter='\t')
		for row in my_file:
			video_data.append([row[0], int(row[1])])
		


def main(argc, argv):
	video_data = []
	read_data(argv[1], video_data)

	write_frames(video_data, argv[2], argv[3])

if __name__ == "__main__":
	main(len(sys.argv), sys.argv)

