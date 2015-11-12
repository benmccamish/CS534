

import csv
import sys
import string



def main(argc, argv):
	if (argc != 5):
		print("arguments: bad_root_folder good_root_folder data_file new_file")
		return 1
		
	bad_root_folder = argv[1]
	good_root_folder = argv[2]
	data_file = argv[3]
	new_file = argv[4]
	
	bad_names = []
	starting_frames = []
	with open(data_file, "r") as csv_file:
		data_reader = csv.reader(csv_file, delimiter='\t')
		for row in data_reader:
			if (row[2] != 'k'):
				bad_names.append(row[0])
				starting_frames.append(int(row[3]))
	
	good_names = []
	for name in bad_names:
		g = good_root_folder + string.lstrip(name, bad_root_folder)
		good_names.append(g)
	
	with open(new_file, 'w') as csv_file:
		data_writer = csv.writer(csv_file, delimiter='\t')
		
		for i in range(len(good_names)):
			data_writer.writerow([good_names[i], starting_frames[i]])
	
	print("Finished creating: %s" % (new_file))



if __name__ == "__main__":
	main(len(sys.argv), sys.argv)
