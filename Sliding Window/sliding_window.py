import csv
import math

win_size = 5                                                        # Specify the size of the sliding window

f = open('sample_data.csv', 'r')                                    # name of the file to be processed
re = csv.reader(f, quoting=csv.QUOTE_NONE)                          
fo = open('sample_out.csv', 'w')                                    # name of the output file
wr = csv.writer(fo, quoting=csv.QUOTE_NONE, lineterminator='\n')

window = []
labels = []
ctr = 0
for row in re:
    feature_num = len(row) - 1                                      # reads how many cells per row excluding the labels column
    if(ctr == 0):                                                   # repeats the column header
        for i in range(0, win_size):
            window = window + row[:-1]
        wr.writerow(window)
        window = []
        ctr+=1
    elif(ctr != win_size+1):                                        # initializes the window with the first five residues
        window = window + row[:-1]
        labels.append(row[-1])
        ctr+=1
    else:                                                           # sliding window part
        labels.append(row[-1])
        window.append(labels[math.ceil(win_size/2)-1])              # append the label of the center residue to the window
        labels = labels[1:]
        wr.writerow(window)
        window = window[:-1]                                        
        window = window[feature_num:len(window)]                    # remove the data of the first residue in the window
        window = window + row[:-1]                                  # add the data of the next residue to the window

window.append(labels[math.ceil(win_size/2)-1])
wr.writerow(window)
f.close()                                                           # close
fo.close()
