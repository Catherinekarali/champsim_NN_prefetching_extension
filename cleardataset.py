import csv

def cleardata(outfile):
    with open(str(outfile), 'a') as output:
        with open('out'+str(outfile), 'rt') as f:
            reader =csv.reader(f, delimiter=',', skipinitialspace=True)
            for line in reader:
                if (len(line)>=34):
                    for x in line:
                        if (x == line[0]):
                            output.write(x[1:] +','+str(int(x[1:])&int(511)) +','+str(int(x[1:])&int(255))+','+str(int(x[1:])&int(127))+ ",")
                        else:
                            if (x == line[-1]):
                                xn = int(x[:-1])
                                if (xn > 64 or xn < -64):
                                    output.write("0\n")
                                else:
                                    output.write(x[:-1]+"\n")
                            else:
                                xn = int(x)
                                if (xn>64 or xn<-64):
                                    output.write("0,")
                                else:
                                    output.write(x+",")