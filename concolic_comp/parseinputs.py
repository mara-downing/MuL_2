#replace this code with code to read inputs from file, normalize, and sort into lists

def parse_inputs(inputsfile):
    infile = open(inputsfile, "r")
    data = infile.readlines()
    infile.close()
    inputlist = []
    outputlist = []
    for line in data:
        if line[-1] == "\n":
            line = line[:-1]
        if line == "":
            continue
        linelist = line.split(",")
        singleinput = linelist[:-1]
        singleinput = [float(x) for x in singleinput]
        inputlist += [singleinput]
        outputlist += [int(linelist[-1])]
    return(inputlist, outputlist)