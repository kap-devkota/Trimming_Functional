def annotate(infile, adict, outfile, ignore_missing_annotations = False):
    edge_list = []
    with open(infile, "r") as ifp:
        for line in ifp:
            e1, e2, wt = line.lstrip().rstrip().split()[: 3]
            wt = float(wt)

            if ignore_missing_annotations == True:
                if e1 not in adict or e2 not in adict:
                    continue

            e1 = adict[e1]
            e2 = adict[e2]

            edge_list.append((e1, e2, wt))

    str_o = ""
    with open(outfile, "w") as ofp:
        for edge in edge_list:
            e1, e2, wt = edge
            str_ = "{} {} {}\n".format(e1, e2, wt)
            str_o += str_

        str_o = str_o.lstrip().rstrip()

        ofp.write(str_o)

    return 
