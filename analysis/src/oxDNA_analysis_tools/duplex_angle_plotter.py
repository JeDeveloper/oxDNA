#!/usr/bin/env python3

from typing import List, Tuple
import numpy as np
from sys import exit
from os import path
import argparse
from oxDNA_analysis_tools.UTILS.logger import log, logger_settings
from oxDNA_analysis_tools.UTILS.constants import FIG_DPI

def rad2degree(angle:float) -> float:
    """
    Convert radians to degrees

    Parameters:
        angle (float): The angle in radians to convert.

    Returns:
        (float): The angle converted to degrees.
    """
    return (angle * 180 / np.pi)

def angle_between (axis1:np.ndarray, axis2:np.ndarray) -> float:
    """
    Find the angle between two vectors.

    Parameters:
        axis1 (numpy.array): The first vector.
        axis2 (numpy.array): The second vector.
    
    Returns:
        (float): The angle between the vectors in radians.
    """
    return (np.arccos(np.dot(axis1, axis2)/(np.linalg.norm(axis1)*np.linalg.norm(axis2))))

def get_angle_between(files:List[str], p1s:List[List[int]], p2s:List[List[int]], invert_mask:List[bool]) -> Tuple[List[List[np.ndarray]], List[List[float]], List[List[float]], List[List[float]], List[List[float]]]:
    """
        Read in a duplex list file and return the angles between specified duplexes.

        Parameters:
            files (List[str]): The list of duplex files to read.
            p1s (List[List[int]]): Find duplexes containing these nucleotides...
            p2s (List[List[int]]): ...and get the angles between them and duplexes containing these nucleotides.
            invert_mask (List[bool]): Invert one of the vectors in the i-th duplex pair?

        Returns:
            Tuple[List[List[ndarray]], List[List[float]], List[List[float]], List[List[float]], List[List[float]]]: 
            | The list of angles between the specified duplexes.
            | The mean angle between each pair of duplexes.
            | The median angle between each pair of duplexes.
            | The standard deviation of the angle between each pair of duplexes.
            | The percentage of confs that have the duplexes.
    """
    if(len(files) != len(p1s) != len(p2s)):
        raise RuntimeError("Bad input arguments!\nThe number of files must equal the number of lists of particle pairs.\n\
                           For example, files=['duplexes1.txt', 'duplexes2.txt'], p1s=[[23, 42], [7, 13, 45]], p2s=[[35, 99], [35, 74, 68]]")
    
    all_angles = [[] for _ in files]
    means = [[] for _ in files]
    medians = [[] for _ in files]
    stdevs = [[] for _ in files]
    representations = [[] for _ in files]

    global_count = 0

    # For each angle file...
    for i, (anglefile, search1, search2) in enumerate(zip(files, p1s, p2s)):
        if len(search1) != len(search2):
            raise RuntimeError("Bad input arguments!\nThe lengths of the particle pair lists are unequal.\n\
                               For example, files=['duplexes1.txt', 'duplexes2.txt'], p1s=[[23, 42], [7, 13, 45]], p2s=[[35, 99], [35, 74, 68]]")

        steps = 0 # counts the number of configurations in the file
        last_step = 0
        count = 0 # counts the number of search terms found in each step
        all_angles[i] = [[] for _ in p1s[i]]
        found = False

        #the format of the angle file is as follows: (tbh this should be a JSON)
        # 0: time
        # 1: duplex id
        # 2: strand 1 start nucleotide id
        # 3: strand 1 end nucleotide id
        # 4: strand 2 start nucleotide id
        # 5: strand 2 end nucleotide id
        # 6: X-component of the axis vector
        # 7: Y-component of the axis vector
        # 8: Z-component of the axis vector
        # 9: Helix position

        # Search the angle file for the search ids...
        with open(anglefile) as file:
            all_search = search1.copy()
            all_search.extend(search2)
            all_search = list(set(all_search)) # get rid of duplicate searches
            d = {i : np.array([0, 0, 0]) for i in all_search}
            for l in file.readlines()[1:]: #the first line is a header, so it can be dropped
                try:
                    l = l.split("\t")
                    t = float(l[0])
                except Exception as e:
                    print("ERROR: The following line is incorrectly formatted:")
                    print(l)
                    print("The error was:\n",e)
                    print("skiping the line")
                    continue

                #dump values and reset if we're in a new time (but also skip the first pass)
                if (t != last_step):
                    for j, (p1, p2) in enumerate(zip(search1, search2)):
                        if np.linalg.norm(d[p1]) != 0 and np.linalg.norm(d[p2]) != 0:
                            if invert_mask[j]:
                                d[p1] *= -1
                            angle = rad2degree(angle_between(d[p1], d[p2]))
                            all_angles[i][j].append(angle)
                        else:
                            all_angles[i][j].append(np.nan)

                    found = False
                    steps += 1
                    d = dict.fromkeys(d, np.array([0, 0, 0]))
                    count = 0

                #don't need to do anything if both angles were already found for this timestep
                if found:
                    continue

                #look for the nucleotide IDs.
                for s in all_search:
                    l1 = int(l[2])
                    u1 = int(l[3])
                    l2 = int(l[4])
                    u2 = int(l[5])
                    si = int(s)
                    if (si >= l1 and si <= u1) or (si >= l2 and si <= u2): 
                        d[s] = np.array([float(l[6]), float(l[7]), float(l[8])])
                        count += 1

                #once all are found, can skip the rest of this timestep
                if count == len(d):
                    found = True

                last_step = t

        #catch last configuration
        for j, (p1, p2) in enumerate(zip(search1, search2)):
            if np.linalg.norm(d[p1]) != 0 and np.linalg.norm(d[p2]) != 0:
                if invert_mask[j]:
                    d[p1] *= -1
                angle = rad2degree(angle_between(d[p1], d[p2]))
                all_angles[i][j].append(angle)
            else:
                all_angles[i][j].append(np.nan)
        steps += 1

        #compute some statistics
        all_angles[i] = [np.array(a) for a in all_angles[i]]
        mean = [np.nanmean(a) for a in all_angles[i]]
        median = [np.nanmedian(a) for a in all_angles[i]]
        stdev = [np.nanstd(a) for a in all_angles[i]]
        representation = [np.count_nonzero(~np.isnan(a))/steps for a in all_angles[i]]

        #add to the output data
        means[i] = mean
        medians[i] = median
        stdevs[i] = stdev
        representations[i] = representation

        global_count += len(search1)

    return (all_angles, means, medians, stdevs, representations)

def make_plots(all_angles:List[List[np.ndarray]], names:List[str], outfile:str, hist:bool, line:bool):
    """
        Generate histogram or trajectory plots

        Parameters:
            all_angles (List[List[np.array]]) : Axis 0 -> simulation, Axis 1 -> duplex pair, Axis 2 -> angles
            names (List[str]) : Names for each data series
            outfile (str) : Basename for the output.  Will have hist or traj placed between the name and the first "."
            hist (bool) : Make a histogram?
            line (bool) : Make a trajectory lineplot?
    
    """
    #make a histogram
    import matplotlib.pyplot as plt
    if hist == True:
        if line == True:
            out = outfile[:outfile.find(".")]+"_hist"+outfile[outfile.find("."):]
        else:
            out = outfile
    
        bins = np.linspace(0, 180, 60)
    
        artists = []
        for i,traj_set in enumerate(all_angles):
            for alist in traj_set:
                a = plt.hist(alist, bins, weights=np.ones(len(alist)) / len(alist),  alpha=0.3, label=names[i], histtype=u'stepfilled', edgecolor='k')
                artists.append(a)
        plt.legend(labels=names)
        plt.xlim((0, 180))
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Normalized frequency")
        log("Saving histogram to {}".format(out))
        plt.tight_layout()
        plt.savefig(out, dpi=FIG_DPI)

    #make a trajectory plot
    if line == True:
        if hist == True:
            plt.clf()
            out = outfile[:outfile.find(".")]+"_traj"+outfile[outfile.find("."):]
        else:
            out = outfile
        
        artists = []
        for i,traj_set in enumerate(all_angles):
            for alist in traj_set:
                a = plt.plot(alist)
                artists.append(a)
        plt.legend(labels=names)
        plt.xlabel("Configuration Number")
        plt.ylabel("Angle (degrees)")
        log("Saving line plot to {}".format(out))
        plt.tight_layout()
        plt.savefig(out, dpi=FIG_DPI)
    
    return

def cli_parser(prog="duplex_angle_plotter.py"):
    #Get command line arguments.
    parser = argparse.ArgumentParser(prog = prog, description="Finds the ensemble of angles between pairs duplexes defined by nucleotides in the duplexes.")
    parser.add_argument('-i', '--input', metavar='angle_file', dest='input', nargs='+', action='append', help='An angle file from duplex_angle_finder.py and a list of particle id pairs found in the duplexes you want to compare.  Can call -i multiple times to plot multiple datasets together.')
    parser.add_argument('-v', '--invert_mask', dest='invert_mask', nargs='+', help='If 1 invert the i-th vector, if 0, do nothing.')
    parser.add_argument('-o', '--output', metavar='output_file', help='The name to save the graph file to')
    parser.add_argument('-f', '--format', metavar='<histogram/trajectory/both>', help='Output format for the graphs.  Defaults to histogram.  Options are \"histogram\", \"trajectory\", and \"both\"')
    parser.add_argument('-d', '--data', metavar='data_file', help='If set, the output for the graphs will be dropped as a json to this filename for loading in oxView or your own scripts')
    parser.add_argument('-n', '--names', metavar='names', nargs='+', help='Names of the data series.  Will default to particle ids if not provided')
    parser.add_argument('-q', '--quiet', metavar='quiet', dest='quiet', action='store_const', const=True, default=False, help="Don't print 'INFO' messages to stderr")
    return(parser)

def main():
    parser = cli_parser(path.basename(__file__))
    args = parser.parse_args()

    logger_settings.set_quiet(args.quiet)
    from oxDNA_analysis_tools.config import check
    check(["python", "numpy", "matplotlib"])

    try:
        files = [i[0] for i in args.input]
        p1s = [i[1::2] for i in args.input]
        p2s = [i[2::2] for i in args.input]
    except Exception as e:
        print("ERROR: Failed to read files")
        print(e)
        parser.print_help()
        exit(1)

    n_angles = sum(len(p) for p in p1s)

    if args.invert_mask:
        invert_mask = args.invert_mask
    else:
        invert_mask = np.zeros(n_angles)

    invert_mask = [bool(int(m)) for m in invert_mask]

    #Make sure that the input is correctly formatted
    if(len(files) != len(p1s) != len(p2s)):
        raise RuntimeError("Bad input arguments\nThe number of particle pairs must be a multiple of 2.\
                           For example, oat duplex_angle_plotter -i duplexes1.txt 23 35 42 99 -i duplexes2.txt 7 35 13 74 45 68")
    if len(invert_mask) != n_angles:
        raise RuntimeError("Bad input arguments\nThe length of the invert mask must be equal to the number of angles to calculate.\
                           For example, oat duplex_angle_plotter -i duplexes1.txt 23 35 42 99 -v 1 0")

    #-o names the output file
    if args.output:
        outfile = args.output
    else:
        log("No outfile name provided, defaulting to \"angle.png\"")
        outfile = "angle.png"

    #-f defines which type of graph to produce
    hist = False
    line = False
    if args.format:
        if "histogram" in args.format:
            hist = True
        if "trajectory" in args.format:
            line = True
        if "both" in args.format:
            hist = line = True
        if hist == line == False:
            raise RuntimeError("Unrecognized graph format\nAccepted formats are \"histogram\", \"trajectory\", and \"both\"")
    else:
        log("No graph format specified, defaulting to histogram")
        hist = True

    # actual computation
    all_angles, means, medians, stdevs, representations = get_angle_between(files, p1s, p2s, invert_mask)

    # -n sets the names of the data series
    if args.names:
        names = args.names
        if len(names) < n_angles:
            log("Names list too short.  There are {} items in names and {} angles were calculated.  Will pad with particle IDs".format(len(names), n_angles), level="warning")
            for i in range(len(names), n_angles):
                names.append("{}-{}".format([j for sl in p1s for j in sl][i], [j for sl in p2s for j in sl][i]))
        if len(names) > n_angles:
            log("Names list too long. There are {} items in names and {} angles were calculated.  Truncating to be the same as distances".format(len(names), n_angles), level="warning")
            names = names[:n_angles]
    else:
        log("Defaulting to particle IDs as data series names")
        names = ["{}-{}".format(p1, p2) for p1, p2 in zip([i for sl in p1s for i in sl], [i for sl in p2s for i in sl])]

    # -d will dump the distances as json files for loading with the trajectories in oxView
    if args.data:
        from json import dump
        if len(files) > 1:
            f_names = [path.basename(f) for f in files]
            log("Angle lists from separate trajectories are printed to separate files for oxView compatibility.  Trajectory names will be appended to your provided data file name.")
            file_names = ["{}_{}.json".format(args.data.strip('.json'), i) for i,_ in enumerate(f_names)]
        else:
            file_names = [args.data.strip('.json')+'.json']
        names_by_traj = [['{}-{}'.format(p1, p2) for p1, p2 in zip(p1l, p2l)] for p1l, p2l in zip(p1s, p2s)]

        for file_name, ns, ang_list in zip(file_names, names_by_traj, all_angles):
            obj = {}
            for n, a in zip(ns, ang_list):
                obj[n] = list(a)
            with open(file_name, 'w+') as f:
                log("Writing data to {}.  This can be opened in oxView using the Order parameter selector".format(file_name))
                dump(obj, f)

    #print statistical information
    print("name:\t", end='')
    [print("{}\t".format(t), end='') for t in names[:n_angles]]
    print("")

    print("mean:\t", end='')
    [print("{:.2f}\t".format(m), end='') for m in [i for sl in means for i in sl]]
    print("")

    print("stdevs:\t", end='')
    [print("{:.2f}\t".format(s), end='') for s in [i for sl in stdevs for i in sl]]
    print("")

    print("median:\t", end='')
    [print("{:.2f}\t".format(m), end='') for m in [i for sl in medians for i in sl]]
    print("")

    print("freqs:\t", end='')
    [print("{:.2f}\t".format(r), end='') for r in [i for sl in representations for i in sl]]
    print("")

    # Make plots
    make_plots(all_angles, names, outfile, hist, line)

if __name__ == '__main__':
    main()
    