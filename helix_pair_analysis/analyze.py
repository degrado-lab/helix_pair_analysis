import sys
import argparse

import numpy as np
import prody as pr

from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

from utils import *

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Analyze helix pair.')
    parser.add_argument('pdb1', type=str,
                        help='First PDB file containing the helix pair.')
    parser.add_argument('selstr11', type=str,
                        help=('ProDy selection string for the first helix '
                              'in the first PDB file.'))
    parser.add_argument('selstr12', type=str,
                        help=('ProDy selection string for the second helix '
                              'in the first PDB file.'))
    parser.add_argument('pdb2', type=str,
                        help='First PDB file containing the helix pair.')
    parser.add_argument('selstr21', type=str,
                        help=('ProDy selection string for the first helix '
                              'in the second PDB file.'))
    parser.add_argument('selstr22', type=str,
                        help=('ProDy selection string for the second helix '
                              'in the second PDB file.'))
    parser.add_argument('--window', type=int, default=7,
                        help=('Number of residues in the window for the '
                              'analysis. Must be odd, default is 7.'))
    args = parser.parse_args()
    return args


def main(args):
    """Analyze the change in relative coordinates in a helix pair in two PDBs.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    """
    assert args.window % 2, 'Window length must be odd.'
    CA_sel = ' and name CA'
    struct1 = pr.parsePDB(args.pdb1)
    helix1_1 = struct1.select(args.selstr11 + CA_sel).getCoords()
    helix1_2 = struct1.select(args.selstr12 + CA_sel).getCoords()
    struct2 = pr.parsePDB(args.pdb2)
    helix2_1 = struct2.select(args.selstr21 + CA_sel).getCoords()
    helix2_2 = struct2.select(args.selstr22 + CA_sel).getCoords()
    resnums = struct1.select(args.selstr11 + CA_sel).getResnums()
    resnums = resnums[args.window // 2:-args.window // 2 + 1]

    assert len(helix1_1) == len(helix1_2) == len(helix2_1) == len(helix2_2), \
        'Helices must have the same number of residues.'

    eff_coords_1 = []
    eff_coords_2 = []
    for i in range(len(helix1_1) - args.window + 1):
        # extract coordinates for the window
        helix1_1_window = helix1_1[i:i+args.window]
        helix1_2_window = helix1_2[i:i+args.window]
        helix2_1_window = helix2_1[i:i+args.window]
        helix2_2_window = helix2_2[i:i+args.window]

        # find closest residues in each of the helix pairs
        dists_1 = cdist(helix1_1_window, helix1_2_window)
        idx11, idx12 = np.unravel_index(np.argmin(dists_1), dists_1.shape)
        dists_2 = cdist(helix2_1_window, helix2_2_window)
        idx21, idx22 = np.unravel_index(np.argmin(dists_2), dists_2.shape)

        idx11, idx12, idx21, idx22 = args.window // 2, \
                                     args.window // 2, \
                                     args.window // 2, \
                                     args.window // 2
        helix_half_length = 1.5 * args.window // 2

        # find rotation and translation matrices for each window
        R11, t11, sse11, _ = kabsch(ideal_helix(args.window, start=-idx11),
                                    helix1_1_window)
        ideal_11 = ideal_helix(args.window, start=-idx11) @ R11 + t11
        R12, t12, sse12, _ = kabsch(ideal_helix(args.window, start=-idx12),
                                    helix1_2_window)
        ideal_12 = ideal_helix(args.window, start=-idx12) @ R12 + t12
        R21, t21, sse21, _ = kabsch(ideal_helix(args.window, start=-idx21),
                                    helix2_1_window)
        ideal_21 = ideal_helix(args.window, start=-idx21) @ R21 + t21
        R22, t22, sse22, _ = kabsch(ideal_helix(args.window, start=-idx22),
                                    helix2_2_window)
        ideal_22 = ideal_helix(args.window, start=-idx22) @ R22 + t22

        '''
        print(np.sqrt(sse11 / args.window),
              np.sqrt(sse12 / args.window),
              np.sqrt(sse21 / args.window),
              np.sqrt(sse22 / args.window))
        '''

        # find relative transforms between the two helices
        # R1 = R11.T @ R12
        # t1 = t12 - t11 @ R1
        # R2 = R21.T @ R22
        # t2 = t22 - t21 @ R2

        '''
        sanity_check_3d_plot(struct1.select('name CA').getCoords(),
                             ideal_11,
                             ideal_12,
                             R11, t11,
                             R12, t12,
                             R1, t1)
        sanity_check_3d_plot(struct2.select('name CA').getCoords(),
                             ideal_21,
                             ideal_22,
                             R21, t21,
                             R22, t22,
                             R2, t2)
        '''

        # calculate the effective coordinates for each transform
        eff_coords_1.append(effective_coords(R11, t11, R12, t12,
                                             helix_half_length))
        eff_coords_2.append(effective_coords(R21, t21, R22, t22,
                                             helix_half_length))

    eff_coords_1 = np.array(eff_coords_1).T
    eff_coords_2 = np.array(eff_coords_2).T

    # dist_diff, angle_diff, piston1_diff, piston2_diff, \
    #     gearbox1_diff, gearbox2_diff = eff_coords_2
    # dist_diff, angle_diff, piston1_diff, piston2_diff, \
    #     gearbox1_diff, gearbox2_diff = eff_coords_1
    # '''
    dist_diff = eff_coords_1[0] - eff_coords_2[0]
    angle_diff = eff_coords_1[1] - eff_coords_2[1]
    piston_diff = eff_coords_1[2] - eff_coords_2[2]
    gearbox1_diff = 2.3 * wrapped_diff(eff_coords_1[3], eff_coords_2[3])
    gearbox2_diff = 2.3 * wrapped_diff(eff_coords_1[4], eff_coords_2[4])
    # '''

    plt.figure()
    plt.plot(resnums, dist_diff, label='Distance difference (Angstroms)')
    plt.plot(resnums, angle_diff, label='Scissor difference (Angstroms)')
    plt.plot(resnums, piston_diff, label='Piston difference (Angstroms)')
    plt.plot(resnums, gearbox1_diff, label='Gearbox 1 difference (Angstroms)')
    plt.plot(resnums, gearbox2_diff, label='Gearbox 2 difference (Angstroms)')
    plt.legend(fontsize='large')
    plt.xlabel('Residue window', fontsize='x-large')
    plt.ylabel('Coordinate difference', fontsize='x-large')
    plt.xticks(resnums, fontsize='large')
    plt.yticks(fontsize='large')
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)
