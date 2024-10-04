import sys
import argparse

import numpy as np
import prody as pr

from itertools import permutations
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

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
    assert len(helix1_1) == len(helix1_2) == len(helix2_1) == len(helix2_2), \
        'Helices must have the same number of residues.'
    # ensure helix1_1 and helix1_2 are symmetry mates about the z-axis
    R, t, ssd = kabsch(helix1_2, helix1_1)
    if np.sqrt(ssd / helix1_1.shape[0]) > 0.1:
        print('Warning: helices from the first PDB are not symmetry mates.')
    eigvals, dR = np.linalg.eigh(R)
    for p in permutations(range(3)):
        idxs = np.array(p)
        if np.allclose(eigvals[idxs], np.array([-1, -1, 1])) and \
                np.linalg.det(dR[:, idxs]) > 0:
            dR = dR[:, idxs]
            break
    dt = -0.5 * t @ dR
    helix1_1 = helix1_1 @ dR + dt
    helix1_2 = helix1_2 @ dR + dt
    helix2_1 = helix2_1 @ dR + dt
    helix2_2 = helix2_2 @ dR + dt
    resnums = struct1.select(args.selstr11 + CA_sel).getResnums()
    resnums = resnums[args.window // 2:-args.window // 2 + 1]

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

        # find rotation and translation matrices for each window
        ideal = ideal_helix(args.window, start=-args.window // 2)
        ideal -= ideal.mean(axis=0)
        R11, t11, sse11 = kabsch(ideal, helix1_1_window)
        R12, t12, sse12 = kabsch(ideal, helix1_2_window)
        R21, t21, sse21 = kabsch(ideal, helix2_1_window)
        R22, t22, sse22 = kabsch(ideal, helix2_2_window)

        # adjust coordinate system so t12 - t11 = [|t12 - t11|, 0, 0]
        dt = t12 - t11
        cos_theta = dt[0] / np.linalg.norm(dt)
        sin_theta = dt[1] / np.linalg.norm(dt)
        dR = np.array([[cos_theta, -sin_theta, 0],
                       [sin_theta, cos_theta, 0],
                       [0, 0, 1]])
        R11 = R11 @ dR
        R12 = R12 @ dR
        R21 = R21 @ dR
        R22 = R22 @ dR
        t11 = t11 @ dR
        t12 = t12 @ dR
        t21 = t21 @ dR
        t22 = t22 @ dR

        # convert the translation vectors to cylindical coordinates
        r11, alpha11, z11 = cylindrical(t11)
        r12, alpha12, z12 = cylindrical(t12)
        r21, alpha21, z21 = cylindrical(t21)
        r22, alpha22, z22 = cylindrical(t22)

        # compute relative cylindrical coordinates
        d_r_1 = r21 - r11
        d_alpha_1 = wrapped_diff(alpha21, alpha11)[0]
        d_z_1 = z21 - z11

        d_r_2 = r22 - r12
        d_alpha_2 = wrapped_diff(alpha22, alpha12)[0]
        d_z_2 = z22 - z12

        # compute Euler angles for each rotation matrix
        phi11, theta11, psi11 = euler(R11)
        phi12, theta12, psi12 = euler(R12)
        phi21, theta21, psi21 = euler(R21)
        phi22, theta22, psi22 = euler(R22)

        # compute relative Euler angles
        d_phi_1 = wrapped_diff(phi21, phi11)[0]
        d_theta_1 = theta21 - theta11
        d_psi_1 = wrapped_diff(psi21, psi11)[0]
        d_phi_2 = wrapped_diff(phi22, phi12)[0]
        d_theta_2 = theta22 - theta12
        d_psi_2 = wrapped_diff(psi22, psi12)[0]
        # d_phi_1, d_theta_1, d_psi_1 = euler(R21 @ R11.T)
        # d_phi_2, d_theta_2, d_psi_2 = euler(R22 @ R12.T)

        # append the relative coordinates to the list
        eff_coords_1.append([d_r_1, d_alpha_1, d_z_1, 
                             d_phi_1, d_theta_1, d_psi_1])
        eff_coords_2.append([d_r_2, d_alpha_2, d_z_2, 
                             d_phi_2, d_theta_2, d_psi_2])

        # calculate the effective coordinates for each transform
        # helix_half_length = 1.5 * args.window // 2
        # eff_coords_1.append(effective_coords(R11, t11, R12, t12,
        #                                      helix_half_length))
        # eff_coords_2.append(effective_coords(R21, t21, R22, t22,
        #                                      helix_half_length))

    eff_coords_1 = np.array(eff_coords_1).T
    eff_coords_2 = np.array(eff_coords_2).T

    for i, eff_coords in enumerate([eff_coords_1, eff_coords_2]):
        plt.figure()
        labels = ['Distance difference (Angstroms)', 
                  'Cylindrical angle difference (Radians)',
                  'Axial difference (Angstroms)',
                  'Phi difference (Radians)',
                  'Theta difference (Radians)',
                  'Psi difference (Radians)']
        for j in range(6):
            plt.plot(resnums, eff_coords[j], label=labels[j])
        plt.title('Effective coordinates for helix {}'.format(i + 1),
                  fontsize='x-large')
        plt.legend(fontsize='large')
        plt.xlabel('Residue window', fontsize='x-large')
        plt.ylabel('Coordinate difference', fontsize='x-large')
        plt.xticks(resnums, fontsize='large')
        plt.yticks(fontsize='large')
        plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)
