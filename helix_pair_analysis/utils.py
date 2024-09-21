import numpy as np
from matplotlib import pyplot as plt

from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def sanity_check_3d_plot(all_coords, window1, window2,
                         R1, t1, R2, t2, R, t):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(all_coords[:, 0], all_coords[:, 1], all_coords[:, 2],
               color='b', marker='.')
    ax.scatter(window1[:, 0], window1[:, 1], window1[:, 2],
               color='r', marker='o')
    ax.scatter(window2[:, 0], window2[:, 1], window2[:, 2],
               color='g', marker='s')
    w_trans = window1 @ R + t
    ax.scatter(w_trans[:, 0], w_trans[:, 1], w_trans[:, 2],
               color='m', marker='o')

    dist, angle, piston1, piston2, gearbox1, gearbox2 = \
        effective_coords(R, t @ R.T)

    # plot helical axes
    piston1, piston2 = line_line_nearest_points(t1, R1[2], t2, R2[2])
    axis1 = np.vstack([t1 + R1[2] * x for x in
                       np.linspace(-piston1, piston1, 50)])
    axis2 = np.vstack([t2 + R2[2] * x for x in
                       np.linspace(-piston2, piston2, 50)])
    ax.plot(axis1[:, 0], axis1[:, 1], axis1[:, 2])
    ax.plot(axis2[:, 0], axis2[:, 1], axis2[:, 2])
    # plot line of closest approach
    right_pt_1 = t1 + R1[2] * piston1
    right_pt_2 = t2 + R2[2] * piston2
    v = np.vstack([f * right_pt_1 + (1 - f) * right_pt_2
                   for f in np.linspace(0, 1, 51)])
    u = right_pt_2 - right_pt_1
    u /= np.linalg.norm(u)
    ax.plot(v[:, 0], v[:, 1], v[:, 2])

    ax.set_xlim([-60, 60])
    ax.set_ylim([-60, 60])
    ax.set_zlim([-60, 60])
    plt.show()


def line_line_nearest_points(p1, u1, p2=None, u2=None):
    """Find the points at which two lines are closest.

    Parameters
    ----------
    p1 : np.array [3] or [3 x 1]
        A point through which the first line passes
    u1 : np.array [3] or [3 x 1]
        A unit vector along the first line.
    p2 : np.array [3] or [3 x 1], optional
        A point through which the second line passes. Default is [0, 0, 0].
    u2 : np.array [3] or [3 x 1], optional
        A unit vector along the second line. Default is [0, 0, 1].

    Returns
    -------
    t1 : float
        Coefficient such that the nearest point on line 1 to line 2 is
        p1 + t1 * u1.
    t2 : float
        Coefficient such that the nearest point on line 1 to line 2 is
        p2 + t2 * u2.
    """
    if p2 is None:
        p2 = np.array([0, 0, 0])
    if u2 is None:
        u2 = np.array([0, 0, 1])
    delta_p = (p2 - p1).reshape((3, 1))
    U = np.vstack([u1.flatten(), -u2.flatten()]).T
    t, _, _, _ = np.linalg.lstsq(U, delta_p, rcond=None)
    t1, t2 = t.flatten()
    return t1, t2


def effective_coords(R1, t1, R2, t2, half_helix_length):
    """Calculate the effective coordinates of a helix pair with frames
       (R1, t1) and (R2, t2).

    Parameters
    ----------
    R1 : np.array [3 x 3]
        Rotation matrix for the frame of Helix 1.
    t : np.array [3]
        Translation vector for the frame of Helix 1.
    R2 : np.array [3 x 3]
        Rotation matrix for the frame of Helix 1.
    t2 : np.array [3]
        Translation vector for the frame of Helix 1.
    half_helix_length : float
        Half-length of each helix (in Angstroms).

    Returns
    -------
    eff_coords : np.array [6]
        Array of effective coordinates for the helix pair.  These coordinates
        are as follows:
            1. Distance between the helices at closest approach as if they
               were infinitely long (in Angstroms).
            2. Angle between the helices at closest approach (in radians).
            3. Piston displacement of the first helix along its z-axis
               (in Angstroms).
            4. Piston displacement of the second helix along its z-axis
               (in Angstroms).
            5. Gearbox rotation of the first helix about its z-axis
               (in radians).
            6. Gearbox rotation of the second helix about its z-axis
               (in radians).
    """
    # calculate the unit vector in the direction of the displacement between
    # the helical axes at closest approach
    u = np.cross(R1[2], R2[2])
    u /= np.linalg.norm(u)

    # compute the minimum distance between the line segments that run along
    # each helix segment
    denom = 1. - np.dot(R1[2], R2[2]) ** 2
    lambda_1 = (np.dot(R1[2], t2 - t1) +
                np.dot(R1[2], R2[2]) * np.dot(R2[2], t1 - t2)) / denom
    lambda_2 = (np.dot(R2[2], t1 - t2) +
                np.dot(R1[2], R2[2]) * np.dot(R1[2], t2 - t1)) / denom
    factor_1 = np.min([half_helix_length, np.abs(lambda_1)]) / \
               np.abs(lambda_1)
    factor_2 = np.min([half_helix_length, np.abs(lambda_2)]) / \
               np.abs(lambda_2)
    displacement = t2 + lambda_2 * factor_2 * R2[2] - \
                   t1 - lambda_1 * factor_1 * R1[2]
    dist = np.linalg.norm(displacement)

    # calculate the scissor degree of freedom, defined as the amount
    # by which the maximum distance between the two line segments
    # exceeds the minimum distance
    endpoints = np.array([t1 + half_helix_length * R1[2],
                          t1 - half_helix_length * R1[2],
                          t2 + half_helix_length * R2[2],
                          t2 - half_helix_length * R2[2]])
    max_dist = cdist(endpoints[:2], endpoints[2:]).max()
    scissor = max_dist - dist

    # calculate the piston degree of freedom, defined as the difference
    # of the distance between each helix frame centroid and the point of
    # nearest approach of the helix axes
    piston = lambda_2 - lambda_1

    # calculate the gearbox angles of each helix about its z-axis.
    # since the inter-helix displacement vector lies in either the xy-plane
    # or the plane formed by R[:, 0] and R[:, 1], compute the signed angle
    # of the displacement in each
    displacement_proj_1 = displacement - np.dot(displacement, R1[2]) * R1[2]
    displacement_proj_2 = displacement - np.dot(displacement, R2[2]) * R2[2]
    gearbox1 = -np.arctan2(np.dot(displacement_proj_1, R1[1]),
                           np.dot(displacement_proj_1, R1[0]))
    gearbox2 = -np.arctan2(np.dot(-displacement_proj_2, R2[1]),
                           np.dot(-displacement_proj_2, R2[0]))
    # without the minus sign, the arctan2 gives us the angle through which
    # we'd need to rotate to get to the displacement vector from the local
    # x-axis; we want to go the other way around, hence the minus sign

    return dist, scissor, piston, gearbox1, gearbox2


def effective_coords_old(R, t):
    """Calculate the effective coordinates of a helix pair with frames
       that differ by the transformation (R, t).

    Parameters
    ----------
    R : np.array [3 x 3]
        Rotation matrix for the frame transformation.
    t : np.array [3]
        Translation vector for the frame transformation.

    Returns
    -------
    eff_coords : np.array [6]
        Array of effective coordinates for the helix pair.  These coordinates
        are as follows:
            1. Distance between the helices at closest approach as if they
               were infinitely long (in Angstroms).
            2. Angle between the helices at closest approach (in radians).
            3. Piston displacement of the first helix along its z-axis
               (in Angstroms).
            4. Piston displacement of the second helix along its z-axis
               (in Angstroms).
            5. Gearbox rotation of the first helix about its z-axis
               (in radians).
            6. Gearbox rotation of the second helix about its z-axis
               (in radians).
    """
    # calculate the unit vector in the direction of the displacement between
    # the helical axes at closest approach
    u = np.cross([0, 0, 1], R[2])
    u /= np.linalg.norm(u)

    # compute the distance between the lines along the z-axis and the
    # R[2] axis at closest approach
    dist = np.abs(np.dot(t, u))

    # calculate the angle between the helical axes at closest approach
    # by dotting the z-axis from the matrix R with the global z-axis
    # (i.e. R[2, 2])
    angle = np.arccos(R[2, 2])

    # calculate the piston degrees of freedom, defined as the distance
    # between each helix frame and the point of nearest approach
    # between their axes
    piston1, piston2 = line_line_nearest_points(t, R[2])

    # calculate the gearbox angles of each helix about its z-axis
    # since u lies in either the xy-plane or the plane formed by
    # R[:, 0] and R[:, 1], compute the signed angle of u in each
    gearbox1 = np.arctan2(u[1], u[0])
    gearbox2 = np.arctan2(np.dot(-u, R[:, 1]), np.dot(-u, R[:, 0]))

    return dist, angle, piston1, piston2, gearbox1, gearbox2


def effective_coords_old(R, t):
    """Calculate the effective coordinates of a helix pair with R and t.

    Parameters
    ----------
    R : np.array [3 x 3]
        Rotation matrix to transform one helix into the other.
    t : np.array [3]
        Translation matrix to transform one helix into the other.

    Returns
    -------
    eff_coords : np.array [6]
        Array of effective coordinates for the helix pair.  These coordinates
        are as follows:
            1. Distance between the helices at closest approach as if they
               were infinitely long (in Angstroms).
            2. Angle between the helices at closest approach (in radians).
            3. Piston displacement of the first helix along its z-axis
               (in Angstroms).
            4. Piston displacement of the second helix along its z-axis
               (in Angstroms).
            5. Gearbox rotation of the first helix about its z-axis
               (in radians).
            6. Gearbox rotation of the second helix about its z-axis
               (in radians).
    """
    # calculate the unit vector in the direction of the displacement between
    # the helical axes at closest approach
    u = np.cross([0, 0, 1], R[2])
    u /= np.linalg.norm(u)

    # compute the distance between the lines along the z-axis and the
    # R[2] axis at closest approach
    dist = np.abs(np.dot(t, u))

    # calculate the angle between the helical axes at closest approach
    # by dotting the z-axis from the matrix R with the global z-axis
    # (i.e. R[2, 2])
    angle = np.arccos(R[2, 2])

    # calculate the piston degrees of freedom, defined as the distance
    # between each helix frame and the point of nearest approach
    # between their axes
    piston1, piston2 = line_line_nearest_points(t, R[2])
    # print(dist, np.linalg.norm(np.array([0, 0, piston1]) -
    #                            t + piston2 * R[2]))

    # calculate the gearbox angles of each helix about its z-axis
    # since u lies in either the xy-plane or the plane formed by
    # R[:, 0] and R[:, 1], compute the signed angle of u in each
    gearbox1 = np.arctan2(u[1], u[0])
    gearbox2 = np.arctan2(np.dot(-u, R[:, 1]), np.dot(-u, R[:, 0]))

    return dist, angle, piston1, piston2, gearbox1, gearbox2


def cylindrical(t):
    """Determine the cylindrical coordinates from a translation vector.

    Parameters
    ----------
    t : np.ndarray [3]
        Translation vector.

    Returns
    -------
    r : float
        Radial coordinate.
    alpha : float
        Angular coordinate.
    z : float
        Axial coordinate.
    """
    r = np.linalg.norm(t[:2])
    alpha = np.arctan2(t[1], t[0])
    z = t[2]
    return r, alpha, z


def euler(R):
    """Determine the extrinsic xzx Euler angles from a rotation matrix.
    
    Parameters
    ----------
    R : np.ndarray [3 x 3]
        Rotation matrix.

    Returns
    -------
    phi : float
        First Euler angle.
    theta : float
        Second Euler angle.
    psi : float
        Third Euler angle.
    """
    phi = np.arctan2(R[2, 0], R[2, 1])
    theta = np.arccos(R[2, 2])
    psi = -np.arctan2(R[0, 2], R[1, 2])
    return phi, theta, psi


def wrapped_diff(a, b, wrap=np.pi):
    """Calculate the wrapped difference between two angles.

    Parameters
    ----------
    a : float
        First angle in radians.
    b : float
        Second angle in radians.
    wrap : float, optional
        The value at which the angle wraps around to the opposite of itself.
        Default is pi.

    Returns
    -------
    diff : float
        The wrapped difference between a and b.
    """
    stack = np.vstack([a - b, a + 2 * wrap - b, a - b - 2 * wrap])
    amin = np.argmin(np.abs(stack), axis=0)
    return stack[amin, np.arange(len(amin))]


def ideal_helix(n_residues, rise=1.5, twist=100, radius=2.3, start=0):
    """Generate coordinates for an ideal helix.

    Parameters
    ----------
    n_residues : int
        Number of residues in the helix.
    rise : float, optional
        Rise per residue of the helix in Angstroms. Default is 1.5.
    twist : float, optional
        Twist per residue of the helix in degrees. Default is 100.
    radius : float, optional
        Radius of the helix in Angstroms. Default is 2.3.
    start : int, optional
        Which residue to start the helix on. The residue with index
        0 will always lie along the x-axis.  Default is 0.

    Returns
    -------
    coords : np.array [n_points x 3]
        Array of coordinates for the ideal helix.
    """
    t = np.linspace(start * twist * np.pi / 180,
                    (n_residues + start) * twist * np.pi / 180,
                    n_residues, endpoint=False)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = np.linspace(0, n_residues * rise, n_residues,
                    endpoint=False)
    return np.array([x, y, z]).T


def kabsch(X, Y):
    """Rotate and translate X into Y to minimize the SSD between the two,
       and find the derivatives of the SSD with respect to the entries of Y.

       Implements the SVD method by Kabsch et al. (Acta Crystallogr. 1976,
       A32, 922) and the SVD differentiation method by Papadopoulo and
       Lourakis (INRIA Sophia Antipolis. 2000, research report no. 3961).

    Parameters
    ----------
    X : np.array [N x 3]
        Array of mobile coordinates to be transformed by a proper rotation
        to minimize sum squared displacement (SSD) from Y.
    Y : np.array [N x 3]
        Array of stationary coordinates against which to transform X.

    Returns
    -------
    R : np.array [3 x 3]
        Proper rotation matrix required to transform X such that its SSD
        with Y is minimized.
    t : np.array [3]
        Translation matrix required to transform X such that its SSD with Y
        is minimized.
    ssd : float
        Sum squared displacement after alignment.
    d_ssd_dY : np.array [N x 3]
        Matrix of derivatives of the SSD with respect to each element of Y.
    """
    n = len(X)
    # compute R using the Kabsch algorithm
    Xbar, Ybar = np.mean(X, axis=0), np.mean(Y, axis=0)
    Xc, Yc = X - Xbar, Y - Ybar
    H = np.dot(Xc.T, Yc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.dot(U, Vt)))
    D = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., d]])
    R = np.dot(U, np.dot(D, Vt))
    t = Ybar - np.dot(Xbar, R)
    # compute SSD from aligned coordinates XR
    XRmY = np.dot(Xc, R) - Yc
    ssd = np.sum(XRmY ** 2)
    return R, t, ssd