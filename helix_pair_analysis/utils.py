import numpy as np
import numba as nb
from matplotlib import pyplot as plt

from scipy.optimize import minimize

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


def effective_coords(R1, t1, R2, t2):
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

    # compute the distance between the lines along the z-axis and the 
    # R[2] axis at closest approach
    dist = np.abs(np.dot(t2 - t1, u))

    # calculate the angle between the helical axes at closest approach
    # by dotting the z-axis from the matrix R with the global z-axis 
    # (i.e. R[2, 2])
    angle = np.arccos(np.dot(R1[2], R2[2]))

    # calculate the piston degrees of freedom, defined as the distance 
    # between each helix frame and the point of nearest approach 
    # between their axes
    piston1, piston2 = line_line_nearest_points(t1, R1[2], t2, R2[2])
    piston1, piston2 = np.mean([piston1, piston2]), piston1 - piston2

    # calculate the gearbox angles of each helix about its z-axis
    # since u lies in either the xy-plane or the plane formed by 
    # R[:, 0] and R[:, 1], compute the signed angle of u in each
    gearbox1 = np.arctan2(np.dot(-u, R1[1]), np.dot(-u, R1[0]))
    gearbox2 = np.arctan2(np.dot(-u, R2[1]), np.dot(-u, R2[0]))

    return dist, angle, piston1, piston2, gearbox1, gearbox2


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
    # compute derivative of R with respect to Y
    omega_U, omega_Vt = populate_omegas(U, S, Vt)
    dUdH = np.einsum('km,ijml->ijkl', U, omega_U)
    dVtdH = -np.einsum('ijkm,ml->ijkl', omega_Vt, Vt)
    dRdH = np.einsum('imkl,mj->ijkl', dUdH, np.dot(D, Vt)) + \
           np.einsum('im,mjkl->ijkl', np.dot(U, D), dVtdH)
    dRdY = np.einsum('km,ijml->ijkl', Xc, dRdH)
    XdRdY = np.einsum('im,mjkl->ijkl', Xc, dRdY)
    d_ssd_dY = 2. * (np.sum(XRmY * XdRdY, axis=(0, 1)) - XRmY) 
    return R, t, ssd, d_ssd_dY


@nb.jit(nopython=True, cache=True)
def populate_omegas(U, S, Vt):
    """Populate omega_U and omega_Vt matrices from U, S, and Vt.

    Parameters
    ----------
    U : np.array [3 x 3]
        Left unitary matrix from a singular value decomposition.
    S : np.array [3]
        Vector of singular values from a singular value decomposition.
    Vt : np.array [3 x 3]
        Right unitary matrix from a singular value decomposition.

    Returns
    -------
    omega_U : np.array [3 x 3 x 3 x 3]
        omega_U matrix of matrices as described in Papadopolou and 
        Lourakis (2000).
    omega_Vt : np.array [3 x 3 x 3 x 3]
        omega_V matrix of matrices as described in Papadopolou and 
        Lourakis (2000), but with the last two dimensions transposed.
    """
    omega_U = np.zeros((3, 3, 3, 3))
    omega_Vt = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k, l in [(0, 1), (1, 2), (2, 0)]:
                system_A = np.array([[S[l], S[k]], 
                                     [S[k], S[l]]])
                system_b = np.array([[U[i, k] * Vt[l, j]], 
                                     [-U[i, l] * Vt[k, j]]])
                if S[k] != S[l]:
                    soln = np.linalg.solve(system_A, system_b)
                else: # solve via least squares in the degenerate case
                    soln, _, _, _ = np.linalg.lstsq(system_A, system_b, 1e-14)
                omega_U[i, j, k, l], omega_Vt[i, j, l, k] = soln.flatten()
                omega_U[i, j, l, k] = -omega_U[i, j, k, l]
                omega_Vt[i, j, k, l] = -omega_Vt[i, j, l, k]
    return omega_U, omega_Vt