# helix_pair_analysis

##Purpose

Analyze the relative motions of a pair of alpha helices in two protein structures.

## Installation

Clone the Github repository and use `pip` to install the ProDy, NumPy, 
SciPy, and Matplotlib:

```bash
pip install prody numpy scipy matplotlib
```

## Use

Use the `analyze.py` script from the command line to visualize the 
differences in the five key degrees of freedom (interhelical distance at 
closest approach, interhelical angle, relative pistoning, and the gearbox 
angles of each helix) between two-helix bundles from two PDB files.

Example:
```bash
python helix_pair_analysis/helix_pair_analysis/analyze.py 
second_structure.pdb 'chain A and resnum 21:50' 'chain B and resnum 21:50' first_structure.pdb 'chain A and resnum 21:50' 'chain B and resnum 21:50'
```

Note that the strings passed as input after each PDB file path are ProDy 
selection strings for the helical windows over which to compute the 
differences in the interhelical degrees of freedom between the two 
structures. These selection strings should correspond to the same 
number of residues each time.

The default helical window length is 7, but the user can modify this from 
the command line with the `--window` command. 
