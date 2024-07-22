import numpy as np
import copy
import scipy
import math
from sklearn.neighbors import NearestNeighbors
import torch

def Vector(x, y, z):
        return (x, y, z)

def subtract(u, v):
    "Return difference between two vectors."
    x = u[0] - v[0]
    y = u[1] - v[1]
    z = u[2] - v[2]
    return Vector(x, y, z)

def length(v):
    "Return length of a vector."
    sum = 0.0
    for c in v:
        sum += c * c
    return math.sqrt(sum)

def cross(u, v):
    "Return the cross product of two vectors."
    x = u[1] * v[2] - u[2] * v[1]
    y = u[2] * v[0] - u[0] * v[2]
    z = u[0] * v[1] - u[1] * v[0]
    return Vector(x, y, z)

def angle(v0, v1):
    "Return angle [0..pi] between two vectors."
    cosa = dot(v0, v1) / length(v0) / length(v1)
    cosa = round(cosa, 10)
    if cosa > 1:
        cosa = 1
    if cosa < -1:
        cosa = -1
    return math.acos(cosa)

def angle_rowwise(A, B):
    "Return angle [0..pi] between vectors row-wise in input matrices"
    p1 = np.einsum('ij,ij->i',A,B)
    p2 = np.linalg.norm(A,axis=1)
    p3 = np.linalg.norm(B,axis=1)
    p4 = p1 / (p2*p3)
    return np.arccos(np.clip(p4,-1.0,1.0))

def dot(u, v):
    "Return dot product of two vectors."
    sum = 0.0
    for cu, cv in zip(u, v):
        sum += cu * cv
    return sum

def calc_dihedral(p0, p1, p2, p3):
    """Return angle [0..2*pi] formed by vertices p0-p1-p2-p3."""

    v01 = subtract(p0, p1)
    v32 = subtract(p3, p2)
    v12 = subtract(p1, p2)
    v0 = cross(v12, v01)
    v3 = cross(v12, v32)
    # The cross product vectors are both normal to the axis
    # vector v12, so the angle between them is the dihedral
    # angle that we are looking for.  However, since "angle"
    # only returns values between 0 and pi, we need to make
    # sure we get the right sign relative to the rotation axis
    a = angle(v0, v3)
    if dot(cross(v0, v3), v12) > 0:
        a = -a
    return a

def calc_local_angle(p0, p1, p2, p3):
    """Return angles of p0-p1 bond and p3-p1 bond in local reference frame of p1-p2 bond."""
    # define vectors and angle to use in updating pos
    v01 = subtract(p0, p1)
    v12 = subtract(p1, p2)
    v0 = cross(v12, v01)
    t1 = cross(v12, v0)
    t2 = cross(v12, t1)

    # define translation and rotation matrices
    v12_norm = np.divide(v12, length(v12))
    t1_norm = np.divide(t1, length(t1))
    t2_norm = np.divide(t2, length(t2))
    matrix_transform = np.transpose(np.array([v12_norm, t1_norm, t2_norm]))
    matrix_transform_inv = np.linalg.inv(matrix_transform)

    # transform to local reference frame and update position, then transform back
    p3_local = np.matmul(matrix_transform_inv, p3)
    p1_local = np.matmul(matrix_transform_inv, p1)
    p0_local = np.matmul(matrix_transform_inv, p0)
    # print('local frame: ', p3_local, p1_local)
    center_local = p1_local
    p3_local_centered = p3_local - center_local
    p0_local_centered = p0_local - center_local
    return np.arctan2(p3_local_centered[2], p3_local_centered[1]), np.arctan2(p0_local_centered[2], p0_local_centered[1])


def extract_rotation_info(R):
    """Extract rotation angle and axis from rotation matrix"""
    trace_R = np.trace(R)
    cos_angle = 0.5*(trace_R - 1)
    sin_angle = 0.5*np.sqrt((3 - trace_R)*(1 + trace_R))
    angle_from_cos = np.arccos(cos_angle)
    angle_from_sin = np.arcsin(sin_angle)
    # print("angles: ", angle_from_cos, angle_from_sin)
    if trace_R == 3:
        return None, None
    elif trace_R == -1:
        e2 = R[0,1] / (np.sqrt((1 + R[0,0]) * (1 + R[1, 1])))
        e3 = R[0,2] / (np.sqrt((1 + R[0,0]) * (1 + R[2, 2])))
        axis = [np.sqrt(0.5*(1 + R[0, 0])), e2*np.sqrt(0.5*(1 + R[1, 1])), e3*np.sqrt(0.5*(1 + R[2, 2]))]
    else:
        axis = np.multiply((1 / np.sqrt((3 - trace_R) * (1 + trace_R))), [R[2, 1] - R[1, 2], R[0,2] - R[2,0], R[1, 0] - R[0, 1]])
    return angle_from_cos, axis

def update_pos_dist_ang(p0, p1, p2, angle_update, bond_length, bond_length_update):
    """Return vertex p0 consistent with old p0, current p1, p2, p3, dihedral angle, and bond length."""
    
    # define vectors and angle to use in updating pos
    v01 = subtract(p0, p1)
    v12 = subtract(p1, p2)
    v0 = cross(v12, v01)
    t1 = cross(v12, v0)
    t2 = cross(v12, t1)
    a1 = angle(v01, v12)
    

    # define translation and rotation matrices
    v12_norm = np.divide(v12, length(v12))
    t1_norm = np.divide(t1, length(t1))
    t2_norm = np.divide(t2, length(t2))
    matrix_transform = np.transpose(np.array([v12_norm, t1_norm, t2_norm]))
    matrix_transform_inv = np.linalg.inv(matrix_transform)

    # transform to local reference frame and update position, then transform back
    p0_local = np.matmul(matrix_transform_inv, p0)
    p1_local = np.matmul(matrix_transform_inv, p1)
    center_offset = np.cos(a1)*bond_length
    center_local = p1_local + [center_offset, 0, 0]
    p0_local_centered = p0_local - center_local
    p0_local_centered[:2] = p0_local_centered[:2] + [-1*np.cos(a1)*bond_length_update, np.sin(a1)*bond_length_update]
    p1_local_centered = p1_local - center_local
    p0_local_centered[0] -= p1_local_centered[0]
    p0l = length(p0_local_centered)
    p0_local_centered /= p0l
    new_ang = a1 + angle_update
    if new_ang > np.pi:
        new_ang = a1 - angle_update
    p0_local_centered[0] = np.cos(new_ang)
    p0_local_centered[1] = np.sin(new_ang)
    p0_local_centered *= p0l
    p0_local_centered[0] += p1_local_centered[0]
    p0_local_updated = p0_local_centered + center_local
    p0_updated = np.matmul(matrix_transform, p0_local_updated)
    return p0_updated

def update_pos(p0, p1, p2, dihedral, bond_length, bond_length_update):
    """Return vertex p0 consistent with old p0, current p1, p2, p3, dihedral angle, and bond length."""
    
    # define vectors and angle to use in updating pos
    v01 = subtract(p0, p1)
    v12 = subtract(p1, p2)
    v0 = cross(v12, v01)
    t1 = cross(v12, v0)
    t2 = cross(v12, t1)
    a1 = angle(v01, v12)

    # define translation and rotation matrices
    v12_norm = np.divide(v12, length(v12))
    t1_norm = np.divide(t1, length(t1))
    t2_norm = np.divide(t2, length(t2))
    matrix_transform = np.transpose(np.array([v12_norm, t1_norm, t2_norm]))
    matrix_transform_inv = np.linalg.inv(matrix_transform)
    R = [[np.cos(dihedral), -1*np.sin(dihedral)], [np.sin(dihedral), np.cos(dihedral)]]

    # transform to local reference frame and update position, then transform back
    p0_local = np.matmul(matrix_transform_inv, p0)
    p1_local = np.matmul(matrix_transform_inv, p1)
    center_offset = np.cos(a1)*bond_length
    center_local = p1_local + [center_offset, 0, 0]
    p0_local_centered = p0_local - center_local
    p0_local_centered[:2] = p0_local_centered[:2] + [-1*np.cos(a1)*bond_length_update, np.sin(a1)*bond_length_update]
    p0_local_centered[1:] = np.matmul(R, p0_local_centered[1:])
    p0_local_updated = p0_local_centered + center_local
    p0_updated = np.matmul(matrix_transform, p0_local_updated)
    return p0_updated


def matrix_update_pos(p0, p1, p2, dihedral, verbose=False):
    """Return matrix of vertices consistent with current dihedral update."""
    p0_translated = p0 - p1
    v = p2 - p1
    v = np.divide(v, length(v))
    
    cos_d = np.cos(dihedral)
    sin_d = np.sin(dihedral)
    # R = np.array([[cos_d + v[0]**2*(1 - cos_d), v[0]*v[1]*(1-cos_d)-v[2]*sin_d, v[0]*v[2]*(1-cos_d)+v[1]*sin_d], 
    #      [v[0]*v[1]*(1-cos_d)+v[2]*sin_d, cos_d + v[1]**2*(1 - cos_d), v[1]*v[2]*(1-cos_d)-v[0]*sin_d],
    #      [v[0]*v[2]*(1-cos_d)-v[1]*sin_d, v[1]*v[2]*(1-cos_d)+v[0]*sin_d, cos_d + v[2]**2*(1 - cos_d)]
    # ])
    u_x = np.array([[0, -1*v[2], v[1]], [v[2], 0, -1*v[0]], [-1*v[1], v[0],  0]])
    R = np.identity(3)*cos_d + sin_d*u_x + (1 - cos_d)*np.tensordot(v, v, axes=0)
    p0_translated_rotated = p0_translated @ R.T
    p0_rotated = p0_translated_rotated + p1
    translation = p1 - p1 @ R.T
    if verbose:
        print("R", R)
        print("cos_d", cos_d)
        print("sin_d", sin_d)
        # print("p0_translated", p0_translated)
        print("v", p2 - p1)
        print("p0_rotated", p0_rotated)
        print("p0 shape: ", p0_translated.shape)
    return p0_rotated, R, translation


def calc_dihedrals_bond_lengths(X=None,chain_lens=None, expected_bond_lengths=None, expected_dihedrals=None):
    """Calculate noise

    Args
    ----
    X : torch.tensor
        Position of backbone residues
        size: num_residues x 4 x 3
    chain_lens : list
        list of chain lengths

    Returns
    -------
    dihedrals : pairs of (phi, psi) for each residue
    bond_lengths : tuples of bond length distances for each residue
    """

    X = X.reshape((X.shape[0]*X.shape[1], X.shape[2]))
    start_idx = 0
    dihedrals = []
    bond_lengths = []
    for c in chain_lens:
        c *= 4
        cur_dihedrals = [np.nan, np.nan, np.nan, np.nan]
        cur_bond_lengths = [np.nan, np.nan, np.nan, np.nan]
        for atom_id in range(start_idx, start_idx + c - 4):
            p0 = X[atom_id,:]
            p1 = X[atom_id+1,:]
            p2 = X[atom_id+2,:]
            p3 = X[atom_id+3,:]
            if atom_id % 4 == 0:
                p3 = X[atom_id+4,:]
            elif atom_id % 4 == 1:
                p2 = X[atom_id+3,:]
                p3 = X[atom_id+4,:]
            elif atom_id % 4 == 2:
                p1 = X[atom_id+2,:]
                p2 = X[atom_id+3,:]
                p3 = X[atom_id+4,:]
            if atom_id % 4 == 3:
                p0 = X[atom_id,:]
                p1 = X[atom_id-1,:]
                p2 = X[atom_id+1,:]
                p3 = X[atom_id+2,:]
            dihedral = calc_dihedral(p0, p1, p2, p3)
            bond_length = length(subtract(p0,p1))
            if atom_id % 4 == 0:
                cur_dihedrals[1] = dihedral
                cur_bond_lengths[0] = bond_length
            elif atom_id % 4 == 1:
                cur_bond_lengths[1] = bond_length
                cur_dihedrals[2] = dihedral
            elif atom_id % 4 == 2:
                cur_dihedrals[0] = dihedral
                cur_bond_lengths[2] = bond_length
            else:
                cur_dihedrals[3] = dihedral
                cur_bond_lengths[3] = bond_length
            if not np.isnan(cur_dihedrals[0]) and not np.isnan(cur_dihedrals[1]) and not np.isnan(cur_dihedrals[2]) and not np.isnan(cur_dihedrals[3]):
                dihedrals.append(cur_dihedrals)
                cur_dihedrals = [np.nan, np.nan, np.nan, np.nan]
            if not np.isnan(cur_bond_lengths[0]) and not np.isnan(cur_bond_lengths[1]) and not np.isnan(cur_bond_lengths[2]) and not np.isnan(cur_bond_lengths[3]):
                bond_lengths.append(cur_bond_lengths)
                cur_bond_lengths = [np.nan, np.nan, np.nan, np.nan]
        start_idx += c
    dihedrals = np.swapaxes(np.array(dihedrals), 0, 1)
    bond_lengths = np.swapaxes(np.array(bond_lengths), 0, 1)
    # if expected_bond_lengths is not None and expected_dihedrals is not None:
    #     for atom_idx in range(bond_lengths.shape[1]):
    #         print(f"expected dihedrals: {expected_dihedrals[:, atom_idx]}")
    #         print(f"dihedrals: {dihedrals[:, atom_idx]}")
    #         print(f"dihedrals diff: {dihedrals[:, atom_idx] - expected_dihedrals[:, atom_idx]}")
    return dihedrals.round(5), bond_lengths.round(5)

def get_angle(p0, p1, p2):
    v01 = subtract(p0, p1)
    v12 = subtract(p1, p2)
    a1 = angle(v01, v12)
    return a1

def generate_noise(flex_type, noise_level, pdb, replicate, epoch, X, mask=None, bond_length_noise_level=0.0, bond_angle_noise_level=0.0, chain_lens=None, noise_lim=2, dtype='torch', constant=-1):
    """Calculate noise

    Args
    ----
    replicate : int
        Replicate run number for setting seed
    epoch : int
        Epoch number for setting seed
    flex_type : str
        methodology to calculate flex data
    noise_level : float
        std of noise to add
    size : tuple
        shape of coordinate entry in TERM data
    X : torch.tensor
        Position of backbone residues
        size: num_residues x 4 x 3
    bond_length_noise_level : float
        std of noise to add to bond lengths
    chain_lens : list
        list of chain lengths

    Returns
    -------
    noise : noise to be added to backbone atoms
    """
    if dtype == 'torch':
        expanded_mask = mask.repeat_interleave(4)
        dev = X.device
        if X.is_cuda:
            X = X.cpu()
            expanded_mask = expanded_mask.cpu()
        X = X.numpy()
        expanded_mask = expanded_mask.numpy()

    size = X.shape
    # print('epoch: ', epoch+1)
    # print('replicate: ', replicate)
    seed = (epoch + 1 + 100*replicate) * sum([ord(char) for char in pdb])
    # print('seed: ', seed)
    np.random.seed(seed)
    flex_type_copy = copy.deepcopy(flex_type)
    dihedral_updates = None
    if flex_type.find("dist_ang") > -1:
        X = X.reshape((X.shape[0]*X.shape[1], X.shape[2]))
        da_noise = np.zeros(X.shape)
        start_idx = 0
        n_chains = len(chain_lens)
        for i_chain, c in enumerate(chain_lens):
            c *= 4
            noise_level_multiplier = 1
            base_noise_level_multiplier = 1
            atom_id = start_idx
            while atom_id < start_idx + c - 4:
                if dtype == 'torch' and (not expanded_mask[atom_id] or not expanded_mask[atom_id + 4]):
                    atom_id += 4
                    continue
                p0 = X[atom_id,:]
                p1_id = atom_id+1
                p2_id = atom_id+2
                p3_id = atom_id+4
                o_offset = 0
                if atom_id % 4 == 3:
                    atom_id += 1
                    continue
                elif atom_id % 4 == 2:
                    p1_id = atom_id+2
                    p2_id = atom_id+3
                    o_offset = 1
                elif atom_id % 4 == 1:
                    p2_id = atom_id+3
                p1 = X[p1_id,:] + da_noise[p1_id, :]
                p2 = X[p2_id,:] + da_noise[p2_id, :]
                p3 = X[p3_id,:] + da_noise[p3_id, :]
                bond_length = length(subtract(p0, p1))
                bond_length_update = np.random.normal(loc=0, scale=bond_length_noise_level, size=1)[0]
                bond_length_update = bond_length_update * bond_length
                bond_angle_update = np.random.normal(loc=0, scale=bond_angle_noise_level, size=1)[0]
                bond_angle = get_angle(p0, p1, p2)
                bond_angle_update = bond_angle * bond_angle_update
                new_pos = update_pos_dist_ang(p0, p1, p2, bond_angle_update, bond_length, bond_length_update)  
                da_noise[atom_id, :] = (new_pos - X[atom_id, :])
                atom_id += 1
            start_idx += c
        X += da_noise
        X = X.reshape((int(X.shape[0]/4), 4, 3))
        da_noise = da_noise.reshape(X.shape)
    if flex_type.find("fixed") == -1 and flex_type.find("torsion") == -1:
        noise = np.random.normal(loc=0, scale=noise_level, size=size)
    elif flex_type.find("fixed") > -1:
        flex_size = (size[0], size[2])
        noise = np.random.normal(loc=0, scale=noise_level, size=flex_size)
        noise = np.repeat(noise, 4)
        noise = np.reshape(noise, size)
    elif flex_type.find("torsion") > -1:
        if flex_type.find("batch") == -1:
            raise Exception(f"flex type {flex_type} must be ran with batch enabled")
        X = X.reshape((X.shape[0]*X.shape[1], X.shape[2]))
        noise = np.zeros(X.shape)
        prev_noise = np.zeros(X.shape)
        start_idx = 0
        atom_clash_check = -1
        dihedral_updates = np.zeros((X.shape[0], ))
        n_chains = len(chain_lens)
        fallback = np.zeros(n_chains)
        for i_chain, c in enumerate(chain_lens):
            c *= 4
            noise_level_multiplier = 1
            base_noise_level_multiplier = 1
            atom_id = start_idx
            all_noises = {}
            if flex_type.find("processive") > -1:
                all_rotation_matrices = {}
                all_translations = {}
            flex_type = flex_type_copy
            if flex_type.find("processive") > -1:
                rotation_matrix = np.identity(3)
                translation = np.zeros(X[0,:].shape)
            while atom_id < start_idx + c - 4:
                if dtype == 'torch' and (not expanded_mask[atom_id] or not expanded_mask[atom_id + 4]):
                    atom_id += 4
                    continue
                prev_noise = copy.deepcopy(noise)
                p0 = X[atom_id,:] + noise[atom_id, :]
                p1_id = atom_id+1
                p2_id = atom_id+2
                p3_id = atom_id+4
                o_offset = 0
                if atom_id % 4 == 1 or atom_id % 4 == 3:
                    atom_id += 1
                    continue
                elif atom_id % 4 == 2:
                    p1_id = atom_id+2
                    p2_id = atom_id+3
                    o_offset = 1
                p1 = X[p1_id,:] + noise[p1_id, :]
                p2 = X[p2_id,:] + noise[p2_id, :]
                p3 = X[p3_id,:] + noise[p3_id, :]
                frac_before = float((atom_id - start_idx) / c)
                if flex_type.find("neighbors") > -1:
                    nbrs = NearestNeighbors(n_neighbors=30, algorithm='ball_tree').fit(X)
                    orig_indices = nbrs.kneighbors([X[p3_id,:]], return_distance=False)
                    orig_indices = np.squeeze(orig_indices)[1:]
                angle_mean = 0
                axes_overlap = 0
                if flex_type.find("processive") > -1:
                    orig_pos_angle, new_pos_angle = calc_local_angle(p3, p1, p2, X[p3_id])
                    angle_difference = np.mod(orig_pos_angle - new_pos_angle, 2*np.pi)
                    if abs(2*np.pi - angle_difference < angle_difference):
                        angle_difference = -1*(2*np.pi - angle_difference)
                    new_X = X[start_idx:start_idx+c] + noise[start_idx:start_idx+c] - p1
                    new_X = X[atom_id:min(atom_id+30, start_idx+c)] + noise[atom_id:min(atom_id+30, start_idx+c)] - p1
                    X_chain = X[start_idx:start_idx+c] - p1
                    X_chain = X[atom_id:min(atom_id+30, start_idx+c)] - p1
                    R, d  = scipy.spatial.transform.Rotation.align_vectors(X_chain, new_X, weights = np.divide(1, X_chain.shape[0]*np.ones(X_chain.shape[0])))
                    align_rotation_angle, align_rotation_axis = extract_rotation_info(R.as_matrix())
                    if align_rotation_angle is not None:
                        axes_overlap = np.exp(-0.5*d) * np.dot((p2 - p1), align_rotation_axis) / np.linalg.norm(p2 - p1)
                        if abs(axes_overlap) > 1:
                            axes_overlap = 1 * np.sign(axes_overlap)
                        angle_mean = axes_overlap*align_rotation_angle

                dihedral_update = np.random.normal(loc=angle_mean, scale=noise_level*(1-abs(axes_overlap)), size=1)[0]
                if flex_type.find("max") > -1:
                    dihedral_update = np.sign(dihedral_update) * min(abs(dihedral_update), 0.2)
                
                if atom_id % 4 == 0 or atom_id % 4 == 2:
                    dihedral_update = noise_level_multiplier*dihedral_update
                if frac_before <= 0.5:
                    other_start_idx = start_idx
                    other_end_idx = atom_id + o_offset + 1
                if flex_type.find("processive") > -1 or frac_before > 0.5:
                    other_start_idx = p3_id - (1 - o_offset)
                    other_end_idx = start_idx + c
                dihedral_updates[atom_id] = dihedral_update
                bond_length = length(subtract(p0, p1))
                if flex_type.find("simple") > -1:
                    bond_length_update = np.random.normal(loc=0, scale=bond_length_noise_level, size=1)[0]
                    new_pos = update_pos(p0, p1, p2, -1*dihedral_update, bond_length, bond_length_update)  
                    noise[atom_id, :] = (new_pos - X[atom_id, :])
                    atom_id += 1
                    continue
                prev_noise = copy.deepcopy(noise)
                
                p0_matrix =  X[other_start_idx:other_end_idx, :] + noise[other_start_idx:other_end_idx, :]
                p0_matrix_new, new_rotation_matrix, new_translation = matrix_update_pos(p0_matrix, p1, p2, dihedral_update, verbose=False)
                if flex_type.find("processive") > -1:
                    base_rotation_matrix = copy.deepcopy(rotation_matrix)
                    base_translation = copy.deepcopy(translation)
                    if atom_id == start_idx:
                        rotation_matrix = copy.deepcopy(new_rotation_matrix)
                        translation = copy.deepcopy(new_translation)
                    else:
                        rotation_matrix = (rotation_matrix.T @ new_rotation_matrix.T).T
                        translation = (translation @ new_rotation_matrix.T) + new_translation                   
                
                noise[other_start_idx:other_end_idx, :] = p0_matrix_new - X[other_start_idx:other_end_idx, :]
                atom_step = 1

                if bond_length_noise_level > 0 and flex_type.find("processive") == -1:
                    bond_length_update = np.random.normal(loc=0, scale=noise_level_multiplier*bond_length_noise_level, size=1)[0]
                    new_p0 = X[atom_id,:] + noise[atom_id, :]
                    new_p1 = X[p1_id,:] + noise[p1_id, :]
                    new_p2 = X[p2_id,:] + noise[p2_id, :]
                    bond_length = length(new_p1 - new_p0)
                    new_pos = update_pos(new_p0, new_p1, new_p2, 0, bond_length, bond_length_update) 
                    noise[start_idx:atom_id + o_offset + 1] = np.add(noise[start_idx:atom_id + o_offset + 1], (new_pos - new_p0))

                if flex_type.find('rmsd') > -1:
                    new_X = X[start_idx:start_idx+c] + noise[start_idx:start_idx+c]
                    new_X_displacement = np.mean(new_X, axis=0)
                    new_X = new_X - new_X_displacement
                    X_displacement = np.mean(X[start_idx:start_idx+c], axis=0)
                    X_chain = X[start_idx:start_idx+c] - X_displacement
                    R, d  = scipy.spatial.transform.Rotation.align_vectors(X_chain, new_X, weights = np.divide(1, new_X.shape[0]*np.ones(new_X.shape[0])))
                    new_X_rotated = R.apply(new_X)
                    noise[start_idx:start_idx+c] = new_X_rotated - X_chain
                    if d > noise_lim:
                        atom_step = 0
                        noise_level_multiplier /= 1.5
                        noise = prev_noise
                elif flex_type.find("checkpoint") > -1:
                    if flex_type.find("fragments") > -1:
                        start_idx_fragment = max(start_idx, atom_id - 30)
                        start_idx_fragment = atom_id
                        end_idx_fragment = min(start_idx+c, p3_id + 30)
                    else:
                        start_idx_fragment = start_idx
                        end_idx_fragment = start_idx + c
                    new_X = X[start_idx_fragment:end_idx_fragment] + noise[start_idx_fragment:end_idx_fragment]
                    new_X_displacement = np.mean(new_X, axis=0)
                    new_X = new_X - new_X_displacement
                    X_displacement = np.mean(X[start_idx_fragment:end_idx_fragment], axis=0)
                    X_chain = X[start_idx_fragment:end_idx_fragment] - X_displacement
                    R, d  = scipy.spatial.transform.Rotation.align_vectors(X_chain, new_X, weights = np.divide(1, new_X.shape[0]*np.ones(new_X.shape[0])))
                    new_X_rotated = R.apply(new_X)
                    d = np.sqrt(np.mean(np.linalg.norm(X_chain - new_X, axis=-1)**2))
                    if d > noise_lim:
                        atom_step = 0
                        noise_level_multiplier /= 1.5
                        noise = prev_noise
                        if noise_level_multiplier < 0.01:
                            if flex_type.find("stepwise") > -1:
                                noises = []
                                steps = list(all_noises.keys())
                                steps.sort()
                                for step in steps:
                                    noise = all_noises[step].reshape(X.shape)
                                    noises.append(noise)
                                return noises, dihedral_updates, fallback
                            flex_type+='_rmsd'
                            flex_type=flex_type.replace("processive_", "")
                            noise[start_idx:start_idx+c] = np.zeros(noise[start_idx:start_idx+c].shape)
                            atom_step = start_idx - atom_id
                            fallback[i_chain] = 1
                elif flex_type.find("neighbors") > -1:
                    new_nbrs = NearestNeighbors(n_neighbors=30, algorithm='ball_tree').fit(X + noise)
                    new_indices = new_nbrs.kneighbors([X[p3_id,:] + noise[p3_id,:]], return_distance=False)
                    new_indices = np.squeeze(new_indices)[1:]
                    indices = np.unique(np.concatenate((orig_indices, new_indices), 0))
                    orig_neighbors = (X[indices, :]) - X[p3_id, :]
                    orig_neighbors = np.squeeze(orig_neighbors)
                    orig_distances = np.sqrt(np.sum(orig_neighbors**2, axis=-1))
                    neighbors_LJ_sigma = np.divide(orig_distances, 2**(1/6))
                    neighbors_LJ_A = np.multiply(4,neighbors_LJ_sigma**(12))
                    neighbors_LJ_B = np.multiply(4,neighbors_LJ_sigma**(6))
                    orig_neighbors_energy = -1*np.ones(neighbors_LJ_A.shape)
                    temperature = 300

                    new_neighbors = X[indices, :] + noise[indices, :] - (X[p3_id, :] + noise[p3_id, :])
                    new_distances = np.sqrt(np.sum(new_neighbors**2, axis=-1))
                    new_neighbors_energy = np.divide(neighbors_LJ_A, new_distances**(12)) - np.divide(neighbors_LJ_B, new_distances**(6))
                    accept_prob = np.exp(np.divide(30*(np.sum(orig_neighbors_energy) - np.sum(new_neighbors_energy)), 30*temperature))
                    if np.sum(np.isnan(X + noise)) > 0:
                        atom_step = 0
                        noise = prev_noise
                        noise_level_multiplier /= 1.5
                        if noise_level_multiplier < 0.01:
                            flex_type+='_rmsd'
                            flex_type=flex_type.replace("processive", "")
                            noise[start_idx:start_idx+c] = np.zeros(noise[start_idx:start_idx+c].shape)
                            atom_step = start_idx - atom_id
                            fallback[i_chain] = 1
                    if np.random.rand(1)[0] >= accept_prob:
                        atom_step = 0
                        noise_level_multiplier /= 1.5
                        noise = prev_noise
                        rotation_matrix = base_rotation_matrix
                        translation = base_translation
                        if atom_id == atom_clash_check:
                            base_noise_level_multiplier /= 1.5
                            atom_step = atom_step_backtrack
                            new_atom_id = atom_id + atom_step
                            noise_level_multiplier = base_noise_level_multiplier
                            noise = all_noises[new_atom_id]
                            rotation_matrix = all_rotation_matrices[new_atom_id]
                            translation = all_translations[new_atom_id]
                    else:
                        if atom_id == atom_clash_check:
                            noise_level_multiplier = 1
                            atom_clash_check = -1
                            base_noise_level_multiplier = 1
                        
                    if base_noise_level_multiplier < 0.01:
                        flex_type+='_rmsd'
                        flex_type=flex_type.replace("processive", "")
                        noise[start_idx:start_idx+c] = np.zeros(noise[start_idx:start_idx+c].shape)
                        atom_step = start_idx - atom_id
                        fallback[i_chain] = 1
                    if noise_level_multiplier / base_noise_level_multiplier < 0.1:
                        if atom_id == start_idx:
                            if noise_level_multiplier < 0.00001:
                                flex_type+='_rmsd'
                                flex_type=flex_type.replace("processive", "")
                                noise[start_idx:start_idx+c] = np.zeros(noise[start_idx:start_idx+c].shape)
                                atom_step = start_idx - atom_id
                                fallback[i_chain] = 1
                        else:
                            if atom_id != atom_clash_check:
                                n_iters = 1
                            else:
                                n_iters += 1
                            if flex_type.find("smart") > -1:
                                possible_rotation_axis_points = X[start_idx:atom_id] + noise[start_idx:atom_id]
                                possible_rotation_axis_points = np.delete(possible_rotation_axis_points, slice(3, None, 4), axis=0)
                                atom_id_dists = atom_id - possible_rotation_axis_points
                                atom_id_dists = np.delete(atom_id_dists, slice(1, None, 2), axis=0)
                                possible_rotation_vecs = np.diff(possible_rotation_axis_points, n=1, axis=0)
                                possible_rotation_vecs = np.delete(possible_rotation_axis_points, slice(1, None, 2), axis=0)
                                atom_id_rotation_vec_dists = np.divide(np.linalg.norm(np.cross(possible_rotation_vecs, atom_id_dists), axis=-1), np.linalg.norm(possible_rotation_vecs))
                                max_vec_dist_ind = np.argmax(atom_id_rotation_vec_dists)
                                new_atom_id = int(2*max_vec_dist_ind)
                            else:    
                                if len(dihedral_updates[max(start_idx, atom_id - 20*n_iters):atom_id]) > 0:
                                    new_atom_id = max(start_idx, atom_id - 20*n_iters) + np.argmax(np.array(abs(dihedral_updates[max(start_idx, atom_id - 20*n_iters):atom_id])))
                                else:
                                    new_atom_id = max(start_idx, atom_id - 20*n_iters)
                            noise = all_noises[new_atom_id]
                            rotation_matrix = all_rotation_matrices[new_atom_id]
                            translation = all_translations[new_atom_id]
                            atom_step = new_atom_id - atom_id
                            atom_step_backtrack = copy.deepcopy(atom_step)
                            noise_level_multiplier = base_noise_level_multiplier
                            base_noise_level_multiplier = base_noise_level_multiplier
                            atom_clash_check = atom_id
                                                    
                if atom_step == 1:
                    noise_level_multiplier = base_noise_level_multiplier
                    all_noises[atom_id] = copy.deepcopy(noise)
                    if flex_type.find("processive") > -1:
                        all_rotation_matrices[atom_id] = copy.deepcopy(rotation_matrix)
                        all_translations[atom_id] = copy.deepcopy(translation)
                atom_id += atom_step
                if atom_id == start_idx + c - 5:
                    new_X = X[start_idx:start_idx+c] + noise[start_idx:start_idx+c]
                    new_X_displacement = np.mean(new_X, axis=0)
                    new_X = new_X - new_X_displacement
                    X_displacement = np.mean(X[start_idx:start_idx+c], axis=0)
                    X_chain = X[start_idx:start_idx+c] - X_displacement
                    R, d  = scipy.spatial.transform.Rotation.align_vectors(X_chain, new_X, weights = np.divide(1, new_X.shape[0]*np.ones(new_X.shape[0])))
                    new_X_rotated = R.apply(new_X)
                    noise[start_idx:start_idx+c] = new_X_rotated - X_chain
                
                if flex_type.find("stepwise") > -1 and atom_id > 100:
                    X = X.reshape((int(X.shape[0]/4), 4, 3))
                    noises = []
                    steps = list(all_noises.keys())
                    steps.sort()
                    for step in steps:
                        noise = all_noises[step].reshape(X.shape)
                        noises.append(noise)
                    dihedral_updates = dihedral_updates.reshape((X.shape[0], 4))
                    dihedral_updates = np.swapaxes(dihedral_updates, 0, 1)
                    return noises, dihedral_updates, fallback

            start_idx += c
        X = X.reshape((int(X.shape[0]/4), 4, 3))
        noise = noise.reshape(X.shape)

    if flex_type.find("dist_ang") > -1:
        noise += da_noise

    if dtype == 'torch':
        noise = torch.from_numpy(noise).to(device=dev)
    return noise
