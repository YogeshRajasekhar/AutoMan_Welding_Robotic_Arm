import numpy as np
import sympy as sp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define symbolic variables for DH parameters
def create_symbolic_dh_parameters(num_joints):
    """Create symbolic variables for DH parameters"""
    # a (link length), alpha (link twist), d (link offset), theta (joint angle)
    a_params = []
    alpha_params = []
    d_params = []
    theta_offsets = []
    
    for i in range(num_joints):
        a_params.append(sp.symbols(f'a_{i}'))
        alpha_params.append(sp.symbols(f'alpha_{i}'))
        d_params.append(sp.symbols(f'd_{i}'))
        theta_offsets.append(sp.symbols(f'theta_offset_{i}'))
    
    return a_params, alpha_params, d_params, theta_offsets

def create_symbolic_joint_angles(num_joints):
    """Create symbolic variables for joint angles"""
    theta = []
    for i in range(num_joints):
        theta.append(sp.symbols(f'theta_{i}'))
    return theta

def dh_transform_matrix(a, alpha, d, theta):
    """Create a DH transformation matrix using symbolic parameters"""
    # Standard DH transformation matrix
    T = sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha), sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
        [sp.sin(theta), sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
        [0, sp.sin(alpha), sp.cos(alpha), d],
        [0, 0, 0, 1]
    ])
    return T

def forward_kinematics_symbolic(num_joints):
    """Generate symbolic forward kinematics expressions"""
    # Create symbolic variables
    a_params, alpha_params, d_params, theta_offsets = create_symbolic_dh_parameters(num_joints)
    theta = create_symbolic_joint_angles(num_joints)
    
    # List to store all symbols for later substitution
    all_symbols = a_params + alpha_params + d_params + theta_offsets + theta
    
    # Initialize transformation matrix as identity
    T = sp.eye(4)
    
    # Build the forward kinematics by chaining transformations
    for i in range(num_joints):
        # Calculate the actual joint angle (theta + offset)
        actual_theta = theta[i] + theta_offsets[i]
        
        # Calculate the transformation matrix for this joint
        T_i = dh_transform_matrix(a_params[i], alpha_params[i], d_params[i], actual_theta)
        
        # Update the transformation chain
        T = T * T_i
    
    # Create lambdified functions for position and orientation
    position = T[:3, 3]
    rotation_matrix = T[:3, :3]
    
    # Return symbolic expressions and symbols
    return T, position, rotation_matrix, all_symbols

def jacobian_symbolic(position, dh_params, joint_angles):
    """Generate symbolic Jacobian expressions"""
    # Calculate the Jacobian matrix for position
    J = sp.Matrix.zeros(3, len(joint_angles))
    
    for i in range(len(joint_angles)):
        J[:, i] = sp.diff(position, joint_angles[i])
    
    return J

def compile_functions(num_joints):
    """Compile numerical functions for forward kinematics and Jacobian"""
    # Get symbolic expressions
    T, position, rotation, all_symbols = forward_kinematics_symbolic(num_joints)
    
    # Extract symbol lists
    a_params, alpha_params, d_params, theta_offsets = create_symbolic_dh_parameters(num_joints)
    theta = create_symbolic_joint_angles(num_joints)
    
    # Flatten DH parameters
    dh_params = []
    for i in range(num_joints):
        dh_params.extend([a_params[i], alpha_params[i], d_params[i], theta_offsets[i]])
    
    # Compute position Jacobian
    J = jacobian_symbolic(position, dh_params, theta)
    
    # Lambdify for numerical evaluation
    param_list = dh_params + theta
    
    # Convert symbolic expressions to numpy functions
    fk_position_func = sp.lambdify(param_list, position, 'numpy')
    fk_rotation_func = sp.lambdify(param_list, rotation, 'numpy')
    jacobian_func = sp.lambdify(param_list, J, 'numpy')
    
    return fk_position_func, fk_rotation_func, jacobian_func

def evaluate_fk(fk_position_func, fk_rotation_func, dh_params, joint_angles):
    """Evaluate forward kinematics using compiled functions"""
    # Flatten the parameters
    params = []
    num_joints = len(joint_angles)
    
    for i in range(num_joints):
        params.extend([dh_params[i][0], dh_params[i][1], dh_params[i][2], dh_params[i][3]])
    params.extend(joint_angles)
    
    # Evaluate
    position = fk_position_func(*params)
    rotation = fk_rotation_func(*params)
    
    return position, rotation

def evaluate_jacobian(jacobian_func, dh_params, joint_angles):
    """Evaluate Jacobian using compiled function"""
    # Flatten the parameters
    params = []
    num_joints = len(joint_angles)
    
    for i in range(num_joints):
        params.extend([dh_params[i][0], dh_params[i][1], dh_params[i][2], dh_params[i][3]])
    params.extend(joint_angles)
    
    # Evaluate
    J = jacobian_func(*params)
    
    return J

# Error function for optimization
def error_function(dh_params_flat, fk_position_func, fk_rotation_func, num_joints, joint_configs, observed_poses):
    """Calculate error between predicted and observed end-effector poses"""
    # Reshape flat DH parameters to expected format
    dh_params = dh_params_flat.reshape(num_joints, 4)
    
    total_error = 0
    for i, joint_angles in enumerate(joint_configs):
        # Calculate predicted pose
        pred_pos, pred_rot = evaluate_fk(fk_position_func, fk_rotation_func, dh_params, joint_angles)
        
        # Extract observed position and rotation
        obs_pos = observed_poses[i][:3, 3]
        obs_rot = observed_poses[i][:3, :3]
        
        # Position error (Euclidean distance)
        pos_error = np.linalg.norm(pred_pos - obs_pos)
        
        # Orientation error (Frobenius norm of difference)
        # Alternative: geodesic distance on SO(3)
        rot_error = np.linalg.norm(pred_rot - obs_rot, 'fro')
        
        # Combined error
        total_error += pos_error + 0.1 * rot_error
    
    # Add regularization to prevent extreme values
    reg_term = 0.01 * np.sum(dh_params_flat**2)
    
    return total_error + reg_term

# Jacobian function for optimization
def error_jacobian(dh_params_flat, jacobian_func, fk_position_func, fk_rotation_func, num_joints, joint_configs, observed_poses):
    """Calculate the gradient of the error function with respect to DH parameters"""
    # This is a numerical approximation
    epsilon = 1e-6
    grad = np.zeros_like(dh_params_flat)
    
    for i in range(len(dh_params_flat)):
        # Forward difference
        dh_params_plus = dh_params_flat.copy()
        dh_params_plus[i] += epsilon
        
        error_plus = error_function(dh_params_plus, fk_position_func, fk_rotation_func, 
                                    num_joints, joint_configs, observed_poses)
        error = error_function(dh_params_flat, fk_position_func, fk_rotation_func, 
                               num_joints, joint_configs, observed_poses)
        
        # Approximate gradient
        grad[i] = (error_plus - error) / epsilon
    
    return grad

# Main function to estimate DH parameters
def estimate_dh_parameters(num_joints, joint_configs, observed_poses, initial_guess=None):
    """Estimate DH parameters from joint configurations and observed end-effector poses"""
    # Compile functions for kinematics
    fk_position_func, fk_rotation_func, jacobian_func = compile_functions(num_joints)
    
    # Initial parameter guess if not provided
    if initial_guess is None:
        # 4 DH parameters per joint: a, alpha, d, theta_offset
        initial_guess = np.zeros(4 * num_joints)
    
    # Bounds for each parameter type
    bounds = []
    for _ in range(num_joints):
        # [a, alpha, d, theta_offset]
        bounds.extend([
            (-0.5, 0.5),      # Link length (a)
            (-np.pi, np.pi),  # Link twist (alpha)
            (-0.5, 0.5),      # Link offset (d)
            (-np.pi, np.pi)   # Joint angle offset (theta_offset)
        ])
    
    # Define the optimization objective
    obj_func = lambda params: error_function(params, fk_position_func, fk_rotation_func, 
                                            num_joints, joint_configs, observed_poses)
    
    # Define the Jacobian function for optimization
    jac_func = lambda params: error_jacobian(params, jacobian_func, fk_position_func, fk_rotation_func,
                                            num_joints, joint_configs, observed_poses)
    
    # Two-step optimization for better convergence
    # Step 1: Coarse optimization with relaxed tolerances
    print("Starting coarse optimization...")
    coarse_result = minimize(
        obj_func,
        initial_guess,
        method='L-BFGS-B',
        bounds=bounds,
        options={'gtol': 1e-4, 'maxiter': 100}
    )
    
    # Step 2: Fine optimization with tighter tolerances
    print("Starting fine optimization...")
    fine_result = minimize(
        obj_func,
        coarse_result.x,
        method='L-BFGS-B',
        jac=jac_func,
        bounds=bounds,
        options={'gtol': 1e-8, 'maxiter': 500}
    )
    
    # Reshape result to [num_joints, 4] format
    dh_params = fine_result.x.reshape(num_joints, 4)
    
    # Print optimization results
    print(f"Optimization converged: {fine_result.success}")
    print(f"Final error: {fine_result.fun}")
    print(f"Number of iterations: {fine_result.nit}")
    
    return dh_params, fk_position_func, fk_rotation_func

# Generate synthetic data for testing
def generate_test_data(num_joints, true_dh_params, num_samples=20):
    """Generate synthetic test data with known DH parameters"""
    # Compile functions for kinematics
    fk_position_func, fk_rotation_func, _ = compile_functions(num_joints)
    
    # Generate random joint configurations
    joint_configs = []
    for _ in range(num_samples):
        # Random joint angles within reasonable limits
        joint_angles = np.random.uniform(-np.pi/2, np.pi/2, num_joints)
        joint_configs.append(joint_angles)
    
    # Calculate poses for each configuration
    observed_poses = []
    for joint_angles in joint_configs:
        position, rotation = evaluate_fk(fk_position_func, fk_rotation_func, true_dh_params, joint_angles)
        
        # Create homogeneous transformation matrix
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = position
        
        # Add some noise to make it realistic
        T[:3, 3] += np.random.normal(0, 0.005, 3)  # Position noise
        observed_poses.append(T)
    
    return joint_configs, observed_poses

# Visualize the results
def visualize_results(true_dh_params, estimated_dh_params, joint_configs, observed_poses, 
                      fk_position_func, fk_rotation_func):
    """Visualize the results by comparing true and estimated end-effector positions"""
    # Create a 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot observed positions
    observed_positions = [pose[:3, 3] for pose in observed_poses]
    observed_x = [pos[0] for pos in observed_positions]
    observed_y = [pos[1] for pos in observed_positions]
    observed_z = [pos[2] for pos in observed_positions]
    
    ax.scatter(observed_x, observed_y, observed_z, c='b', marker='o', label='Observed Positions')
    
    # Plot estimated positions
    estimated_positions = []
    for joint_angles in joint_configs:
        position, _ = evaluate_fk(fk_position_func, fk_rotation_func, estimated_dh_params, joint_angles)
        estimated_positions.append(position)
    
    estimated_x = [pos[0] for pos in estimated_positions]
    estimated_y = [pos[1] for pos in estimated_positions]
    estimated_z = [pos[2] for pos in estimated_positions]
    
    ax.scatter(estimated_x, estimated_y, estimated_z, c='r', marker='x', label='Estimated Positions')
    
    # Connect corresponding points
    for i in range(len(observed_positions)):
        ax.plot([observed_x[i], estimated_x[i]], 
                [observed_y[i], estimated_y[i]], 
                [observed_z[i], estimated_z[i]], 'g-', alpha=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Comparison of Observed and Estimated End-Effector Positions')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Compare DH parameters
    print("\nDH Parameter Comparison:")
    print("True DH Parameters:")
    for i, params in enumerate(true_dh_params):
        print(f"Joint {i}: a={params[0]:.4f}, alpha={params[1]:.4f}, d={params[2]:.4f}, theta_offset={params[3]:.4f}")
    
    print("\nEstimated DH Parameters:")
    for i, params in enumerate(estimated_dh_params):
        print(f"Joint {i}: a={params[0]:.4f}, alpha={params[1]:.4f}, d={params[2]:.4f}, theta_offset={params[3]:.4f}")
    
    # Calculate and print error metrics
    position_errors = []
    rotation_errors = []
    
    for i, joint_angles in enumerate(joint_configs):
        # Calculate predicted pose with estimated parameters
        est_pos, est_rot = evaluate_fk(fk_position_func, fk_rotation_func, estimated_dh_params, joint_angles)
        
        # Extract observed position and rotation
        obs_pos = observed_poses[i][:3, 3]
        obs_rot = observed_poses[i][:3, :3]
        
        # Position error
        pos_error = np.linalg.norm(est_pos - obs_pos)
        position_errors.append(pos_error)
        
        # Orientation error
        rot_error = np.linalg.norm(est_rot - obs_rot, 'fro')
        rotation_errors.append(rot_error)
    
    print(f"\nAverage Position Error: {np.mean(position_errors):.6f} units")
    print(f"Average Rotation Error: {np.mean(rotation_errors):.6f}")


# Example usage
if __name__ == "__main__":
    # Define number of joints
    num_joints = 3
    
    # Define true DH parameters for a simple 3-joint robot
    # Format: [a, alpha, d, theta_offset] for each joint
    true_dh_params = np.array([
        [0.1, np.pi/2, 0.2, 0.0],    # Joint 0
        [0.3, 0.0, 0.0, 0.0],        # Joint 1
        [0.2, -np.pi/2, 0.1, 0.0]    # Joint 2
    ])
    
    print("Generating test data...")
    joint_configs, observed_poses = generate_test_data(num_joints, true_dh_params, num_samples=30)
    
    print("Estimating DH parameters...")
    estimated_dh_params, fk_position_func, fk_rotation_func = estimate_dh_parameters(
        num_joints, joint_configs, observed_poses
    )
    
    print("Visualizing results...")
    visualize_results(true_dh_params, estimated_dh_params, joint_configs, observed_poses,
                     fk_position_func, fk_rotation_func)