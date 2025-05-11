import mujoco as mj
from mujoco.viewer import launch
from mujoco.glfw import glfw
import numpy as np
from transforms import Transforms
import sympy as sp
import os
from scipy.spatial.transform import Rotation as R

xml_path = '../robot.xml' #xml file (assumes this is in the same folder as this file)
simend = 1000000 #simulation time
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

# Path following parameters
points_per_dist = 10  # Increased for smoother paths
indx = 0
path_nature = 'ini'  # Starting with initialization path
start_count = 0

# PID Controller Gains - Adjusted for better response
# Reduced derivative gains to minimize vibration
# Optimized for fast response with minimal vibration
# More conservative approach if vibration persists
# Extreme convergence with vibration control
Kp = np.diag([18.0, 18.0, 18.0, 2.0, 2.0, 2.0]) * 0.002  # Significantly increased proportional gains
Ki = np.diag([1.0, 1.0, 1.0, 0.25, 0.25, 0.25]) * 0.001  # Increased integral gains
Kd = np.diag([3.0, 3.0, 3.0, 0.6, 0.6, 0.6]) * 0.002  # Increased derivative gains with proper ratio # Balanced derivative gains

# Error tracking
terr_prev = np.zeros(6, dtype=np.float32)
integr = np.zeros((6, 1), dtype=np.float32)

# Path visualization
path_markers = []
current_path_index = 0
visualize_path = True


def init_controller(model, data):
    """Initialize the controller with paths and transforms"""
    global inipath, path, integr, modl, path_markers
    
    # Define DH parameters for the robot
    dh = np.array([
        [0.0,     np.pi/2,  0.34, 0],
        [0.0,    -np.pi/2,  0.0 , 0],
        [0.4,    -np.pi/2,  0.0 , 0],
        [0.0,     np.pi/2,  0.4 , 0],
        [0.0,    -np.pi/2,  0.0 , 0],
        [0.0,     0.0,      0.126, 0]
    ])

    # Initialize transform model
    modl = Transforms(dh)
    
    # Define trajectory points (can be customized)
    # Format: [x, y, z, roll, pitch, yaw]
    traj_points = np.array([
        [0.7, -0.1, 0.2, np.pi, 0, 0],  # Point 1
        [0.7,  0.1, 0.2, np.pi, 0, 0],  # Point 2
        [0.7,  0.1, 0.4, np.pi, 0, 0],  # Point 3
        [0.5,  0.1, 0.4, np.pi, 0, 0],  # Point 4
        [0.5, -0.1, 0.4, np.pi, 0, 0],  # Point 5
        [0.5, -0.1, 0.2, np.pi, 0, 0],  # Point 6
        [0.7, -0.1, 0.2, np.pi, 0, 0],  # Back to Point 1
    ])
    
    # Generate paths for initialization and actual trajectory
    inipath, path = generate_path(model, data, traj_points)
    
    # Reset integral term
    integr = np.zeros((6, 1), dtype=np.float32)
    
    print(f"Controller initialized with paths:")
    print(f"Initialization path: {len(inipath)} points")
    print(f"Main path: {len(path)} points")


def reach(model, data, despoint, threshold=0.01):
    """Determine if the end effector has reached the target point"""
    # Check position only for simplicity
    pos_error = np.linalg.norm(despoint[:3] - data.xpos[-1])
    
    # If orientation is critical, you can add orientation check here
    # ori_error = np.linalg.norm(despoint[3:] - rotmat_to_euler(data.xmat[-1].reshape(3,3)))
    
    return pos_error < threshold


def controller(model, data):
    """Main controller function that follows a set of points"""
    global count, indx, path_nature, start_count, integr, terr_prev, current_path_index, inipath, path
    
    # Wait for simulation to stabilize
    if start_count < 1000:
        start_count += 1
        return
    elif start_count == 1000:
        init_controller(model, data)
        start_count += 1
        return
    
    # Get current end effector pose
    act_pos = data.xpos[-1]
    act_ori = rotmat_to_euler(data.xmat[-1].reshape(3, 3))
    act_pt = np.hstack([act_pos, act_ori])
    
    # Get current target point based on path type and index
    if path_nature == 'ini':
        if indx >= len(inipath):
            print("Initialization path completed, switching to main path")
            indx = 0
            path_nature = 'act'
            # Reset integral term when switching paths
            integr = np.zeros((6, 1), dtype=np.float32)
            current_path_index = 0
            return
        target_pt = inipath[indx]
    else:  # path_nature == 'act'
        if indx >= len(path):
            print("Main path completed")
            # You can choose to loop through the path again
            indx = 0  # Reset to start of path
            # Reset integral term
            integr = np.zeros((6, 1), dtype=np.float32)
            return
        target_pt = path[indx]
        current_path_index = indx
    
    # Check if we've reached the target point
    if reach(model, data, target_pt):
        indx += 1
        print(f"Target point reached: {target_pt}, moving to point {indx}")
        # Reset integral term for new point to prevent accumulated error
        integr = np.zeros((6, 1), dtype=np.float32)
        return
    
    # If not reached target, calculate control action
    success, J_inv = calc_invJacobian(model, data)
    
    if not success:
        print("Warning: Failed to calculate inverse Jacobian")
        return
    
    # Calculate pose error
    x_diff = target_pt - act_pt
    try:
        teta_a=modl.solve_ik(*act_pt)
        print('type',type(teta_a))
    except Exception as e:
        print(e)
    # Scale orientation errors (they're often in different units than position)
    x_diff[3:] *= 0.5  # Scale down orientation errors
    
    # Apply PID control to get joint corrections
    teta_diff_PID = PID(model, data, x_diff, J_inv)
    
    # Apply joint corrections with a damping factor for stability
    damping = 0.5  # Adjust based on stability requirements
    delta_q = damping * teta_diff_PID.flatten()
    
    # Apply joint limits if needed
    joint_limits = 0.05  # Maximum change per step
    delta_q = np.clip(delta_q, -joint_limits, joint_limits)
    
    # Update joint positions
    data.qpos[-6:] += delta_q


def PID(model, data, x_err, J_inv):
    """Enhanced PID controller for better path following"""
    global integr, Kp, Ki, Kd, terr_prev
    
    # Map gains to joint space
    kp = J_inv @ Kp
    ki = J_inv @ Ki
    kd = J_inv @ Kd
    
    # Anti-windup for integral term
    max_integr = 10.0
    
    # Update integral term with trapezoidal rule
    integr += ((terr_prev.reshape(6, 1) + x_err.reshape(6, 1)) * 0.5) * 0.01  # Small dt
    
    # Apply integral limit (anti-windup)
    for i in range(6):
        if abs(integr[i, 0]) > max_integr:
            integr[i, 0] = np.sign(integr[i, 0]) * max_integr
    
    # Calculate PID terms
    p_term = kp @ (x_err.reshape(6, 1))
    i_term = ki @ integr
    d_term = kd @ ((x_err.reshape(6, 1) - terr_prev.reshape(6, 1)) / 0.01)  # Using dt=0.01
    
    # Save current error for next iteration
    terr_prev = x_err.copy()
    
    # Combine terms
    return p_term + i_term + d_term


def generate_path(model, data, xpoints, lingen=True):
    """Generate a smooth path between points"""
    # Get current end effector pose
    at_pos = data.xpos[-1]
    at_ori = rotmat_to_euler(data.xmat[-1].reshape(3, 3))
    at_point = np.hstack([at_pos, at_ori])
    
    # Start point from trajectory
    st_point = xpoints[0]
    
    # Distance for initialization path (current position to first trajectory point)
    dist = np.linalg.norm(st_point[:3] - at_point[:3])
    
    # Generate initialization path with more points for smoother motion
    path1 = np.linspace(at_point, st_point, max(5, int(dist * points_per_dist)))
    
    # Generate main trajectory path
    path_segments = []
    if lingen:
        for i in range(1, len(xpoints)):
            # Calculate distance between consecutive points
            dist = np.linalg.norm(xpoints[i][:3] - xpoints[i-1][:3])
            
            # Generate linear interpolation with more points for smoother motion
            segment = np.linspace(xpoints[i-1], xpoints[i], max(5, int(dist * points_per_dist)))
            path_segments.append(segment)
        
        # Concatenate all segments
        if path_segments:
            path2 = np.vstack(path_segments)
        else:
            path2 = np.array([])
    
    return path1, path2


def rotmat_to_euler(val):
    """Convert rotation matrix to Euler angles"""
    return R.from_matrix(val).as_euler('xyz')


def euler_to_rotmat(val):
    """Convert Euler angles to rotation matrix"""
    return R.from_euler('xyz', val, degrees=False).as_matrix()


def calc_invJacobian(model, data):
    """Calculate the inverse Jacobian matrix with error handling"""
    try:
        # Get body ID for end effector
        body_name = "translated_frame"
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
        
        # Initialize Jacobian matrices
        jacp = np.zeros((3, model.nv))  # Position Jacobian (3, nv)
        jacr = np.zeros((3, model.nv))  # Rotation Jacobian (3, nv)
        
        # Calculate Jacobian
        mj.mj_jacBody(model, data, jacp, jacr, body_id)
        
        # Extract relevant part of Jacobian (for the last 6 joints if needed)
        # jacp = jacp[:, -6:]  # Uncomment if you want to use only the last 6 joints
        # jacr = jacr[:, -6:]  # Uncomment if you want to use only the last 6 joints
        
        # Stack position and rotation parts
        J = np.vstack([jacp, jacr])
        
        # Calculate pseudoinverse for better stability
        J_inv = np.linalg.pinv(J)
        
        return True, J_inv
    
    except Exception as e:
        print(f"Error in calc_invJacobian: {e}")
        # Return identity matrix as fallback
        return False, np.eye(6)


def add_path_visualization(scene, model, pos, color=None):
    """Add a marker at the specified position for path visualization"""
    if color is None:
        color = [1, 0, 0, 0.8]  # Red by default
    
    # Create a sphere at the position
    mj.mjv_initGeom(
        scene.geoms[scene.ngeom], 
        mj.mjtGeom.mjGEOM_SPHERE,
        np.zeros(3),  # Size parameters
        pos,  # Position
        np.eye(3).flatten(),  # Orientation as flattened rotation matrix
        color  # RGBA color
    )
    
    # Set size (radius of sphere)
    scene.geoms[scene.ngeom].size[0] = 0.01  # Small sphere
    
    # Set category for rendering
    scene.geoms[scene.ngeom].category = mj.mjtCatBit.mjCAT_DECOR
    
    # Increment geom counter
    scene.ngeom += 1


def keyboard(window, key, scancode, act, mods):
    """Handle keyboard inputs"""
    global visualize_path
    
    if act == glfw.PRESS:
        if key == glfw.KEY_BACKSPACE:
            # Reset simulation
            mj.mj_resetData(model, data)
            mj.mj_forward(model, data)
        elif key == glfw.KEY_V:
            # Toggle path visualization
            visualize_path = not visualize_path
            print(f"Path visualization: {'ON' if visualize_path else 'OFF'}")


def mouse_button(window, button, act, mods):
    """Handle mouse button events"""
    global button_left, button_middle, button_right
    
    # Update button state
    button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
    
    # Update mouse position
    glfw.get_cursor_pos(window)


def mouse_move(window, xpos, ypos):
    """Handle mouse movement"""
    global lastx, lasty, button_left, button_middle, button_right
    
    # Compute mouse displacement, save
    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos
    
    # No buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return
    
    # Get current window size
    width, height = glfw.get_window_size(window)
    
    # Get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)
    
    # Determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM
    
    # Move camera
    mj.mjv_moveCamera(model, action, dx/height, dy/height, scene, cam)


def scroll(window, xoffset, yoffset):
    """Handle scroll wheel events"""
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 * yoffset, scene, cam)


def visualize_paths(scene, model):
    """Add visualization of the entire path"""
    global path_nature, indx, current_path_index
    
    # Only visualize if controller is initialized and paths exist
    if not hasattr(globals(), 'inipath') or not hasattr(globals(), 'path'):
        return
        
    # Get paths from global scope
    inipath = globals().get('inipath', [])
    path = globals().get('path', [])
    
    # Don't try to visualize if paths aren't defined yet
    if len(inipath) == 0 and len(path) == 0:
        return
    
    # Visualize initialization path
    if path_nature == 'ini':
        for i, point in enumerate(inipath):
            color = [0, 0, 1, 0.5] if i >= indx else [0, 1, 0, 0.5]  # Blue for future, green for passed
            add_path_visualization(scene, model, point[:3], color)
    
    # Visualize main path
    for i, point in enumerate(path):
        if path_nature == 'act':
            color = [0, 0, 1, 0.5] if i >= indx else [0, 1, 0, 0.5]  # Blue for future, green for passed
        else:
            color = [0.5, 0.5, 0.5, 0.3]  # Gray for inactive path
        add_path_visualization(scene, model, point[:3], color)
    
    # Highlight current target
    if path_nature == 'ini' and indx < len(inipath):
        add_path_visualization(scene, model, inipath[indx][:3], [1, 0, 0, 1])  # Red for current target
    elif path_nature == 'act' and indx < len(path):
        add_path_visualization(scene, model, path[indx][:3], [1, 0, 0, 1])  # Red for current target


# Get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                     # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Robot Path Following", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# Initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)  # Increased maxgeom for path visualization
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# Install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration
cam.azimuth = 90
cam.elevation = -30
cam.distance = 4.101136
cam.lookat = np.array([1.2098175552578596, 0.07084043959349502, 0.2512892541810182])

# Set the controller
mj.set_mjcb_control(controller)

# Initialize global variables to avoid NameError
inipath = []
path = []

# Main simulation loop
while not glfw.window_should_close(window):
    #print(data.xpos[-1])
    time_prev = data.time
    
    # Step simulation
    while (data.time - time_prev < 0.01):
        mj.mj_step(model, data)
    
    # Check for simulation end
    if (data.time >= simend):
        break
    
    # Get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
    
    # Print camera configuration (help to initialize the view)
    if (print_camera_config == 1):
        print('cam.azimuth =', cam.azimuth, ';', 'cam.elevation =', cam.elevation, ';', 'cam.distance = ', cam.distance)
        print('cam.lookat =np.array([', cam.lookat[0], ',', cam.lookat[1], ',', cam.lookat[2], '])')
    
    # Update scene
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    
    # Add path visualization if enabled
    if visualize_path:
        visualize_paths(scene, model)
    
    # Render scene
    mj.mjr_render(viewport, scene, context)
    
    # Swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)
    
    # Process pending GUI events, call GLFW callbacks
    glfw.poll_events()

# Clean up
glfw.terminate()