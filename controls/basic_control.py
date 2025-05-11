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
Flg=True
eqn_count=0
count=0
points_per_dist=5
indx=0
path_nature='ini'    # or act
val=0.3
Kd=np.diag([1,1,1,14,1,1])*0.002
Kp=np.diag([1,1,1,0.5,1,1])*0.002
Ki=np.diag([1,1,1,1,1,1])*0.001*0
terr_prev=np.zeros((6,),dtype=np.float32)
start_count=0
buffr=0
bf=True

def init_controller(model,data):
    #initialize the controller here. This function is called once, in the beginning
    global inipath,path,integr,modl
    dh=np.array( [
    [0.0,     np.pi/2,  0.34, 0],
    [0.0,    -np.pi/2,  0.0 , 0],
    [0.4,    -np.pi/2,  0.0 , 0],
    [0.0,     np.pi/2,  0.4 , 0],
    [0.0,    -np.pi/2,  0.0 , 0],
    [0.0,     0.0,      0.126, 0]
])

    modl=Transforms(dh)
    traj_points=np.array([[0.7,-0.1,0.2,np.pi,0,0],[0.7,0.1,0.2,np.pi,0,0]])
    inipath,path=generate_path(model,data,traj_points)
    integr=np.zeros((6,1),dtype=np.float32)
    pass


def reach(model,data,despoint):
    return np.linalg.norm(despoint[:3]-data.xpos[-1])<0.01

def controller(model, data):
    #put the controller here. This function is called inside the simulation.
    global count,indx,path_nature,start_count
    #print(len(data.qpos))
    if start_count<1000:
        start_count+=1
        return
    elif start_count==1000:
        init_controller(model,data)
        start_count+=1
        return
    act_pt=np.hstack([data.xpos[-1],rotmat_to_euler(data.xmat[-1].reshape(3,3))])
    target_pt= inipath[indx] if path_nature=='ini' else path[indx]
    if path_nature=='ini' and indx==len(inipath):
        indx=0
        path_nature='act'
    
    if path_nature=='act' and indx==len(inipath):
        return
    if not reach(model,data,target_pt):
        _,J_inv=calc_invJacobian(model,data)
        try:
            teta_a=modl.solve_ik(*act_pt)
            print('type',type(teta_a))
        except Exception as e:
            print(e)
        x_diff=target_pt-act_pt
        teta_diff=(J_inv@x_diff.reshape(6,1)).flatten()
        teta_diff_PID=PID(model,data,teta_diff,J_inv)
        data.qpos[-6:]+=teta_diff_PID.flatten()
    if reach(model,data,target_pt):
        indx+=1
        print(f'target point : {target_pt} and new index={indx}')
    pass

def PID(model,data,x_err,jm):
    global integr,Kp,Kd,Ki,terr_prev
    #print(x_err)
    ki=jm@Ki
    kp=jm@Kp
    kd=jm@Kd
    integr+=((terr_prev.reshape(6,1)+x_err.reshape(6,1))*0.5)
    terr_prev=x_err
    #print(Kp@(x_err.reshape(6,1)) + integr + Kd@((x_err.reshape(6,1)-terr_prev.reshape(6,1))))
    return kp@(x_err.reshape(6,1)) + ki@integr + kd@((x_err.reshape(6,1)-terr_prev.reshape(6,1)))

def generate_path(model,data,xpoints,lingen=True):
    atpoint=np.hstack([data.xpos[-1],rotmat_to_euler(data.xmat[-1].reshape(3,3))])   #format x,y,z,r,p,y
    stpoint=xpoints[0]
    dist=np.linalg.norm(stpoint[:3]-atpoint[:3])
    path1=np.linspace(atpoint,stpoint,int(dist*points_per_dist))
    path_lst=[]
    if lingen:
        for i in range(1,len(xpoints)):
            dists=np.linalg.norm(xpoints[i]-xpoints[i-1])
            pathi=np.linspace(xpoints[i-1],xpoints[i],int(dists*points_per_dist))
            path_lst.append(pathi)
        path2=np.hstack(path_lst)
    #path=np.hstack([path1,path2])
    return path1,path2

def rotmat_to_euler(val):
    return R.from_matrix(val).as_euler('xyz')

def euler_to_rotmat(val):
    return R.from_euler('xyz',val,degrees=False).as_matrix()

def calc_invJacobian(model,data):
        try:
            body_name="translated_frame"
            body_id=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,body_name)
            jacp = np.zeros((3, model.nv))  # Position Jacobian (3, nv)
            jacr = np.zeros((3, model.nv))  # Rotation Jacobian (3, nv)
            mj.mj_jacBody(model, data, jacp, jacr, body_id)
            jacp=jacp[:,:]      #[:,6:12]
            jacr=jacr[:,:]      #[:,6:12]
            J = np.vstack([jacp,jacr])
            return True,np.linalg.inv(J)
        except Exception as e:
            print(e)
            print(J)
            return False,J

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos*np.eye(6)
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
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

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

#get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)


####################################
#opt.frame = mj.mjtFrame.mjFRAME_BODY

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration
cam.azimuth = 90
cam.elevation = -30
cam.distance = 4.101136
cam.lookat = np.array([ 1.2098175552578596 , 0.07084043959349502 , 0.2512892541810182 ])

#initialize the controller
#init_controller(model,data)

#set the controller
mj.set_mjcb_control(controller)

while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev < 0.01):
        mj.mj_step(model, data)
        
    
    if (data.time>=simend):
        break

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    #print camera configuration (help to initialize the view)
    if (print_camera_config==1):
        print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
        print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()