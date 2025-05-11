import numpy as np
import sympy as sp
from sympy import *
from scipy.spatial.transform import Rotation as R

class Transforms():
    def __init__(self,dh,ini_T=None):
        self.dh=dh
        self.ini_T = ini_T
        self.theta_symbols = sp.symbols(f'theta0:{len(self.dh)}')
        self.A = sp.symbols(f'a0:6')
        self.Alpha = sp.symbols(f'alpha0:6')
        self.D = sp.symbols(f'd0:6')

        x, y, z = symbols('x y z')
        roll, pitch, yaw = symbols('roll pitch yaw')
        self.pose_symbols = (x, y, z, roll, pitch, yaw)

        try:
            if ini_T.shape == (4,4):
                self.function = self.transforms(ini_T)
            else:
                self.function = self.transforms()
        except:
            self.function = self.transforms()
        self.jacobian_matrix = self.calculate_jacobian()
        self.solve_inverse_kinematics()
        self.lambdaf = self.lambdify()
        self.lambdaGen = self.lambdafy_gen()
        self.jacobian_func = self.lambdify_jacobian()

    def transforms(self,ini_T=None):
        if ini_T is None:
            T_f=np.eye(4)
            T_test=np.eye(4)
        else:
            T_f=np.matrix(ini_T)
        for i in range(len(self.dh)):
            a,alpha,d,_=self.dh[i]
            teta=self.theta_symbols[i]
            a_,alpha_,d_=self.A[i],self.Alpha[i],self.D[i]
            T=sp.Matrix([[sp.cos(teta)               ,-sp.sin(teta)              ,  0            ,  a               ],
                        [sp.cos(alpha)*sp.sin(teta)  , sp.cos(alpha)*sp.cos(teta), -sp.sin(alpha),  sp.sin(alpha)*d ],
                        [sp.sin(alpha)*sp.sin(teta)  , sp.sin(alpha)*sp.cos(teta),  sp.cos(alpha),  sp.cos(alpha)*d ],
                        [0                           , 0                         ,  0            ,  1               ]
                    ])
            
            T2=sp.Matrix([[sp.cos(teta)              ,-sp.sin(teta)              ,  0            ,  a_              ],
                        [sp.cos(alpha_)*sp.sin(teta) , sp.cos(alpha_)*sp.cos(teta), -sp.sin(alpha_),  sp.sin(alpha_)*d_ ],
                        [sp.sin(alpha_)*sp.sin(teta)  , sp.sin(alpha_)*sp.cos(teta),  sp.cos(alpha_),  sp.cos(alpha_)*d_ ],
                        [0                           , 0                         ,  0            ,  1               ]
                    ])
            self.T_test=T_test@T2
            T_f=T_f@T
        return T_f
    def solve_inverse_kinematics(self):
        """
        Set up the inverse kinematics problem framework.
        This prepares the expressions but doesn't solve for specific values.
        """
        # Define joint angle symbols (t1 through t6)
        t1, t2, t3, t4, t5, t6 = self.theta_symbols
        
        # Define end effector pose symbols
        
        
        # Extract position and orientation from transformation matrix
        position = self.function[0:3, 3]
        rotation = self.function[0:3, 0:3]
        
        # Convert rotation matrix to roll, pitch, yaw angles
        rpy = [sp.atan2(rotation[2, 1], rotation[2, 2]),
            sp.asin(-rotation[2, 0]),
            sp.atan2(rotation[1, 0], rotation[0, 0])]
        x, y, z, roll, pitch, yaw = self.pose_symbols
        # Define forward kinematics expressions
        self.fk_expressions = {
            x: position[0],
            y: position[1],
            z: position[2],
            roll: rpy[0],
            pitch: rpy[1],
            yaw: rpy[2]
        }
        
        # Store symbols for later use
        
        return self.fk_expressions

    def solve_ik(self, target_x, target_y, target_z, target_roll, target_pitch, target_yaw, initial_guess=None):
        """
        Solve inverse kinematics for the robot.
        
        Args:
            target_x, target_y, target_z: Target position
            target_roll, target_pitch, target_yaw: Target orientation
            initial_guess: Initial guess for joint angles (optional)
            
        Returns:
            The joint angles [t1, t2, t3, t4, t5, t6] that achieve the target pose
        """
        t1, t2, t3, t4, t5, t6 = self.theta_symbols
        x, y, z, roll, pitch, yaw = self.pose_symbols
        
        # Create the system of equations
        equations = [
            self.fk_expressions[x] - target_x,
            self.fk_expressions[y] - target_y,
            self.fk_expressions[z] - target_z,
            self.fk_expressions[roll] - target_roll,
            self.fk_expressions[pitch] - target_pitch,
            self.fk_expressions[yaw] - target_yaw
        ]
        
        # Determine initial guess if not provided
        if initial_guess is None:
            initial_guess = {t1: 0, t2: 0, t3: 0, t4: 0, t5: 0, t6: 0}
            
        # Try using sympy's solve function for analytical solution
        try:
            print("Attempting analytical solution...")
            solutions = solve(equations, [t1, t2, t3, t4, t5, t6], dict=True)
            
            if solutions:
                print(f"Found {len(solutions)} analytical solutions")
                # Select the solution with minimum joint movement from initial position
                best_solution = min(solutions, key=lambda sol: sum(abs(sol[t] - initial_guess.get(t, 0))**2 
                                                            for t in [t1, t2, t3, t4, t5, t6]))
                return [best_solution[t] for t in [t1, t2, t3, t4, t5, t6]]
            else:
                print("No analytical solution found, trying numerical approach")
        except Exception as e:
            print(f"Analytical solution failed: {str(e)}")
            print("Switching to numerical approach")
        
        # If analytical solution fails, use numerical optimization
        from scipy.optimize import minimize
        
        # Convert symbolic equations to numerical functions
        joint_vars = [t1, t2, t3, t4, t5, t6]
        pose_vars = [x, y, z, roll, pitch, yaw]
        target_vals = [target_x, target_y, target_z, target_roll, target_pitch, target_yaw]
        
        # Create numerical error function
        def error_func(joint_vals):
            # Substitute joint values into forward kinematics
            substitutions = {joint_vars[i]: joint_vals[i] for i in range(6)}
            
            # Calculate resulting pose
            result_pose = [float(self.fk_expressions[pose_var].subs(substitutions).evalf()) 
                        for pose_var in pose_vars]
            
            # Calculate error (squared distance)
            error = sum((result_pose[i] - target_vals[i])**2 for i in range(6))
            return error
        
        # Initial guess for numerical optimization
        initial_joints = [initial_guess.get(joint, 0) for joint in joint_vars]
        
        # Run optimization
        result = minimize(error_func, initial_joints, method='BFGS')
        
        if result.success:
            print("Found numerical solution")
            return result.x
        else:
            print("Failed to find solution")
            return None

    def derive_analytical_ik(self):
        """
        Attempt to derive analytical inverse kinematics solutions.
        This is advanced and may not work for all robot configurations.
        """
        t1, t2, t3, t4, t5, t6 = self.theta_symbols
        x, y, z, roll, pitch, yaw = self.pose_symbols
        
        try:
            print("Attempting to derive analytical inverse kinematics...")
            
            # Step 1: Solve for t1 (often depends on x and y)
            # Example (replace with your derivation):
            # t1_sol = sp.atan2(y, x)
            
            # Step 2: Solve for t5 (often from orientation)
            # Example (replace with your derivation):
            # t5_sol = sp.asin(-sp.sin(pitch))
            
            # Step 3: Solve for remaining angles
            # Continue with your specific derivation
            
            # This is a demonstration placeholder - replace with your actual derivation
            t1_sol = sp.atan2(self.fk_expressions[y], self.fk_expressions[x])
            
            # Print the derived solution
            print("Derived t1 solution:", t1_sol)
            
            # Return the derived equations
            # This would be a dictionary of solutions for each joint angle
            return {"t1": t1_sol, "t2": None, "t3": None, "t4": None, "t5": None, "t6": None}
            
        except Exception as e:
            print(f"Analytical derivation failed: {str(e)}")
            return None
        
    def calc_DH(self, ee_Tfs, joint_vals):
        eqns=[]
        for  i in range(len(ee_Tfs)):
            tfc=ee_Tfs[i]
            jv=joint_vals[i]
            sk=self.T_test.subs({self.theta_symbols: jv.reshape(6,)})
            count=0
            for eq in sk:
                row_no=count//4
                column_no=count%4
                eqn=sp.Eq(eq,tfc[row_no,column_no])
                eqns.append(eqn)
                count+=1
            
        soln=sp.solve(tuple(eqns),[*self.A,*self.Alpha,*self.D],dict=True)
        return soln

    def lambdify(self):
        return sp.lambdify([self.theta_symbols],self.function,modules="numpy")
    
    def lambdafy_gen(self):
        return sp.lambdify([[self.theta_symbols],[self.A],[self.Alpha],[self.D]],self.T_test,modules="numpy")

    def lambdify_jacobian(self):
        return sp.lambdify([self.theta_symbols], self.jacobian_matrix, modules="numpy")

    def Transform(self,joint_angles):
        if len(joint_angles) != len(self.theta_symbols):
            return None
        return np.matrix(self.lambdaf([*joint_angles]),dtype=float)

    def calculate_jacobian(self):
        position = self.function[0:3, 3]
        rotation = self.function[0:3,0:3]
        rpy = [sp.atan2(rotation[2, 1], rotation[2, 2]),sp.asin(-rotation[2, 0]),sp.atan2(rotation[1, 0], rotation[0, 0])]
        n = len(self.theta_symbols)
        J = sp.zeros(6, n)
        for i in range(n):
            J[0, i] = sp.diff(position[0], self.theta_symbols[i])
            J[1, i] = sp.diff(position[1], self.theta_symbols[i])
            J[2, i] = sp.diff(position[2], self.theta_symbols[i])
            J[3, i] = sp.diff(rpy[0]     , self.theta_symbols[i])
            J[4, i] = sp.diff(rpy[1]     , self.theta_symbols[i])
            J[5, i] = sp.diff(rpy[2]     , self.theta_symbols[i])
        return J
    
    def Jacobian(self, joint_angles=None):
        if joint_angles is None:
            return lambda x: self.Jacobian(x)
        
        if len(joint_angles) != len(self.theta_symbols):
            return None
        
        return np.array(self.jacobian_func([*joint_angles]), dtype=float)

if __name__=='__main__':
    dh_params = np.array([
    [0.0,  np.pi/2, 0.3, 0],
    [0.0, -np.pi/2, 0.0, 0],
    [0.0,  np.pi/2, 0.3, 0],
    [0.0, -np.pi/2, 0.0, 0],
    [0.0,  np.pi/2, 0.25,0],
    [0.0, -np.pi/2, 0.0, 0],
    [0.0,  0.0,     0.1, 0]
], dtype=float)
    trans=Transforms(dh_params)
    for i in range(3):
        fn = trans.Transform(np.array([
    0.0,           # theta_1
    -np.pi / 4,    # theta_2
    np.pi / 2,     # theta_3
    -np.pi / 6,    # theta_4
    np.pi / 3,     # theta_5
    -np.pi / 2,    # theta_6
    np.pi / 6      # theta_7
]))
        jac= trans.Jacobian(np.array([
    0.0,           # theta_1
    -np.pi / 4,    # theta_2
    np.pi / 2,     # theta_3
    -np.pi / 6,    # theta_4
    np.pi / 3,     # theta_5
    -np.pi / 2,    # theta_6
    np.pi / 6      # theta_7
]))
        print(jac)   
