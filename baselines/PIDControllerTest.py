import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.irrlicht as chronoirr
import numpy as np
import os
import random
import csv

from verti_bench.envs.utils.utils import SetChronoDataDirectories
import pychrono.sensor as sens

class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.previous_error = 0.0
        self.integral = 0.0

    def compute(self, current_value, dt):
        error = self.setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output

def initialize_vw_pos(m_vehicle, m_terrain, m_terrain_length, m_terrain_width, seed=None):
    theta = (random.random() - 0.5) * 1.0 * np.pi
    x, y = m_terrain_length * 0.5 * np.cos(theta) + 2, m_terrain_width * 0.5 * np.sin(theta)
    z = m_terrain.GetHeight(chrono.ChVectorD(x, y, 0)) + 3
    ang = np.pi + theta
    m_initLoc = chrono.ChVectorD(x, y, z)
    m_initRot = chrono.Q_from_AngZ(ang)
    m_vehicle.SetInitPosition(chrono.ChCoordsysD(m_initLoc, m_initRot))
    return m_initLoc, m_initRot, theta

def set_goal(art_theta, current_terrain_stage, m_terrain, m_system, m_terrain_length, m_terrain_width, m_initLoc, seed):
    delta_theta = (random.random() - 0.5) * 1.0 * np.pi
    gx, gy = m_terrain_length * 0.5 * np.cos(art_theta + np.pi + delta_theta), m_terrain_width * 0.5 * np.sin(art_theta + np.pi + delta_theta)
    m_goal = chrono.ChVectorD(gx, gy, m_terrain.GetHeight(chrono.ChVectorD(gx, gy, 0)) + 2.5)

    i = 0
    while (m_goal - m_initLoc).Length() < 20:
        gx = random.random() * m_terrain_length - m_terrain_length / 2
        gy = random.random() * m_terrain_width - m_terrain_width / 2
        m_goal = chrono.ChVectorD(gx, gy, m_terrain.GetHeight(chrono.ChVectorD(gx, gy, 0)) + 3)
        if i > 100:
            print('Failed setting goal randomly, using default')
            gx = m_terrain_length * 0.625 * np.cos(art_theta + np.pi + delta_theta)
            gy = m_terrain_width * 0.625 * np.sin(art_theta + np.pi + delta_theta)
            break
        i += 1

    goal_contact_material = chrono.ChMaterialSurfaceNSC()
    goal_mat = chrono.ChVisualMaterial()
    goal_mat.SetAmbientColor(chrono.ChColor(1., 0., 0.))
    goal_mat.SetDiffuseColor(chrono.ChColor(1., 0., 0.))

    goal_body = chrono.ChBodyEasySphere(0.35, 1000, True, False, goal_contact_material)
    goal_body.SetPos(m_goal)
    goal_body.SetBodyFixed(True)
    goal_body.GetVisualShape(0).SetMaterial(0, goal_mat)

    m_system.Add(goal_body)

    return m_goal

def run_simulation(render=False):
    SetChronoDataDirectories()
    
    m_system = chrono.ChSystemNSC()
    m_system.Set_G_acc(chrono.ChVectorD(0, 0, -9.81))
    m_system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    
    m_terrain = veh.RigidTerrain(m_system)
    patch_mat = chrono.ChMaterialSurfaceNSC()
    patch_mat.SetFriction(0.9)
    patch_mat.SetRestitution(0.01)
    
    rotation_quaternion = chrono.ChQuaternionD()
    rotation_quaternion.Q_from_AngAxis(chrono.CH_C_PI, chrono.ChVectorD(0, 0, 1))
    patch = m_terrain.AddPatch(patch_mat, chrono.ChCoordsysD(chrono.ChVectorD(0, 0, 0), rotation_quaternion), 
                                terrain_path, m_terrain_length*1.5, m_terrain_width*1.5, 
                                m_min_terrain_height, m_max_terrain_height)
    half_length = m_terrain_length * 1.5 // 2
    half_width = m_terrain_width * 1.5 // 2

    top_patch_trans = chrono.ChCoordsysD(chrono.ChVectorD(0, half_width, 0), chrono.Q_ROTATE_Y_TO_Z)
    top_patch = m_terrain.AddPatch(patch_mat, top_patch_trans, m_terrain_length * 1.5, m_terrain_width / 3, 0.5)
    bottom_patch_trans = chrono.ChCoordsysD(chrono.ChVectorD(0, -half_width, 0), chrono.Q_ROTATE_Y_TO_Z)
    bottom_patch = m_terrain.AddPatch(patch_mat, bottom_patch_trans, m_terrain_length * 1.5, m_terrain_width / 3, 0.5)
    left_patch_trans = chrono.ChCoordsysD(chrono.ChVectorD(half_length, 0, 0), chrono.Q_ROTATE_X_TO_Z)
    left_patch = m_terrain.AddPatch(patch_mat, left_patch_trans, m_terrain_length / 3, m_terrain_width * 1.5, 0.5)
    right_patch_trans = chrono.ChCoordsysD(chrono.ChVectorD(-half_length, 0, 0), chrono.Q_ROTATE_X_TO_Z)
    right_patch = m_terrain.AddPatch(patch_mat, right_patch_trans, m_terrain_length / 3, m_terrain_width * 1.5, 0.5)
    
    patch.SetTexture(veh.GetDataFile(texture_file), m_terrain_length*1.5, m_terrain_width*1.5)
    top_patch.SetTexture(veh.GetDataFile(texture_file), m_terrain_length*1.5, m_terrain_width*1.5)
    bottom_patch.SetTexture(veh.GetDataFile(texture_file), m_terrain_length*1.5, m_terrain_width*1.5)
    left_patch.SetTexture(veh.GetDataFile(texture_file), m_terrain_length*1.5, m_terrain_width*1.5)
    right_patch.SetTexture(veh.GetDataFile(texture_file), m_terrain_length*1.5, m_terrain_width*1.5)
    
    m_terrain.Initialize()
    
    m_vehicle = veh.HMMWV_Reduced(m_system)
    m_vehicle.SetContactMethod(chrono.ChContactMethod_NSC)
    m_vehicle.SetChassisCollisionType(veh.CollisionType_PRIMITIVES)
    m_vehicle.SetChassisFixed(False)
    m_vehicle.SetEngineType(veh.EngineModelType_SIMPLE_MAP)
    m_vehicle.SetTransmissionType(veh.TransmissionModelType_AUTOMATIC_SIMPLE_MAP)
    m_vehicle.SetDriveType(veh.DrivelineTypeWV_AWD)
    m_vehicle.SetTireType(veh.TireModelType_RIGID)
    m_vehicle.SetTireStepSize(m_step_size)
    m_vehicle.SetInitFwdVel(0.0)
    
    m_initLoc, m_initRot, art_theta = initialize_vw_pos(m_vehicle, m_terrain, m_terrain_length, m_terrain_width, seed=None)
    m_vehicle.Initialize()

    m_vehicle.LockAxleDifferential(0, True)    
    m_vehicle.LockAxleDifferential(1, True)
    m_vehicle.LockCentralDifferential(0, True)
    m_vehicle.LockCentralDifferential(1, True)
    m_vehicle.GetVehicle().EnableRealtime(False)
    
    m_vehicle.SetChassisVisualizationType(veh.VisualizationType_PRIMITIVES)
    m_vehicle.SetWheelVisualizationType(veh.VisualizationType_MESH)
    m_vehicle.SetTireVisualizationType(veh.VisualizationType_MESH)
    m_vehicle.SetSuspensionVisualizationType(veh.VisualizationType_MESH)
    m_vehicle.SetSteeringVisualizationType(veh.VisualizationType_MESH)
    
    m_chassis_body = m_vehicle.GetChassisBody()
    
    # Set the driver
    m_driver = veh.ChDriver(m_vehicle.GetVehicle())
    m_driver_inputs = m_driver.GetInputs()

    # Set PID controller for speed
    m_speedController = veh.ChSpeedController()
    m_speedController.Reset(m_vehicle.GetRefFrame())
    m_speedController.SetGains(1, 0, 0)
    
    # Set goal and get m_goal
    m_goal = set_goal(art_theta, terrain_stage, m_terrain, m_system, m_terrain_length, m_terrain_width, m_initLoc, seed=None)
    # Initialize the custom PID controller for steering
    m_steeringController = PIDController(kp=0.3, ki=0.0, kd=0)
    
    if render:
        vis = veh.ChWheeledVehicleVisualSystemIrrlicht()
        vis.SetWindowTitle('vws in the wild')
        vis.SetWindowSize(3840, 2160)
        trackPoint = chrono.ChVectorD(-3, 0.0, 2)
        vis.SetChaseCamera(trackPoint, 3, 1)
        vis.Initialize()
        vis.AddLightDirectional()
        vis.AddSkyBox()
        vis.AttachVehicle(m_vehicle.GetVehicle())
    
    # Continuous speed
    speed = float(4)
    start_time = m_system.GetChTime()
    
    roll_angles = []
    pitch_angles = []
    
    while True:
        time = m_system.GetChTime()
        
        if render:
            vis.BeginScene()
            vis.Render()
            vis.EndScene()

        m_vehicle_pos = m_vehicle.GetVehicle().GetPos()
        m_vector_to_goal_noNoise = m_goal - m_vehicle_pos
        m_vector_to_goal = m_goal - m_vehicle_pos
        vector_to_goal_local = m_chassis_body.GetRot().RotateBack(m_vector_to_goal)
        target_heading_local = np.arctan2(vector_to_goal_local.y, vector_to_goal_local.x)
        heading_diff_local = (target_heading_local - 0 + np.pi) % (2 * np.pi) - np.pi
        
        euler_angles = m_vehicle.GetVehicle().GetRot().Q_to_Euler123()
        roll = euler_angles.x
        pitch = euler_angles.y
        roll_angles.append(np.degrees(abs(roll)))
        pitch_angles.append(np.degrees(abs(pitch)))
        
        # Update the custom PID controller
        steering = -m_steeringController.compute(heading_diff_local, m_step_size)
        
        if m_vector_to_goal_noNoise.Length() < 4:
            print('--------------------------------------------------------------')
            print('Goal Reached')
            print(f'Initial position: {m_initLoc}')
            print(f'Goal position: {m_goal}')
            print('--------------------------------------------------------------')
            if render:
                vis.Quit()
            return time - start_time, True, np.mean(roll_angles), np.mean(pitch_angles)
        
        if m_system.GetChTime() > m_max_time:
            print('--------------------------------------------------------------')
            print('Time out')
            print('Initial position: ', m_initLoc)
            dist = m_vector_to_goal_noNoise.Length()
            print('Final position of art: ', m_chassis_body.GetPos())
            print('Goal position: ', m_goal)
            print('Distance to goal: ', dist)
            print('--------------------------------------------------------------')
            if render:
                vis.Quit()
            return time - start_time, False, np.mean(roll_angles), np.mean(pitch_angles)
        
        # Desired throttle/braking value
        out_throttle = m_speedController.Advance(m_vehicle.GetRefFrame(), speed, time, m_step_size)
        out_throttle = np.clip(out_throttle, -1, 1)
        if out_throttle >= 0:
            m_driver_inputs.m_braking = 0
            m_driver_inputs.m_throttle = out_throttle
        else:
            m_driver_inputs.m_braking = -out_throttle
            m_driver_inputs.m_throttle = 0

        # smooth the driver inputs
        m_driver_inputs.m_steering = np.clip(steering, -1.0, 1.0)
        
        for _ in range(0, m_steps_per_control):
            time = m_system.GetChTime()
            # Update modules (process inputs from other modules)
            m_terrain.Synchronize(time)
            m_vehicle.Synchronize(time, m_driver_inputs, m_terrain)
            if render:
                vis.Synchronize(time, m_driver_inputs)

            # Advance simulation for one timestep for all modules
            m_driver.Advance(m_step_size)
            m_terrain.Advance(m_step_size)
            m_vehicle.Advance(m_step_size)
            if render:
                vis.Advance(m_step_size)
            
            m_system.DoStepDynamics(m_step_size)
    
    return None, False  # Return None if goal not reached

def run_simulation_on_terrain(terrain_stage, num_trials=50, render=False):
    global terrain_path
    terrain_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "../envs/data/terrain_bitmaps/Automatic-CL/TestLevels", terrain_file[terrain_stage])
    
    successful_runs = 0
    failures = 0
    
    traversal_times = []
    roll_angles = []
    pitch_angles = []

    for i in range(num_trials):
        print(f'Terrain {terrain_stage + 1}, Run {i + 1}')
        time_to_goal, success, avg_roll, avg_pitch = run_simulation(render=render)
        if success:
            successful_runs += 1
            traversal_times.append(time_to_goal)
        else:
            failures += 1

        roll_angles.append(avg_roll)
        pitch_angles.append(avg_pitch)

    success_rate = successful_runs / num_trials
    average_time_to_goal = np.mean(traversal_times) if traversal_times else None
    var_time_to_goal = np.var(traversal_times) if traversal_times else None
    average_roll = np.mean(roll_angles)
    var_roll = np.var(roll_angles)
    average_pitch = np.mean(pitch_angles)
    var_pitch = np.var(pitch_angles)

    print('--------------------------------------------------------------')
    print(f'Terrain {terrain_stage + 1} - Number of successful trials: {successful_runs} out of {num_trials}')
    print(f'Success rate: {success_rate * 100:.2f}%')
    print(f'Failures: {failures}')
    if average_time_to_goal is not None:
        print(f'Mean traversal time: {average_time_to_goal:.2f} seconds (Variance: {var_time_to_goal:.2f})')
    print(f'Average roll angle: {average_roll:.4f} degrees (Variance: {var_roll:.4f})')
    print(f'Average pitch angle: {average_pitch:.4f} degrees (Variance: {var_pitch:.4f})')
    print('--------------------------------------------------------------')

    return success_rate, average_roll, var_roll, average_pitch, var_pitch, average_time_to_goal, var_time_to_goal


def evaluate_overall_success_rate(num_runs=3, num_trials=50):
    overall_success_rates = []
    os.makedirs('./TestPID', exist_ok=True)
    for run_index in range(1, num_runs + 1):
        print(f'Run {run_index} / {num_runs}')
        terrain_success_rates = []
        
        output_csv = f'./TestPID/run{run_index}.csv'
        try:
            with open(output_csv, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Terrain', 'Success Rate', 'Average Roll Angle', 'Variance Roll Angle', 
                                 'Average Pitch Angle', 'Variance Pitch Angle', 
                                 'Average Traversal Time', 'Variance Traversal Time'])
                print(f"Header row written to {output_csv}")
        except Exception as e:
            print(f"Error writing header to {output_csv}: {e}")
            continue
        
        terrain_stage = 4
        print(f"Starting simulation for terrain stage {terrain_stage + 1}")
        success_rate, avg_roll, var_roll, avg_pitch, var_pitch, avg_time, var_time = run_simulation_on_terrain(
            terrain_stage, num_trials=num_trials, render=False)
        terrain_success_rates.append(success_rate)
        
        # Append the results to the CSV file
        with open(output_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([terrain_stage + 1, success_rate, avg_roll, var_roll, avg_pitch, var_pitch, avg_time, var_time])

        overall_success_rate = np.mean(terrain_success_rates)
        overall_success_rates.append(overall_success_rate)
        
        print(f'Overall success rate for run {run_index}: {overall_success_rate * 100:.2f}%\n')

    print('--------------------------------------------------------------')
    print('Final Results After All Runs:')
    for i, overall_success_rate in enumerate(overall_success_rates):
        print(f'Run {i + 1} - Overall Success Rate: {overall_success_rate * 100:.2f}%')
    print('--------------------------------------------------------------')
    
    return overall_success_rates

# Parameters
terrain_stage = 4  # For level_5.bmp                                                                                                                                                                                                                     
m_max_time = 20
texture_file_options = ["terrain/textures/grass.jpg", "terrain/textures/dirt.jpg", 
                        "terrain/textures/Gravel034_1K-JPG/Gravel034_1K_Color.jpg", 
                        "terrain/textures/concrete.jpg"]
texture_file = texture_file_options[-1]
terrain_file = ["level_1.bmp", "level_2.bmp", "level_3.bmp", "level_4.bmp", "level_5.bmp"]

m_min_terrain_height = 0  # min terrain height
m_max_terrain_height = 7  # max terrain height
m_terrain_length = 21  # size in X direction
m_terrain_width = 21  # size in Y direction

# Simulation step sizes
m_step_size = 5e-3 # seconds per step
m_control_freq = 10 # control updates per second
m_steps_per_control = round(1 / (m_step_size * m_control_freq))

# Execute the evaluation
evaluate_overall_success_rate(num_runs=3, num_trials=50)
