import time
import numpy as np
import math

useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 11 #8
pandaNumDofs = 7

ll = [-7]*pandaNumDofs
#upper limits for null space
ul = [7]*pandaNumDofs
#joint ranges for null space
jr = [7]*pandaNumDofs
#rest poses for null space
jointPositions=[0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]


class PandaEnv(object):
  def __init__(self, bullet_client, offset, fps):
    self.bullet_client = bullet_client
    self.bullet_client.setPhysicsEngineParameter(solverResidualThreshold=0)    
    flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES

    self.state_t = 0
    self.states = [0, 3, 5, 4, 6, 3, 7]
    self.cur_state = 0
    self.state_durations =[3, 1, 1, 1, 1, 1, 1]
  
    orn=[-0.707107, 0.0, 0.0, 0.707107]
    self.panda = self.bullet_client.loadURDF("franka_panda/panda.urdf", np.array([0,0,0])+self.offset, orn, useFixedBase=True, flags=flags)
    
    self.offset = np.array(offset)
    self.bullet_client.loadURDF("tray/traybox.urdf", [0+offset[0], 0+offset[1], -0.6+offset[2]], [-0.5, -0.5, -0.5, 0.5], flags=flags)
    self.bullet_client.changeVisualShape(self.legos[0],-1,rgbaColor=[1,0,0,1])

    self.legos=[]
    self.legos.append(self.bullet_client.loadURDF("lego/lego.urdf",np.array([0.1, 0.3, -0.5])+self.offset, flags=flags))
    self.legos.append(self.bullet_client.loadURDF("lego/lego.urdf",np.array([-0.1, 0.3, -0.5])+self.offset, flags=flags))
    self.legos.append(self.bullet_client.loadURDF("lego/lego.urdf",np.array([0.1, 0.3, -0.7])+self.offset, flags=flags))
    self.sphereId = self.bullet_client.loadURDF("sphere_small.urdf",np.array( [0, 0.3, -0.6])+self.offset, flags=flags)
    
    index = 0
    self.state = 0
    self.fps = fps
    self.control_dt = 1./fps
    self.finger_target = 0
    self.gripper_height = 0.2

    #create a constraint to keep the fingers centered
    c = self.bullet_client.createConstraint(self.panda,
                       9,
                       self.panda,
                       10,
                       jointType=self.bullet_client.JOINT_GEAR,
                       jointAxis=[1, 0, 0],
                       parentFramePosition=[0, 0, 0],
                       childFramePosition=[0, 0, 0])
    self.bullet_client.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)
 
    for j in range(self.bullet_client.getNumJoints(self.panda)):
      self.bullet_client.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
      info = self.bullet_client.getJointInfo(self.panda, j)
      jointType = info[2]
      if (jointType == self.bullet_client.JOINT_PRISMATIC):
        self.bullet_client.resetJointState(self.panda, j, jointPositions[index]) 
        index=index+1

      if (jointType == self.bullet_client.JOINT_REVOLUTE):
        self.bullet_client.resetJointState(self.panda, j, jointPositions[index]) 
        index=index+1
    self.t = 0.


  def reset(self):
    pass


  def update_state(self):
    self.state_t += self.control_dt
    if self.state_t > self.state_durations[self.cur_state]:
      self.cur_state += 1
      if self.cur_state >= len(self.states):
        self.cur_state = 0
      self.state_t = 0
      self.state=self.states[self.cur_state]


  def step(self, graspWidth):
    # 设置抓取器张开宽度
    if self.state==6:
      self.finger_target = 0.01
    if self.state==5:
      self.finger_target = 0.04 
    
    self.update_state()
  
    alpha = 0.9 #0.99
    if self.state==1 or self.state==2 or self.state==3 or self.state==4 or self.state==7:

      self.gripper_height = alpha * self.gripper_height + (1.-alpha)*0.03

      if self.state == 2 or self.state == 3 or self.state == 7:
        self.gripper_height = alpha * self.gripper_height + (1.-alpha)*0.2
      
      t = self.t
      self.t += self.control_dt
      
      pos = [self.offset[0]+0.2 * math.sin(1.5 * t), self.offset[1]+self.gripper_height, self.offset[2]+-0.6 + 0.1 * math.cos(1.5 * t)] # 圆形位置

      if self.state == 3 or self.state== 4:
        # 获取红色积木的位置和方向
        pos, o = self.bullet_client.getBasePositionAndOrientation(self.legos[0])    #sphereId self.legos[0]
        pos = [pos[0], self.gripper_height, pos[2]] # 机械手位置
        self.prev_pos = pos

      if self.state == 7:
        pos = self.prev_pos
        diffX = pos[0] - self.offset[0]
        diffZ = pos[2] - (self.offset[2]-0.6)
        self.prev_pos = [self.prev_pos[0] - diffX*0.1, self.prev_pos[1], self.prev_pos[2]-diffZ*0.1]

      orn = self.bullet_client.getQuaternionFromEuler([math.pi/2.,0.,0.])   # 机械手方向
      
      # 根据目标位置计算关节位置
      jointPoses = self.bullet_client.calculateInverseKinematics(self.panda, pandaEndEffectorIndex, pos, orn, ll, ul, jr, jointPositions, maxNumIterations=20)

      for i in range(pandaNumDofs):
        self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL, jointPoses[i],force=5 * 240.)

    #target for fingers
    for i in [9,10]:
      self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,self.finger_target ,force= 10)
