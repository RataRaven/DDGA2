import random 
import time 
from time import gmtime, strftime 
from numpy import array 
import numpy 
import pybullet as p 
import pybullet_data 
import scipy.ndimage
from scipy.signal import butter, lfilter
import math
import numpy as np
import sys, os
from statistics import mean
from time import sleep
from collections import deque
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from scoop import futures
import matplotlib.pyplot as plt
# Set up global parameters
SLEEP_TIME= 0.002
N_NUM_ACTUATORS = 1
N_NUM = 120 # Number Of Torques
SEARCH_SPACE = [[-10, 0]]
# Sample rate or control frequency
TimeSlotSteps = 30 # The Minimum TIme Slot For Each Actuation signal
torqueList0 = []
torqueList1 = []
torqueList2 = []
hoppingHeightList = []
targetListx = []
pitchAngleList = []
yawAngleList = []
connected = False
startTime = time.time()
# Reward list is the accumulated reward. Step reward is temporary reward.
rewardList = []
stepRewardList = []
# Robot motion control
def sendTorqueControl(robotId, joint, torque, recordEnabled):
    if recordEnabled:
        if joint == 0:
            torqueList0.append(torque)
        if joint == 2:
            torqueList2.append(torque)
    p.setJointMotorControl2(
                bodyUniqueId=robotId,
                jointIndex=joint,
                controlMode=p.TORQUE_CONTROL,
                force=torque)


def controlRobot(parameters, robotId, planeId, recordEnabled):
    sendTorqueControl(robotId, 2, parameters[0], recordEnabled)


def convertLogTorqueToLinearTorque(x):
    return np.sign(x)*(430 / (1 + math.exp(-abs(x) + 5)) - 2)


# Bullet physics related functions
# Contain update GUI, robot control and step simulation
# Auto camera tracking
def stepRobotSimulation(parameters,robotId,planeId, recordEnabled):
    p.stepSimulation()
    controlRobot(parameters, robotId, planeId, recordEnabled)
    cameraPos = p.getBasePositionAndOrientation(robotId)[0]
    p.resetDebugVisualizerCamera(1.5, 0.6, 0, cameraPos)
    if recordEnabled:
        hoppingHeightList.append(p.getBasePositionAndOrientation(robotId)[0][2])
        targetListx.append(p.getBasePositionAndOrientation(robotId)[0][0])
        pitchAngleList.append(p.getBasePositionAndOrientation(robotId)[1][1])
        yawAngleList.append(p.getBasePositionAndOrientation(robotId)[1][0])
    

def evaluate(parameters, realtime, guiEnabled, recordEnabled):
    # Clean up old torque lists and angle records
    global torqueList0
    global toruqeList2
    global hoppingHeightList
    global pitchAngleList
    global rewardList
    global connected
    global stepRewardList
    if recordEnabled:
        torqueList0 = []
        torqueList2 = []
        hoppingHeightList = []
        pitchAngleList = []
        rewardList = []

    if guiEnabled:
        p.disconnect()
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        if (len(sys.argv) > 1) and (str(sys.argv[1]) == '-r'):
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.ER_TINY_RENDERER, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1)
            p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, fileName=(str(os.getcwd()) + '/recordedVideos/' + 
                                                            str(strftime("%Y-%m-%d%H:%M:%S", gmtime())) +
                                                            '.mp4'))
    else:
        if connected:
            p.resetSimulation()
        else:
            p.connect(p.DIRECT)
            connected = True
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    p.setGravity(0, 0, -10)
    planeId = p.loadURDF("plane.urdf")
    robotStartPos = [0, 0, 2.2]
    robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
    robotId = p.loadURDF("3d_hop_only.urdf", robotStartPos, robotStartOrientation, useFixedBase=0)
    robotPos, robotOrn = p.getBasePositionAndOrientation(robotId)
    p.setJointMotorControlArray(
    robotId,
    [0, 1, 2],
    p.POSITION_CONTROL,
    targetPositions=[0.0, 0, 0],
    forces=[0, 0, 0]
    )
    n = 0
    energyCost = sum(map(abs, parameters))
    parameters = np.repeat(parameters, TimeSlotSteps)
    torqueList = list(map(convertLogTorqueToLinearTorque, parameters))
    smoothedTorqueList = scipy.ndimage.filters.gaussian_filter1d(torqueList, 2)
    totalTorqueList = list(smoothedTorqueList)
    t0 = totalTorqueList
    totalCost = 0
    for i in range(len(t0)):
                stepRobotSimulation([t0[i]], robotId, planeId, recordEnabled)
                # speedCost += sum(map(abs, list(p.getBaseVelocity(robotId)[1])))
                positionCost = p.getBasePositionAndOrientation(robotId)[0][0]
                # postureCost = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(robotId)[1])[1]
                postureCost = abs(p.getBasePositionAndOrientation(robotId)[1][1])
                # postureCosty += abs(p.getBasePositionAndOrientation(robotId)[1][1])
                heightCost = p.getBasePositionAndOrientation(robotId)[0][2]
                # positionCost += abs(p.getBasePositionAndOrientation(robotId)[0][0])
                torquePenulaty = abs(t0[i])
                contactPoint = 0
                contact = p.getContactPoints(bodyA=robotId, linkIndexA=-1)
                if len(contact) > 0:
                    contactPoint = 1
                tempList = 100*heightCost - 0.5*torquePenulaty
                totalCost += tempList
                if recordEnabled:
                    stepRewardList.append(tempList)
                    rewardList.append(totalCost)
    if recordEnabled:
        p.stopStateLogging(0)
    return  totalCost/len(t0),


# Genetic algorithm functions
# Initialize chromosome
def split(a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def init1d(icls, searchSpace):
    dim = array(searchSpace).shape[0]
    subSize = int(N_NUM / N_NUM_ACTUATORS)
    actList = []
    for k in range(dim):
        for i in range(subSize):
            actList.append(random.uniform(searchSpace[k][0], searchSpace[k][1]))
    return icls(actList)


def evalIndividual(individual):
    return evaluate(individual, False, False, False)

# Mutation
def mutateActList(individual, indpb, searchSpace):
    size = len(individual)
    partSize = int(size/N_NUM_ACTUATORS)
    for i in range(size):
        if random.random() < indpb:
            partIndex = int(i / partSize)
            individual[i] = random.uniform(searchSpace[partIndex][0], searchSpace[partIndex][1])
    return individual,


# Crossover For Torquelists
def crossOver(ind1, ind2):
    child1, child2 = [toolbox.clone(ind) for ind in (ind1, ind2)]
    del child1.fitness.values
    del child2.fitness.values
    l1 = list(split(child1, N_NUM_ACTUATORS))
    l2 = list(split(child2, N_NUM_ACTUATORS))
    crossedTotalList = []
    new1 = toolbox.individual()
    new2 = toolbox.individual()
    del new1[:]
    del new2[:]
    for i in range(N_NUM_ACTUATORS):
        crossedTotalList.append(tools.cxOnePoint(l1[i], l2[i]))
    for sublist in crossedTotalList:
        new1.extend(sublist[0])
        new2.extend(sublist[1])
    return new1, new2,


# Set up DEAP
creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.Fitness)
toolbox = base.Toolbox()
toolbox.register("individual", init1d, creator.Individual, SEARCH_SPACE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evalIndividual)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("map", futures.map)
toolbox.register("mate", crossOver)
toolbox.register("mutate", mutateActList, indpb=0.02, searchSpace=SEARCH_SPACE)


def main():
    random.seed(None)
    pop = toolbox.population(n=20)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.1, ngen=40,
                                stats=stats, halloffame=hof, verbose=True)
    bests = tools.selBest(hof, k=1)
    print('Training time', (time.time() - startTime))
    print(bests[0])
    print(bests[0].fitness)
    evaluate(bests[0],False,False,True)
    p1,ax1=plt.subplot(411)
    plt.plot(torqueList2)
    p1.set_ylabel('Thrust torque')
    p2=plt.subplot(412)
    plt.plot(hoppingHeightList)
    p2.set_ylabel('Height')
    p3=plt.subplot(413)
    p3.set_ylabel('Accumulated Reward')
    plt.plot(rewardList)
    p4=plt.subplot(414)
    plt.plot(stepRewardList)
    p4.set_ylabel('Reward')
    plt.show()
    return pop, log, hof


if __name__ == "__main__":
    main()

