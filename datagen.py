#!/usr/bin/env python3
import sys
import os
import yaml
import pprint
import time
import subprocess
import warnings
import json
import argparse

from typing import Optional, Union
from pathlib import Path

import numpy as np

import acts
import acts.examples
from acts.examples.odd import getOpenDataDetector
from acts.examples.reconstruction import *
from acts.examples.simulation import *

u = acts.UnitConstants

#########################
# Command line handling #
#########################

parser = argparse.ArgumentParser(description='Exa.TrkX data generation/reconstruction script')
parser.add_argument('config_dir', help="where config data are stored", type=str)
parser.add_argument('output_dir', help="where to store output data", type=str)
parser.add_argument('--emb', help="embedding dim for Exa.TrkX", type=int, default=8)
parser.add_argument('--with_ckf', action="store_true")
parser.add_argument('--with_truthtracking', action="store_true")
parser.add_argument('--profile_gpu', action="store_true")
args = vars(parser.parse_args())

outputDir = Path(args['output_dir'])
(outputDir/"train_all").mkdir(exist_ok=True, parents=True)

#baseDir = Path(os.path.dirname(__file__))

configDir = Path(args["config_dir"])
assert (configDir / "torchscript/embed.pt").exists()
assert (configDir / "torchscript/filter.pt").exists()
assert (configDir / "torchscript/gnn.pt").exists()

logger = acts.logging.getLogger("main")


###########################
# Load Open Data Detector #
###########################

oddDir = Path("/home/iwsatlas1/bhuth/acts/thirdparty/OpenDataDetector")

oddMaterialMap = oddDir / "data/odd-material-maps.root"
assert os.path.exists(oddMaterialMap)

oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)
detector, trackingGeometry, decorators = getOpenDataDetector(oddDir, mdecorator=oddMaterialDeco)

geoSelectionExaTrkX = configDir / "odd-geo-selection-whole-detector.json"
assert os.path.exists(geoSelectionExaTrkX)

geoSelectionSeeding = oddDir / "config/odd-material-mapping-config.json"
assert os.path.exists(geoSelectionSeeding)

digiConfigFile = configDir / "odd-digi-config.json"
assert os.path.exists(digiConfigFile)

#######################
# Start GPU profiling #
#######################

if args["profile_gpu"]:
    gpu_profiler_args = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,memory.total,memory.reserved,memory.free,memory.used",
        "--format=csv,nounits",
        "--loop-ms=10",
        "--filename={}".format(outputDir / "gpu_memory_profile.csv")
    ]

    gpu_profiler = subprocess.Popen(gpu_profiler_args)
else:
    gpu_profiler = None

#############################
# Prepare and run sequencer #
#############################

field = acts.ConstantBField(acts.Vector3(0, 0, 2 * u.T))

s = acts.examples.Sequencer(
    events=10,
    numThreads=1,
    outputDir=str(outputDir),
)

s.config.logLevel = acts.logging.INFO

rnd = acts.examples.RandomNumbers(seed=42)

s = addPythia8(
    s,
    rnd=rnd,
    outputDirCsv=str(outputDir/"train_all"),
    hardProcess=["HardQCD:all = on"],
    #hardProcess=["Top:qqbar2ttbar=on"],
)

particleSelection = ParticleSelectorConfig(
    rho=(0.0*u.mm, 2.0*u.mm),
    pt=(500*u.MeV, 20*u.GeV),
    absEta=(0.0, 3.0)
)

addFatras(
    s,
    trackingGeometry,
    field,
    rnd=rnd,
    preselectParticles=particleSelection,
)

logger.info("Digitization config: {}".format(digiConfigFile))
s = addDigitization(
    s,
    trackingGeometry,
    field,
    digiConfigFile=digiConfigFile,
    outputDirRoot=None,
    outputDirCsv=str(outputDir/"train_all"),
    rnd=rnd,
    logLevel=acts.logging.INFO,
)


s.addWriter(
    acts.examples.CsvSimHitWriter(
        level=acts.logging.INFO,
        inputSimHits="simhits",
        outputDir=str(outputDir/"train_all"),
        outputStem="truth",
    )
)

s.addWriter(
    acts.examples.CsvMeasurementWriter(
        level=acts.logging.INFO,
        inputMeasurements="measurements",
        inputClusters="clusters",
        inputSimHits="simhits",
        inputMeasurementSimHitsMap="measurement_simhits_map",
        outputDir=str(outputDir/"train_all"),
    )
)

s.addWriter(
    acts.examples.CsvTrackingGeometryWriter(
        level=acts.logging.INFO,
        trackingGeometry=trackingGeometry,
        outputDir=str(outputDir),
        writePerEvent=False,
    )
)

###########################
# ExaTrkX + Kalman-Fitter #
###########################
s.addAlgorithm(
    acts.examples.SpacePointMaker(
        level=acts.logging.INFO,
        inputSourceLinks="sourcelinks",
        inputMeasurements="measurements",
        outputSpacePoints="exatrkx_spacepoints",
        trackingGeometry=trackingGeometry,
        geometrySelection=acts.examples.readJsonGeometryList(
            str(geoSelectionExaTrkX)
        ),
    )
)

exaTrkXConfig = {
    "modelDir": str(configDir / "torchscript"),
    "spacepointFeatures": 3,
    "embeddingDim": 8,
    "rVal": 0.2,
    "knnVal": 500,
    "filterCut": 0.01,
    "n_chunks": 5,
    "edgeCut": 0.5,
    "embeddingDim": args["emb"],
}

print("Exa.TrkX Configuration")
pprint.pprint(exaTrkXConfig, indent=4)

s.addAlgorithm(
    acts.examples.TrackFindingAlgorithmExaTrkX(
        level=acts.logging.INFO,
        inputSpacePoints="exatrkx_spacepoints",
        outputProtoTracks="exatrkx_prototracks",
        trackFinderML=acts.examples.ExaTrkXTrackFindingTorch(**exaTrkXConfig),
        rScale = 1000.,
        phiScale = np.pi,
        zScale = 1000.,
    )
)

s.addWriter(
    acts.examples.TrackFinderPerformanceWriter(
        level=acts.logging.INFO,
        inputProtoTracks="exatrkx_prototracks",
        inputParticles="particles_initial",
        inputMeasurementParticlesMap="measurement_particles_map",
        filePath=str(outputDir / "track_finding_performance_exatrkx.root"),
    )
)

s.addAlgorithm(
    acts.examples.TrackParamsEstimationAlgorithm(
        level=acts.logging.FATAL,
        inputSpacePoints=["exatrkx_spacepoints"],
        inputProtoTracks="exatrkx_prototracks",
        inputSourceLinks="sourcelinks",
        outputProtoTracks="exatrkx_estimated_prototracks",
        outputTrackParameters="exatrkx_estimated_parameters",
        trackingGeometry=trackingGeometry,
        magneticField=field,
        initialVarInflation=[100.0]*6,
    )
)

kalmanOptions = {
    "multipleScattering": True,
    "energyLoss": True,
    "reverseFilteringMomThreshold": 0.0,
    "freeToBoundCorrection": acts.examples.FreeToBoundCorrection(False),
}

s.addAlgorithm(
    acts.examples.TrackFittingAlgorithm(
        level=acts.logging.FATAL,
        inputMeasurements="measurements",
        inputSourceLinks="sourcelinks",
        inputProtoTracks="exatrkx_estimated_prototracks",
        inputInitialTrackParameters="exatrkx_estimated_parameters",
        outputTrajectories="exatrkx_kalman_trajectories",
        directNavigation=False,
        pickTrack=-1,
        trackingGeometry=trackingGeometry,
        fit=acts.examples.makeKalmanFitterFunction(
            trackingGeometry, field, **kalmanOptions
        ),
    )
)

s.addWriter(
    acts.examples.RootTrajectorySummaryWriter(
        level=acts.logging.ERROR,
        inputTrajectories = "exatrkx_kalman_trajectories",
        inputParticles = "particles_initial",
        inputMeasurementParticlesMap = "measurement_particles_map",
        filePath = str(outputDir / "tracksummary_exatrkx_kalman.root"),
    )
)


#####################################################
# Truth tracking (Truth Track Findinig -> KF & CKF) #
#####################################################
if args["with_truthtracking"]:
    s.addAlgorithm(
        acts.examples.TruthTrackFinder(
            level=acts.logging.INFO,
            inputParticles="particles_initial",
            inputMeasurementParticlesMap="measurement_particles_map",
            outputProtoTracks="truth_tracking_prototracks",
        )
    )

    s.addAlgorithm(
        acts.examples.TrackParamsEstimationAlgorithm(
            level=acts.logging.FATAL,
            inputSpacePoints=["exatrkx_spacepoints"],
            inputProtoTracks="truth_tracking_prototracks",
            inputSourceLinks="sourcelinks",
            outputProtoTracks="truth_tracking_prototracks_selected",
            outputTrackParameters="truth_tracking_parameters",
            trackingGeometry=trackingGeometry,
            magneticField=field,
            initialVarInflation=[100.0]*6,
        )
    )

    kalmanOptions = {
        "multipleScattering": True,
        "energyLoss": True,
        "reverseFilteringMomThreshold": 0.0,
        "freeToBoundCorrection": acts.examples.FreeToBoundCorrection(False),
    }

    s.addAlgorithm(
        acts.examples.TrackFittingAlgorithm(
            level=acts.logging.FATAL,
            inputMeasurements="measurements",
            inputSourceLinks="sourcelinks",
            inputProtoTracks="truth_tracking_prototracks_selected",
            inputInitialTrackParameters="truth_tracking_parameters",
            outputTrajectories="truth_tracking_kalman_trajectories",
            directNavigation=False,
            pickTrack=-1,
            trackingGeometry=trackingGeometry,
            fit=acts.examples.makeKalmanFitterFunction(
                trackingGeometry, field, **kalmanOptions
            ),
        )
    )

    s.addWriter(
        acts.examples.RootTrajectorySummaryWriter(
            level=acts.logging.ERROR,
            inputTrajectories = "truth_tracking_kalman_trajectories",
            inputParticles = "particles_initial",
            inputMeasurementParticlesMap = "measurement_particles_map",
            filePath = str(outputDir / "tracksummary_truth_tracking_kalman.root"),
        )
    )

    s.addAlgorithm(
        acts.examples.TrackFindingAlgorithm(
            level=acts.logging.FATAL,
            measurementSelectorCfg=acts.MeasurementSelector.Config(
                [(acts.GeometryIdentifier(), ([], [15.0], [10]))]
            ),
            inputMeasurements="measurements",
            inputSourceLinks="sourcelinks",
            inputInitialTrackParameters="truth_tracking_parameters",
            outputTrajectories="truth_tracking_ckf_trajectories",
            findTracks=acts.examples.TrackFindingAlgorithm.makeTrackFinderFunction(
                trackingGeometry, field
            ),
        )
    )

    s.addWriter(
        acts.examples.RootTrajectorySummaryWriter(
            level=acts.logging.ERROR,
            inputTrajectories = "truth_tracking_ckf_trajectories",
            inputParticles = "particles_initial",
            inputMeasurementParticlesMap = "measurement_particles_map",
            filePath = str(outputDir / "tracksummary_truth_tracking_ckf.root"),
        )
    )

#########################
# Default seeding + CKF #
#########################
if args["with_ckf"]:
    seedFinderConfig = SeedFinderConfigArg(
        impactMax = 4.426123855748383,
        deltaR = (13.639924973033985, 50.0854850448914),
        sigmaScattering = 7.3401486140533985,
        radLengthPerSeed = 0.06311548593790932,
        maxSeedsPerSpM = 0,
        cotThetaMax = 16.541921673890172,
        #cotThetaMax=27.310 # eta = 4
    )

    s = addSeeding(
        s,
        trackingGeometry,
        field,
        seedFinderConfigArg=seedFinderConfig,
        geoSelectionConfigFile=geoSelectionSeeding,
        initialVarInflation=[100.0]*6,
        seedingAlgorithm=SeedingAlgorithm.Default,
        outputDirRoot=None, #str(outputDir / "seeding"),
    )

    s.addAlgorithm(
        acts.examples.TrackFindingAlgorithm(
            level=acts.logging.FATAL,
            measurementSelectorCfg=acts.MeasurementSelector.Config(
                [(acts.GeometryIdentifier(), ([], [15.0], [10]))]
            ),
            inputMeasurements="measurements",
            inputSourceLinks="sourcelinks",
            inputInitialTrackParameters="estimatedparameters",
            outputTrajectories="ckf_trajectories",
            findTracks=acts.examples.TrackFindingAlgorithm.makeTrackFinderFunction(
                trackingGeometry, field
            ),
        )
    )

    s.addAlgorithm(
        acts.examples.TrajectoriesToPrototracks(
            level=acts.logging.INFO,
            inputTrajectories="ckf_trajectories",
            outputPrototracks="ckf_prototracks",
        )
    )

    s.addWriter(
        acts.examples.TrackFinderPerformanceWriter(
            level=acts.logging.INFO,
            inputProtoTracks="ckf_prototracks",
            inputParticles="particles_initial",
            inputMeasurementParticlesMap="measurement_particles_map",
            filePath=str(outputDir / "track_finding_performance_ckf.root"),
        )
    )

    s.addWriter(
        acts.examples.RootTrajectorySummaryWriter(
            level=acts.logging.ERROR,
            inputTrajectories = "ckf_trajectories",
            inputParticles = "particles_initial",
            inputMeasurementParticlesMap = "measurement_particles_map",
            filePath = str(outputDir / "tracksummary_ckf.root"),
        )
    )

s.run()
del s

with open(outputDir / "args.json", 'w') as f:
    json.dump(args, f, indent=4)

with open(outputDir / "exatrkx_config.json", 'w') as f:
    json.dump(exaTrkXConfig, f, indent=4)

if gpu_profiler is not None:
    gpu_profiler.kill()
