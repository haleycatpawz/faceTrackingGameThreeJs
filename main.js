import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

import {
    FaceLandmarker, FilesetResolver, DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

// --- DOM elements and state variables ---
const demosSection = document.getElementById("demos");
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);
const threejsCanvas = document.getElementById("threejs_canvas");

let faceLandmarker;
let runningMode = "VIDEO";
let enableWebcamButton;
let webcamRunning = false;
let lastVideoTime = -1;

const videoHeight = "360px";
const videoWidth = 480;

// --- THREE.JS SETUP ---
const renderer = new THREE.WebGLRenderer({
    canvas: threejsCanvas,
    alpha: true,
    antialias: true
});
renderer.setSize(window.innerWidth, window.innerHeight);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
    45,
    window.innerWidth / window.innerHeight,
    0.1,
    1000
);
const orbit = new OrbitControls(camera, renderer.domElement);
camera.position.set(0, 0, 10);
orbit.update();

// Add a cube to the scene for 3D tracking
const cubeGeometry = new THREE.BoxGeometry(2, 2, 2); // Adjusted size for better visibility
const cubeMaterial = new THREE.MeshNormalMaterial({ wireframe: false });
const trackedCube = new THREE.Mesh(cubeGeometry, cubeMaterial);
scene.add(trackedCube);

// --- MEDIAPIPE FACE LANDMARKER SETUP ---
async function createFaceLandmarker() {
    const filesetResolver = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );
    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
            delegate: "GPU"
        },
        outputFaceBlendshapes: true,
        runningMode,
        numFaces: 1
    });
    demosSection.classList.remove("invisible");
}
createFaceLandmarker();

// --- WEBCAM SETUP ---
function hasGetUserMedia() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

if (hasGetUserMedia()) {
    enableWebcamButton = document.getElementById("webcamButton");
    enableWebcamButton.addEventListener("click", enableCam);
} else {
    console.warn("getUserMedia() is not supported by your browser");
}

function enableCam() {
    if (!faceLandmarker) {
        console.log("Wait! FaceLandmarker not loaded yet.");
        return;
    }

    if (webcamRunning === true) {
        webcamRunning = false;
        enableWebcamButton.innerText = "ENABLE PREDICTIONS";
    } else {
        webcamRunning = true;
        enableWebcamButton.innerText = "DISABLE PREDICTIONS";
    }

    const constraints = { video: true };
    navigator.mediaDevices.getUserMedia(constraints)
        .then((stream) => {
            video.srcObject = stream;
            video.addEventListener("loadeddata", predictWebcam);
        })
        .catch((error) => {
            webcamRunning = false;
            enableWebcamButton.innerText = "ENABLE PREDICTIONS";
            let errorMessage = "Could not access the webcam.";

            if (error.name === 'NotReadableError' || error.name === 'OverconstrainedError') {
                errorMessage = "Webcam is in use by another application. Please close other apps and try again.";
            } else if (error.name === 'NotAllowedError') {
                errorMessage = "Webcam permission denied. Please enable webcam access in your browser settings.";
            }

            console.error(errorMessage);
            alert(errorMessage);
        });
}

// --- MAIN PREDICTION AND RENDERING LOOP ---
const historyLength = 10;
const facePositionHistory = [];

// --- STATE VARIABLES FOR SMOOTHING ---
let lastKnownLandmarks = null;
const SMOOTHING_FACTOR = 0.7; // Lower value for more smoothing, higher for less

async function predictWebcam() {
    canvasElement.style.height = videoHeight;
    video.style.height = videoHeight;
    canvasElement.style.width = videoWidth;
    video.style.width = videoWidth;
    threejsCanvas.style.height = videoHeight;
    threejsCanvas.style.width = videoWidth;

    if (runningMode === "IMAGE") {
        runningMode = "VIDEO";
        await faceLandmarker.setOptions({ runningMode: "VIDEO" });
    }

    if (!webcamRunning || !faceLandmarker) {
        return;
    }

    let startTimeMs = performance.now();
    let results = null;

    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        results = faceLandmarker.detectForVideo(video, startTimeMs);
    }
    
    // Clear both canvases for fresh render
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // Check if new landmarks were detected
    if (results && results.faceLandmarks && results.faceLandmarks.length > 0) {
        // If this is the first detection, or tracking is lost, just use the new data
        if (!lastKnownLandmarks || lastKnownLandmarks.length !== results.faceLandmarks.length) {
            lastKnownLandmarks = results.faceLandmarks;
        } else {
            // Apply smoothing to each landmark for a smoother transition
            lastKnownLandmarks = results.faceLandmarks.map((newLandmarks, faceIndex) => {
                return newLandmarks.map((landmark, landmarkIndex) => {
                    const lastLandmark = lastKnownLandmarks[faceIndex][landmarkIndex];
                    return {
                        x: lastLandmark.x * SMOOTHING_FACTOR + landmark.x * (1 - SMOOTHING_FACTOR),
                        y: lastLandmark.y * SMOOTHING_FACTOR + landmark.y * (1 - SMOOTHING_FACTOR),
                        z: lastLandmark.z * SMOOTHING_FACTOR + landmark.z * (1 - SMOOTHING_FACTOR)
                    };
                });
            });
        }
    }
    
    // Always draw using the last known, smoothed landmarks
    if (lastKnownLandmarks) {
        for (const landmarks of lastKnownLandmarks) {
            // Draw face outline
            drawingUtils.drawConnectors(
                landmarks,
                FaceLandmarker.FACE_LANDMARKS_TESSELATION,
                { color: "#00000070", lineWidth: .05 }
            );
            // Draw eyes, eyebrows, and lips
           // drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: "rgba(153, 97, 225, .5)" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, { color: "rgba(48, 227, 255, .2)" });
           // drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: "rgba(153, 97, 225, .5)"});
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, { color: "rgba(46, 227, 255, .2)" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: "rgba(255, 128, 128, .2)" });

            // Update Three.js cube using the smoothed landmarks
            const noseTipLandmark = landmarks[1];
            if (noseTipLandmark) {
                // A more robust way to map normalized coordinates to 3D space
                const aspectRatio = video.videoWidth / video.videoHeight;
                const fieldOfView = 45;
                const tanFov = Math.tan(THREE.MathUtils.degToRad(fieldOfView / 2));
                
                const x = (noseTipLandmark.x - 0.5) * 2 * aspectRatio * tanFov;
                const y = (0.5 - noseTipLandmark.y) * 2 * tanFov;
                const z = -noseTipLandmark.z; // Use negative Z for depth
                
                trackedCube.position.set(-(x * 10), y * 10, z * 10);
            }
        }
    }

    canvasCtx.restore();

    // Render the Three.js scene
    renderer.render(scene, camera);

    if (webcamRunning === true) {
        window.requestAnimationFrame(predictWebcam);
    }
}

window.addEventListener('resize', function() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});