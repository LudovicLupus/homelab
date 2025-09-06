#!/usr/bin/env python3
"""
ML Processor for Frigate Security System
-----------------------------------------
A custom machine learning pipeline for facial recognition and other
advanced computer vision tasks that integrates with Frigate NVR.

This service:
1. Connects to MQTT to receive events from Frigate
2. Processes image/video data for facial recognition
3. Stores face embeddings in a PostgreSQL database with pgvector
4. Provides real-time notifications for recognized individuals
"""

import json
import os
import queue
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any

import cv2
import numpy as np
import requests
import torch
from insightface.app import FaceAnalysis
from insightface.model_zoo import face_recognition
from loguru import logger
from paho.mqtt import client as mqtt_client
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

# ====================================================
# Configuration and Environment Setup
# ====================================================

# Environment variables with defaults
MQTT_HOST = os.environ.get('MQTT_HOST', 'mqtt')
MQTT_PORT = int(os.environ.get('MQTT_PORT', 1883))
FRIGATE_API_URL = os.environ.get('FRIGATE_API_URL', 'http://frigate:5000')
POSTGRES_URL = os.environ.get('POSTGRES_URL', 'postgresql://myuser:mypassword@postgres-vector:5432/mydatabase')

# Application paths
BASE_DIR = Path('/app')
MODEL_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data'
LOGS_DIR = BASE_DIR / 'logs'
FACES_DIR = DATA_DIR / 'faces'
UNKNOWN_FACES_DIR = DATA_DIR / 'unknown_faces'

# Ensure directories exist
for directory in [MODEL_DIR, DATA_DIR, LOGS_DIR, FACES_DIR, UNKNOWN_FACES_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Configure logging
logger.remove()
logger.add(sys.stderr, level='INFO')
logger.add(LOGS_DIR / 'ml_processor.log', rotation='10 MB', retention='7 days', level='DEBUG')

# Check CUDA availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device: {DEVICE}')
if DEVICE.type == 'cuda':
    logger.info(f'CUDA Device: {torch.cuda.get_device_name(0)}')
    logger.info(f'CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')


# ====================================================
# Database Models
# ====================================================

Base = declarative_base()


class Person(Base):
    """Represents a person who can be recognized by the system."""

    __tablename__ = 'persons'

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    face_embeddings = relationship('FaceEmbedding', back_populates='person', cascade='all, delete-orphan')
    appearances = relationship('Appearance', back_populates='person', cascade='all, delete-orphan')

    def __repr__(self):
        return f"<Person(id={self.id}, name='{self.name}')>"


class FaceEmbedding(Base):
    """Stores face embedding vectors for a person."""

    __tablename__ = 'face_embeddings'

    id = Column(Integer, primary_key=True)
    person_id = Column(Integer, ForeignKey('persons.id'), nullable=False)
    embedding = Column(Vector(512))  # Using pgvector for vector storage
    confidence = Column(Float, nullable=False)
    image_path = Column(String(255), nullable=True)
    model_version = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    person = relationship('Person', back_populates='face_embeddings')

    def __repr__(self):
        return f'<FaceEmbedding(id={self.id}, person_id={self.person_id})>'


class Appearance(Base):
    """Records when a person is detected in a camera feed."""

    __tablename__ = 'appearances'

    id = Column(Integer, primary_key=True)
    person_id = Column(Integer, ForeignKey('persons.id'), nullable=False)
    camera_name = Column(String(100), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    confidence = Column(Float, nullable=False)
    bbox = Column(ARRAY(Float), nullable=False)  # [x1, y1, x2, y2]
    snapshot_path = Column(String(255), nullable=True)

    # Metadata
    event_id = Column(String(100), nullable=True)
    event_data = Column(JSON, nullable=True)

    # Relationships
    person = relationship('Person', back_populates='appearances')

    def __repr__(self):
        return (
            f'<Appearance(id={self.id}, person_id={self.person_id} \n'
            f"camera='{self.camera_name}', timestamp={self.timestamp})>"
        )


# ====================================================
# Database Connection
# ====================================================


class Database:
    """Database connection and session management."""

    def __init__(self, url: str):
        self.engine = create_engine(url)
        self.Session = sessionmaker(bind=self.engine)
        self._setup_database()

    def _setup_database(self):
        """Create tables if they don't exist."""
        Base.metadata.create_all(self.engine)

    def get_session(self):
        """Get a new database session."""
        return self.Session()


# ====================================================
# Face Recognition System
# ====================================================


class FaceRecognitionSystem:
    """Handles face detection and recognition."""

    def __init__(self, model_dir: Path, db: Database, recognition_threshold: float = 0.6):
        self.model_dir = model_dir
        self.db = db
        self.recognition_threshold = recognition_threshold
        self.detection_queue = Queue(maxsize=100)
        self.recognition_queue = Queue(maxsize=100)
        self.processing_threads = []
        self.running = False

        # Initialize face detection model (InsightFace)
        logger.info('Loading face detection model...')
        self.face_detector = FaceAnalysis(
            name='buffalo_l',
            root=str(model_dir),
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        self.face_detector.prepare(ctx_id=0, det_size=(640, 640))

        # Initialize face recognition model (InsightFace ArcFace)
        logger.info('Loading face recognition model...')
        self.face_recognizer = face_recognition.FaceRecognition(
            model_file=str(model_dir / 'buffalo_l' / 'w600k_r50.onnx'),
            root=str(model_dir),
        )

        logger.info('Face recognition system initialized')

    def start(self):
        """Start the face recognition threads."""
        self.running = True

        # Start face detection thread
        detection_thread = threading.Thread(target=self._detection_worker, name='FaceDetectionThread')
        detection_thread.daemon = True
        detection_thread.start()
        self.processing_threads.append(detection_thread)

        # Start face recognition thread
        recognition_thread = threading.Thread(target=self._recognition_worker, name='FaceRecognitionThread')
        recognition_thread.daemon = True
        recognition_thread.start()
        self.processing_threads.append(recognition_thread)

        logger.info('Face recognition threads started')

    def stop(self):
        """Stop the face recognition threads."""
        self.running = False
        for thread in self.processing_threads:
            thread.join(timeout=2.0)
        logger.info('Face recognition threads stopped')

    def queue_image(self, image_data: dict[str, Any]):
        """Queue an image for face detection and recognition.

        Args:
            image_data: Dict containing image, source info, and metadata
        """
        try:
            self.detection_queue.put(image_data, block=False)
        except Exception as e:
            logger.warning(f'Failed to queue image: {str(e)}')

    def _detection_worker(self):
        """Worker thread that processes images for face detection."""
        while self.running:
            try:
                image_data = self.detection_queue.get(timeout=1.0)

                image = image_data.get('image')
                if image is None:
                    continue

                # Detect faces
                faces = self.face_detector.get(image)
                if not faces:
                    continue

                # For each detected face, queue for recognition
                for face in faces:
                    # Skip low-confidence detections
                    if face.det_score < 0.5:
                        continue

                    bbox = face.bbox.astype(int)
                    face_img = image[bbox[1] : bbox[3], bbox[0] : bbox[2]]

                    recognition_data = {
                        'face_img': face_img,
                        'bbox': bbox,
                        'det_score': face.det_score,
                        'source_image': image,
                        'metadata': image_data.get('metadata', {}),
                        'camera': image_data.get('camera', 'unknown'),
                        'timestamp': image_data.get('timestamp', datetime.utcnow()),
                        'event_id': image_data.get('event_id', None),
                    }

                    self.recognition_queue.put(recognition_data, block=False)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f'Error in detection worker: {str(e)}')

    def _recognition_worker(self):
        """Worker thread that processes detected faces for recognition."""
        while self.running:
            try:
                face_data = self.recognition_queue.get(timeout=1.0)

                face_img = face_data.get('face_img')
                if face_img is None or face_img.size == 0:
                    continue

                # Get face embedding
                embedding = self.face_recognizer.get(face_img)
                if embedding is None:
                    continue

                # Search for matching person in database
                session = self.db.get_session()
                try:
                    # Find closest matches using pgvector
                    result = (
                        session.query(FaceEmbedding, Person)
                        .join(Person)
                        .order_by(FaceEmbedding.embedding.l2_distance(embedding.tolist()))
                        .limit(1)
                        .first()
                    )

                    if result:
                        face_embedding, person = result
                        distance = np.linalg.norm(np.array(face_embedding.embedding) - embedding)

                        # If match is found and distance is below threshold
                        if distance < self.recognition_threshold:
                            # Record the appearance
                            appearance = Appearance(
                                person_id=person.id,
                                camera_name=face_data.get('camera', 'unknown'),
                                timestamp=face_data.get('timestamp', datetime.utcnow()),
                                confidence=1.0 - (distance / self.recognition_threshold),
                                bbox=face_data['bbox'].tolist(),
                                event_id=face_data.get('event_id'),
                                event_data=face_data.get('metadata'),
                            )
                            session.add(appearance)

                            # Save the snapshot
                            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                            snapshot_path = f'faces/{person.name}_{timestamp_str}.jpg'
                            full_path = DATA_DIR / snapshot_path
                            cv2.imwrite(str(full_path), face_data['face_img'])
                            appearance.snapshot_path = snapshot_path

                            session.commit()

                            logger.info(
                                f"Recognized {person.name} in {face_data.get('camera', 'unknown')} camera"
                            )
                        else:
                            # Unknown face - save for later processing
                            self._save_unknown_face(face_data, embedding)
                    else:
                        # No matching faces in database
                        self._save_unknown_face(face_data, embedding)

                except Exception as e:
                    logger.error(f'Database error in recognition worker: {str(e)}')
                    session.rollback()
                finally:
                    session.close()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f'Error in recognition worker: {str(e)}')

    def _save_unknown_face(self, face_data: dict[str, Any], embedding: np.ndarray):
        """Save unknown face for later enrollment."""
        try:
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            unknown_id = f'unknown_{timestamp_str}_{np.random.randint(10000)}'

            # Save face image
            unknown_path = UNKNOWN_FACES_DIR / f'{unknown_id}.jpg'
            cv2.imwrite(str(unknown_path), face_data['face_img'])

            # Save embedding and metadata
            metadata = {
                'camera': face_data.get('camera', 'unknown'),
                'timestamp': face_data.get('timestamp', datetime.utcnow()).isoformat(),
                'bbox': face_data['bbox'].tolist(),
                'det_score': float(face_data['det_score']),
                'event_id': face_data.get('event_id'),
            }

            # Save embedding as numpy file
            embedding_path = UNKNOWN_FACES_DIR / f'{unknown_id}.npy'
            np.save(str(embedding_path), embedding)

            # Save metadata as JSON
            metadata_path = UNKNOWN_FACES_DIR / f'{unknown_id}.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)

            logger.debug(f'Saved unknown face to {unknown_path}')

        except Exception as e:
            logger.error(f'Error saving unknown face: {str(e)}')

    def enroll_person(self, name: str, face_images: list[np.ndarray], description: str = None) -> int:
        """Enroll a new person with multiple face images.

        Args:
            name: Name of the person
            face_images: List of face images (numpy arrays)
            description: Optional description

        Returns:
            ID of the enrolled person
        """
        session = self.db.get_session()
        try:
            # Create a new person
            person = Person(name=name, description=description)
            session.add(person)
            session.flush()  # Get the person ID

            # Process each face image
            for img in face_images:
                # Detect faces
                faces = self.face_detector.get(img)
                if not faces:
                    continue

                # Use the face with highest confidence
                face = max(faces, key=lambda x: x.det_score)

                # Extract face region
                bbox = face.bbox.astype(int)
                face_img = img[bbox[1] : bbox[3], bbox[0] : bbox[2]]

                # Get embedding
                embedding = self.face_recognizer.get(face_img)
                if embedding is None:
                    continue

                # Save face image
                timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                img_path = f'faces/{name}_{timestamp_str}.jpg'
                full_path = DATA_DIR / img_path
                cv2.imwrite(str(full_path), face_img)

                # Store embedding
                face_embedding = FaceEmbedding(
                    person_id=person.id,
                    embedding=embedding.tolist(),
                    confidence=float(face.det_score),
                    image_path=img_path,
                    model_version='insightface_arcface_r50',
                )
                session.add(face_embedding)

            session.commit()
            return person.id

        except Exception as e:
            session.rollback()
            logger.error(f'Error enrolling person: {str(e)}')
            raise
        finally:
            session.close()


# ====================================================
# MQTT Client
# ====================================================


class MQTTClient:
    """MQTT client to receive events from Frigate."""

    def __init__(
        self,
        host: str,
        port: int,
        frigate_api_url: str,
        face_system: FaceRecognitionSystem,
    ):
        self.host = host
        self.port = port
        self.frigate_api_url = frigate_api_url
        self.face_system = face_system
        self.client = None
        self.connected = False
        self.client_id = f'ml-processor-{int(time.time())}'
        self.topics = [
            'frigate/events',
            'frigate/+/person/snapshot',
            'frigate/+/+/person/snapshot',
            'frigate/+/person/detect',
        ]

    def connect(self):
        """Connect to MQTT broker."""
        try:
            self.client = mqtt_client.Client(client_id=self.client_id)
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            self.client.on_disconnect = self._on_disconnect

            self.client.connect(self.host, self.port)
            self.client.loop_start()

            # Wait for connection to establish
            start_time = time.time()
            while not self.connected and time.time() - start_time < 10:
                time.sleep(0.1)

            if not self.connected:
                logger.error(f'Failed to connect to MQTT broker at {self.host}:{self.port}')
                return False

            return True

        except Exception as e:
            logger.error(f'MQTT connection error: {str(e)}')
            return False

    def disconnect(self):
        """Disconnect from MQTT broker."""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
        self.connected = False

    def _on_connect(self, client, userdata, flags, rc):
        """Callback when client connects to the broker."""
        if rc == 0:
            self.connected = True
            logger.info(f'Connected to MQTT broker at {self.host}:{self.port}')

            # Subscribe to topics
            for topic in self.topics:
                self.client.subscribe(topic)
                logger.debug(f'Subscribed to {topic}')
        else:
            logger.error(f'Failed to connect to MQTT broker, return code {rc}')

    def _on_disconnect(self, client, userdata, rc):
        """Callback when client disconnects from the broker."""
        self.connected = False
        logger.warning(f'Disconnected from MQTT broker with code {rc}')

    def _on_message(self, client, userdata, msg):
        """Callback when a message is received."""
        try:
            topic = msg.topic
            payload = msg.payload.decode()

            logger.debug(f'Received message on topic {topic}')

            # Handle different message types
            if 'snapshot' in topic:
                self._handle_snapshot(topic, payload)
            elif 'detect' in topic:
                self._handle_detection(topic, payload)
            elif topic == 'frigate/events':
                self._handle_event(payload)

        except Exception as e:
            logger.error(f'Error processing MQTT message: {str(e)}')

    def _handle_snapshot(self, topic: str, payload: str):
        """Process a snapshot message."""
        try:
            # Extract camera name from topic
            parts = topic.split('/')
            camera_name = parts[1]

            # Fetch image from Frigate API
            snapshot_url = f'{self.frigate_api_url}/{payload}'
            response = requests.get(snapshot_url, timeout=5)

            if response.status_code == 200:
                # Convert image
                img_array = np.frombuffer(response.content, np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if image is not None:
                    # Queue for face detection
                    self.face_system.queue_image(
                        {
                            'image': image,
                            'camera': camera_name,
                            'timestamp': datetime.utcnow(),
                            'source': 'snapshot',
                            'path': payload,
                        }
                    )

            else:
                logger.warning(f'Failed to fetch snapshot: {response.status_code}')

        except Exception as e:
            logger.error(f'Error processing snapshot: {str(e)}')

    def _handle_detection(self, topic: str, payload: str):
        """Process a detection message."""
        try:
            # Parse JSON payload
            data = json.loads(payload)

            # Extract camera name from topic
            parts = topic.split('/')
            camera_name = parts[1]

            # Check if this is a person detection with a snapshot
            if 'snapshot_time' in data:
                # Construct path to snapshot
                snapshot_path = f"frigate/{camera_name}/{data['id']}/snapshot.jpg"

                # Queue for processing
                # Note: The actual image will be fetched in the snapshot handler
                self._handle_snapshot(f'frigate/{camera_name}/snapshot', snapshot_path)

        except Exception as e:
            logger.error(f'Error processing detection: {str(e)}')

    def _handle_event(self, payload: str):
        """Process an event message."""
        try:
            # Parse JSON payload
            data = json.loads(payload)

            # Only process person events with snapshots
            if data.get('type') == 'new' and data.get('label') == 'person' and data.get('has_snapshot', False):
                # Extract camera name and event ID
                camera_name = data.get('camera')
                event_id = data.get('id')

                # Construct path to snapshot
                snapshot_path = f'events/{event_id}/snapshot.jpg'

                # Fetch the snapshot
                snapshot_url = f'{self.frigate_api_url}/{snapshot_path}'
                response = requests.get(snapshot_url, timeout=5)

                if response.status_code == 200:
                    # Convert image
                    img_array = np.frombuffer(response.content, np.uint8)
                    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                    if image is not None:
                        # Queue for face detection
                        self.face_system.queue_image(
                            {
                                'image': image,
                                'camera': camera_name,
                                'timestamp': datetime.fromisoformat(data.get('start_time')),
                                'source': 'event',
                                'event_id': event_id,
                                'metadata': data,
                            }
                        )
                else:
                    logger.warning(f'Failed to fetch event snapshot: {response.status_code}')

        except Exception as e:
            logger.error(f'Error processing event: {str(e)}')


# ====================================================
# Main Application
# ====================================================


class MLProcessor:
    """Main application class."""

    def __init__(self):
        self.running = False
        self.db = None
        self.face_system = None
        self.mqtt_client = None

    def setup(self):
        """Set up application components."""
        try:
            # Initialize database
            logger.info(f'Connecting to database: {POSTGRES_URL}')
            self.db = Database(POSTGRES_URL)

            # Initialize face recognition system
            self.face_system = FaceRecognitionSystem(MODEL_DIR, self.db)

            # Initialize MQTT client
            self.mqtt_client = MQTTClient(MQTT_HOST, MQTT_PORT, FRIGATE_API_URL, self.face_system)

            return True

        except Exception as e:
            logger.error(f'Setup error: {str(e)}')
            return False

    def start(self):
        """Start the application."""
        if self.running:
            return

        self.running = True

        # Start face recognition system
        self.face_system.start()

        # Connect to MQTT
        if not self.mqtt_client.connect():
            logger.error('Failed to connect to MQTT, exiting')
            self.stop()
            return

        logger.info('ML Processor started successfully')

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Main loop - keep running until stopped
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stop the application."""
        self.running = False

        # Disconnect MQTT
        if self.mqtt_client:
            self.mqtt_client.disconnect()

        # Stop face recognition system
        if self.face_system:
            self.face_system.stop()

        logger.info('ML Processor stopped')

    def _signal_handler(self, sig, frame):
        """Handle termination signals."""
        logger.info(f'Received signal {sig}, shutting down')
        self.stop()


# ====================================================
# Entry Point
# ====================================================

if __name__ == '__main__':
    logger.info('Starting ML Processor')

    app = MLProcessor()
    if app.setup():
        app.start()
    else:
        logger.error('Failed to set up ML Processor, exiting')
        sys.exit(1)
