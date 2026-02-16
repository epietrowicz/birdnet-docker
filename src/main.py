from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from datetime import datetime

# Load and initialize the BirdNET-Analyzer models.
analyzer = Analyzer()

recording = Recording(
    analyzer,
    "/app/samples/sample.mp3",
    lat=35.6,
    lon=-77.3,
    date=datetime(year=2023, month=6, day=27),  # use date or week_48
    min_conf=0.25,
)
recording.analyze()
print(recording.detections)