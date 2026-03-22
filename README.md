# SmartRoad AI — Web App

## How it works
- Your PC runs Flask + YOLO (all heavy processing stays on laptop)
- Phone opens the web app in its browser
- Phone camera streams frames to the server via HTTP
- YOLO detects damage on the server
- Annotated result streams back to phone display
- Real GPS from the phone browser is attached to every saved image
- Dashboard shows map, history, and saved images

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Put your model in the same folder
```
smartroad/
  app.py
  yolo12s_RDD2022_best.pt   ← your model here
  templates/
  outputs/
```

### 3. Run the server
```bash
python app.py
```

### 4. Find your PC's local IP
- Windows: `ipconfig` → look for IPv4 Address (e.g. 192.168.1.5)
- Mac/Linux: `ifconfig` → look for inet (e.g. 192.168.1.5)

### 5. Open on phone
- Make sure phone and PC are on the same WiFi
- Open browser on phone: `http://192.168.1.5:5000`
- Allow camera + location permissions when prompted
- Tap START SCAN

## Notes
- GPS accuracy depends on phone hardware. Outdoors = best accuracy.
- Torch (flashlight) only works on Android Chrome, not iOS Safari.
- For best detection: hold phone steady, good lighting, camera facing road.
- Saved images go to: `outputs/event_001/`, `outputs/event_002/`, etc.
- Dashboard auto-refreshes every 3 seconds.

## Folder structure after detection
```
outputs/
  event_001/
    frame_01_20260322_162348_123.jpg
    frame_02_...
    (up to 5 per event)
  event_002/
    ...
```
