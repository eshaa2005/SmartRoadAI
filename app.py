from flask import Flask, jsonify, request, render_template, send_from_directory, session, redirect, url_for, Response
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os
import csv
import io
import shutil
import threading
import json
from datetime import datetime
from functools import wraps

app = Flask(__name__)
app.secret_key = "smartroad_secret_2024"

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH    = "yolo12s_RDD2022_best.pt"
OUTPUT_DIR    = "outputs"
STAGING_DIR   = "staging"          # temp folder while event is in progress
MAX_PER_EVENT = 3                  # max 3 images per event
CONF_THRESH   = 0.25
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STAGING_DIR, exist_ok=True)

# ── Credentials ───────────────────────────────────────────────────────────────
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"
USERS = {
    "user1": "pass123",
    "user2": "pass456",
}

# ── Model ─────────────────────────────────────────────────────────────────────
def geocode_location(lat, lon):
    """Convert lat/lon to a readable location name using Nominatim."""
    try:
        import urllib.request
        url = (f"https://nominatim.openstreetmap.org/reverse"
               f"?lat={lat}&lon={lon}&format=json&zoom=18&addressdetails=1")
        req = urllib.request.Request(url, headers={"User-Agent": "SmartRoadAI/1.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            d = json.loads(resp.read())
        a = d.get("address", {})
        road = a.get("road") or a.get("pedestrian") or a.get("footway") or ""
        locality = (a.get("suburb") or a.get("neighbourhood") or a.get("quarter") or
                    a.get("city_district") or a.get("city") or a.get("town") or
                    a.get("village") or a.get("hamlet") or a.get("county") or "")
        if road and locality and road != locality:
            return f"{road}, {locality}"
        return locality or road or f"{lat}, {lon}"
    except Exception:
        return f"{lat}, {lon}"


print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("Model loaded!")

# ── Global state ──────────────────────────────────────────────────────────────
detections_log  = []          # submitted reports only
event_counter   = 0           # global event ID counter
lock            = threading.Lock()

# Per-user active event: { username: { event_id, frames[], lat, lon, city, started } }
active_events   = {}


# ── Auth ──────────────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get("role") != "admin":
            return jsonify({"error": "Admin only"}), 403
        return f(*args, **kwargs)
    return decorated


# ── Pages ─────────────────────────────────────────────────────────────────────
@app.route("/")
def root():
    return redirect(url_for("scan_page") if "username" in session else url_for("login_page"))

@app.route("/login")
def login_page():
    return render_template("login.html")

@app.route("/scan")
@login_required
def scan_page():
    return render_template("index.html")

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")

@app.route("/outputs/<path:filename>")
@login_required
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)

@app.route("/staging/<path:filename>")
@login_required
def serve_staging(filename):
    return send_from_directory(STAGING_DIR, filename)


# ── Auth API ──────────────────────────────────────────────────────────────────
@app.route("/api/login", methods=["POST"])
def do_login():
    data     = request.json or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        session["username"] = username
        session["role"]     = "admin"
        return jsonify({"role": "admin"})
    if username in USERS and USERS[username] == password:
        session["username"] = username
        session["role"]     = "user"
        return jsonify({"role": "user"})
    return jsonify({"error": "Invalid username or password"}), 401

@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True})

@app.route("/api/me")
def me():
    if "username" not in session:
        return jsonify({"error": "Not logged in"}), 401
    return jsonify({"username": session["username"], "role": session["role"]})


# ── Detection API ─────────────────────────────────────────────────────────────
@app.route("/api/detect", methods=["POST"])
@login_required
def detect():
    """
    Receives a frame. Runs YOLO.
    If damage found and user has no active event → starts one.
    Saves frames to STAGING (not OUTPUT) until user submits.
    Returns: detections count, annotated frame, event status.
    """
    global event_counter

    data = request.json or {}
    if "image" not in data:
        return jsonify({"error": "No image"}), 400

    img_bytes = base64.b64decode(data["image"].split(",")[-1])
    np_arr    = np.frombuffer(img_bytes, np.uint8)
    frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    lat      = data.get("lat",  "N/A")
    lon      = data.get("lon",  "N/A")
    city     = data.get("city", "")
    username = session["username"]

    # If client didn't resolve a name, try server-side geocode
    if (not city or city.strip().lower() in ("", "unknown")) and lat != "N/A":
        city = geocode_location(lat, lon)

    results   = model(frame, conf=CONF_THRESH, verbose=False)
    num_boxes = len(results[0].boxes)
    annotated = results[0].plot()
    now       = datetime.now()

    frame_num  = 0
    max_reached = False

    with lock:
        ev = active_events.get(username)

        if num_boxes > 0:
            # Start a new event if user has none
            if ev is None:
                event_counter += 1
                stage_dir = os.path.join(STAGING_DIR, f"{username}_ev{event_counter}")
                os.makedirs(stage_dir, exist_ok=True)
                ev = {
                    "event_id":  event_counter,
                    "stage_dir": stage_dir,
                    "frames":    [],
                    "lat":       lat,
                    "lon":       lon,
                    "city":      city,
                    "started":   now.isoformat(),
                }
                active_events[username] = ev

            # Save frame to staging (max 3)
            if len(ev["frames"]) < MAX_PER_EVENT:
                ts    = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
                fname = f"frame_{len(ev['frames'])+1:02d}_{ts}.jpg"
                fpath = os.path.join(ev["stage_dir"], fname)

                save_frame = annotated.copy()
                lines = [
                    f"Event #{ev['event_id']}  Frame {len(ev['frames'])+1}/{MAX_PER_EVENT}",
                    f"User     : {username}",
                    f"Location : {city}",
                    f"GPS      : {lat}, {lon}",
                    f"Time     : {now.strftime('%Y-%m-%d %H:%M:%S')}",
                ]
                for i, line in enumerate(lines):
                    cv2.putText(save_frame, line, (10, 25 + i*22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1, cv2.LINE_AA)
                cv2.imwrite(fpath, save_frame)

                rel = os.path.join(f"{username}_ev{ev['event_id']}", fname).replace("\\", "/")
                ev["frames"].append({"file": fname, "rel": rel, "boxes": num_boxes})
                frame_num = len(ev["frames"])

            max_reached = len(ev["frames"]) >= MAX_PER_EVENT

    # Build staging preview paths
    stage_frames = []
    with lock:
        ev2 = active_events.get(username)
        if ev2:
            stage_frames = [f["rel"] for f in ev2["frames"]]

    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
    annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    return jsonify({
        "detections":   num_boxes,
        "annotated":    annotated_b64,
        "has_event":    active_events.get(username) is not None,
        "frame_num":    frame_num,
        "max_reached":  max_reached,
        "stage_frames": stage_frames,
    })


@app.route("/api/event/status")
@login_required
def event_status():
    username = session["username"]
    with lock:
        ev = active_events.get(username)
    if not ev:
        return jsonify({"active": False})
    return jsonify({
        "active":      True,
        "event_id":    ev["event_id"],
        "frame_count": len(ev["frames"]),
        "max_reached": len(ev["frames"]) >= MAX_PER_EVENT,
        "city":        ev["city"],
        "stage_frames": [f["rel"] for f in ev["frames"]],
    })


@app.route("/api/event/submit", methods=["POST"])
@login_required
def submit_event():
    """
    User confirms the report. Moves frames from staging → outputs.
    Adds note to each log entry.
    """
    username = session["username"]
    note     = (request.json or {}).get("note", "").strip()

    with lock:
        ev = active_events.get(username)
        if not ev or not ev["frames"]:
            return jsonify({"error": "No active event to submit"}), 400

        # Move staging folder → outputs
        event_id  = ev["event_id"]
        out_dir   = os.path.join(OUTPUT_DIR, f"event_{event_id:03d}")
        os.makedirs(out_dir, exist_ok=True)

        for f in ev["frames"]:
            src = os.path.join(ev["stage_dir"], f["file"])
            dst = os.path.join(out_dir, f["file"])
            if os.path.exists(src):
                shutil.move(src, dst)

            rel_out = os.path.join(f"event_{event_id:03d}", f["file"]).replace("\\", "/")
            detections_log.append({
                "id":        len(detections_log) + 1,
                "event":     event_id,
                "frame":     len(detections_log) + 1,
                "timestamp": ev["started"],
                "lat":       ev["lat"],
                "lon":       ev["lon"],
                "city":      ev["city"],
                "boxes":     f["boxes"],
                "image":     rel_out,
                "username":  username,
                "note":      note,
            })

        # Clean up staging dir
        try:
            shutil.rmtree(ev["stage_dir"])
        except Exception:
            pass

        del active_events[username]

    return jsonify({"ok": True, "event_id": event_id})


@app.route("/api/event/discard", methods=["POST"])
@login_required
def discard_event():
    """User discards — delete staging files, reset."""
    username = session["username"]
    with lock:
        ev = active_events.get(username)
        if ev:
            try:
                shutil.rmtree(ev["stage_dir"])
            except Exception:
                pass
            del active_events[username]
    return jsonify({"ok": True})


# ── Data API ──────────────────────────────────────────────────────────────────
@app.route("/api/log")
@login_required
def get_log():
    with lock:
        role = session.get("role")
        user = session.get("username")
        data = list(reversed(detections_log)) if role == "admin" \
               else list(reversed([d for d in detections_log if d["username"] == user]))
    return jsonify(data)

@app.route("/api/stats")
@login_required
def get_stats():
    with lock:
        role = session.get("role")
        user = session.get("username")
        data = detections_log if role == "admin" \
               else [d for d in detections_log if d["username"] == user]
        events = len(set(d["event"] for d in data))
    return jsonify({
        "total_events":     events,
        "total_detections": len(data),
        "model":            MODEL_PATH,
        "role":             role,
        "username":         user,
    })

@app.route("/api/delete/<int:detection_id>", methods=["DELETE"])
@login_required
@admin_required
def delete_detection(detection_id):
    with lock:
        entry = next((d for d in detections_log if d["id"] == detection_id), None)
        if not entry:
            return jsonify({"error": "Not found"}), 404
        img_path = os.path.join(OUTPUT_DIR, entry["image"])
        if os.path.exists(img_path):
            os.remove(img_path)
        detections_log.remove(entry)
    return jsonify({"ok": True})

@app.route("/api/export")
@login_required
@admin_required
def export_csv():
    with lock:
        data = list(detections_log)
    output = io.StringIO()
    fields = ["id","event","frame","timestamp","username","city","lat","lon","boxes","note","image"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in data:
        writer.writerow({k: row.get(k, "") for k in fields})
    response = Response(output.getvalue(), mimetype="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=smartroad_report.csv"
    return response


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  SmartRoad AI — Server starting")
    print(f"  Admin : {ADMIN_USERNAME} / {ADMIN_PASSWORD}")
    print(f"  Users : user1/pass123  user2/pass456")
    print("  Open  : http://<YOUR_PC_IP>:5000")
    print("="*50 + "\n")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
