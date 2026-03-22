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
import logging
import socket
from datetime import datetime
from functools import wraps

app = Flask(__name__)
app.secret_key = "smartroad_secret_2024"

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH    = "yolo12s_RDD2022_best.pt"
OUTPUT_DIR    = "outputs"
STAGING_DIR   = "staging"
LOG_FILE      = os.path.join(OUTPUT_DIR, "detections_log.json")
MAX_PER_EVENT = 3
CONF_THRESH   = 0.25
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STAGING_DIR, exist_ok=True)

# ── Damage class labels ───────────────────────────────────────────────────────
CLASS_LABELS = {
    "D00":    "Longitudinal Crack",
    "D10":    "Transverse Crack",
    "D20":    "Alligator Crack",
    "D40":    "Pothole",
    "Repair": "Repaired Area",
}

# ── Credentials ───────────────────────────────────────────────────────────────
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"
USERS = {
    "user1": "pass123",
    "user2": "pass456",
}

# ── Geocode ───────────────────────────────────────────────────────────────────
def geocode_location(lat, lon):
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


# ── Model ─────────────────────────────────────────────────────────────────────
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("Model loaded!")

# ── Global state ──────────────────────────────────────────────────────────────
detections_log = []
event_counter  = 0
lock           = threading.Lock()
active_events  = {}


def next_event_id_locked():
    global event_counter
    in_use_active = {int(ev.get("event_id", 0) or 0) for ev in active_events.values()}
    candidate = max(int(event_counter or 0), max(in_use_active, default=0)) + 1
    while True:
        out_dir = os.path.join(OUTPUT_DIR, f"event_{candidate:03d}")
        if candidate not in in_use_active and not os.path.exists(out_dir):
            event_counter = candidate
            return candidate
        candidate += 1


def save_log_to_disk():
    try:
        tmp_file = LOG_FILE + ".tmp"
        # Convert sets to lists before serialising
        serialisable = []
        for d in detections_log:
            row = dict(d)
            if isinstance(row.get("damage_types"), set):
                row["damage_types"] = list(row["damage_types"])
            serialisable.append(row)
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, ensure_ascii=False, indent=2)
        os.replace(tmp_file, LOG_FILE)
    except Exception as e:
        print(f"Warning: could not save log file: {e}")


def load_log_from_disk():
    global detections_log, event_counter
    if not os.path.exists(LOG_FILE):
        return
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print("Warning: log file format invalid; expected list")
            return
        detections_log = [d for d in data if isinstance(d, dict)]
        max_event = 0
        for d in detections_log:
            try:
                max_event = max(max_event, int(d.get("event", 0) or 0))
            except Exception:
                pass
        max_dir_event = 0
        try:
            for name in os.listdir(OUTPUT_DIR):
                if name.startswith("event_") and name[6:].isdigit():
                    max_dir_event = max(max_dir_event, int(name[6:]))
        except Exception:
            pass
        event_counter = max(max_event, max_dir_event)
        print(f"Loaded {len(detections_log)} report entries from {LOG_FILE}")
    except Exception as e:
        print(f"Warning: could not load log file: {e}")


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

    if (not city or city.strip().lower() in ("", "unknown")) and lat != "N/A":
        city = geocode_location(lat, lon)

    results   = model(frame, conf=CONF_THRESH, verbose=False)
    num_boxes = len(results[0].boxes)
    annotated = results[0].plot()
    now       = datetime.now()

    # ── Extract damage class names ────────────────────────────────────────────
    damage_classes = []
    class_counts   = {}
    if num_boxes > 0:
        for box in results[0].boxes:
            cls_id   = int(box.cls[0])
            cls_name = model.names.get(cls_id, f"class_{cls_id}")
            damage_classes.append(cls_name)
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    damage_summary = ", ".join(
        f"{CLASS_LABELS.get(k, k)} x{v}" for k, v in class_counts.items()
    ) if class_counts else ""

    frame_num   = 0
    max_reached = False

    with lock:
        ev = active_events.get(username)

        if num_boxes > 0:
            if ev is None:
                new_event_id = next_event_id_locked()
                stage_dir    = os.path.join(STAGING_DIR, f"{username}_ev{new_event_id}")
                os.makedirs(stage_dir, exist_ok=True)
                ev = {
                    "event_id":     new_event_id,
                    "stage_dir":    stage_dir,
                    "frames":       [],
                    "lat":          lat,
                    "lon":          lon,
                    "city":         city,
                    "started":      now.isoformat(),
                    "damage_types": set(),
                }
                active_events[username] = ev

            # Accumulate all damage types seen across all frames
            ev["damage_types"].update(damage_classes)

            if len(ev["frames"]) < MAX_PER_EVENT:
                ts    = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
                fname = f"frame_{len(ev['frames'])+1:02d}_{ts}.jpg"
                fpath = os.path.join(ev["stage_dir"], fname)

                save_frame = annotated.copy()
                lines = [
                    f"Event #{ev['event_id']}  Frame {len(ev['frames'])+1}/{MAX_PER_EVENT}",
                    f"User     : {username}",
                    f"Damage   : {damage_summary}",
                    f"Location : {city}",
                    f"GPS      : {lat}, {lon}",
                    f"Time     : {now.strftime('%Y-%m-%d %H:%M:%S')}",
                ]
                for i, line in enumerate(lines):
                    cv2.putText(save_frame, line, (10, 25 + i*22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1, cv2.LINE_AA)
                cv2.imwrite(fpath, save_frame)

                rel = os.path.join(f"{username}_ev{ev['event_id']}", fname).replace("\\", "/")
                ev["frames"].append({
                    "file":           fname,
                    "rel":            rel,
                    "boxes":          num_boxes,
                    "damage_summary": damage_summary,
                    "class_counts":   class_counts,
                })
                frame_num = len(ev["frames"])

            max_reached = len(ev["frames"]) >= MAX_PER_EVENT

    stage_frames = []
    with lock:
        ev2 = active_events.get(username)
        if ev2:
            stage_frames = [f["rel"] for f in ev2["frames"]]

    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
    annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    return jsonify({
        "detections":     num_boxes,
        "annotated":      annotated_b64,
        "has_event":      active_events.get(username) is not None,
        "frame_num":      frame_num,
        "max_reached":    max_reached,
        "stage_frames":   stage_frames,
        "damage_summary": damage_summary,
        "class_counts":   class_counts,
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
        "active":       True,
        "event_id":     ev["event_id"],
        "frame_count":  len(ev["frames"]),
        "max_reached":  len(ev["frames"]) >= MAX_PER_EVENT,
        "city":         ev["city"],
        "stage_frames": [f["rel"] for f in ev["frames"]],
    })


@app.route("/api/event/submit", methods=["POST"])
@login_required
def submit_event():
    username = session["username"]
    note     = (request.json or {}).get("note", "").strip()

    with lock:
        ev = active_events.get(username)
        if not ev or not ev["frames"]:
            return jsonify({"error": "No active event to submit"}), 400

        event_id = ev["event_id"]
        out_dir  = os.path.join(OUTPUT_DIR, f"event_{event_id:03d}")
        os.makedirs(out_dir, exist_ok=True)

        next_id = max((int(d.get("id", 0) or 0) for d in detections_log), default=0) + 1

        # Build readable damage type list for this whole event
        damage_types  = sorted(ev.get("damage_types", set()))
        damage_labels = [CLASS_LABELS.get(d, d) for d in damage_types]

        for idx, f in enumerate(ev["frames"], start=1):
            src = os.path.join(ev["stage_dir"], f["file"])
            dst = os.path.join(out_dir, f["file"])
            if os.path.exists(src):
                shutil.move(src, dst)

            rel_out = os.path.join(f"event_{event_id:03d}", f["file"]).replace("\\", "/")
            detections_log.append({
                "id":             next_id,
                "event":          event_id,
                "frame":          idx,
                "timestamp":      ev["started"],
                "lat":            ev["lat"],
                "lon":            ev["lon"],
                "city":           ev["city"],
                "boxes":          f["boxes"],
                "image":          rel_out,
                "username":       username,
                "note":           note,
                "damage_summary": f.get("damage_summary", ""),
                "damage_types":   damage_labels,
                "class_counts":   f.get("class_counts", {}),
            })
            next_id += 1

        save_log_to_disk()

        try:
            shutil.rmtree(ev["stage_dir"])
        except Exception:
            pass

        del active_events[username]

    return jsonify({"ok": True, "event_id": event_id})


@app.route("/api/event/discard", methods=["POST"])
@login_required
def discard_event():
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
        save_log_to_disk()
    return jsonify({"ok": True})

@app.route("/api/export")
@login_required
@admin_required
def export_csv():
    with lock:
        data = list(detections_log)
    output = io.StringIO()
    fields = ["id","event","frame","timestamp","username","city","lat","lon",
              "boxes","damage_summary","damage_types","note","image"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in data:
        r = {k: row.get(k, "") for k in fields}
        # Flatten list fields for CSV
        if isinstance(r["damage_types"], list):
            r["damage_types"] = " | ".join(r["damage_types"])
        writer.writerow(r)
    response = Response(output.getvalue(), mimetype="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=smartroad_report.csv"
    return response


if __name__ == "__main__":
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    def get_lan_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    with lock:
        load_log_from_disk()

    port   = int(os.environ.get("PORT", 5000))
    lan_ip = get_lan_ip()
    print("\n" + "="*50)
    print("  SmartRoad AI — Server starting")
    print(f"  Admin : {ADMIN_USERNAME} / {ADMIN_PASSWORD}")
    print(f"  Users : user1/pass123  user2/pass456")
    print(f"  Local : http://127.0.0.1:{port}")
    print(f"  LAN   : http://{lan_ip}:{port}")
    print("="*50 + "\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
