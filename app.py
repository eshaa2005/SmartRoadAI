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
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "smartroad_secret_2024"

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = "yolo12s_RDD2022_best.pt"
OUTPUT_DIR    = "outputs"
STAGING_DIR   = "staging"
LOG_FILE      = os.path.join(OUTPUT_DIR, "detections_log.json")
USERS_FILE    = os.path.join(BASE_DIR, "users.json")
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
DEFAULT_USERS = {
    "user1": "pass123",
    "user2": "pass456",
}


def _default_user_store():
    store = {
        ADMIN_USERNAME: {
            "password_hash": generate_password_hash(ADMIN_PASSWORD),
            "role": "admin",
            "created_at": datetime.now().isoformat(),
        }
    }
    for username, password in DEFAULT_USERS.items():
        store[username] = {
            "password_hash": generate_password_hash(password),
            "role": "user",
            "created_at": datetime.now().isoformat(),
        }
    return store


def save_users_to_disk(users):
    tmp_file = USERS_FILE + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)
    os.replace(tmp_file, USERS_FILE)


def load_users_from_disk():
    changed = False
    users = {}

    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                users = raw
        except Exception:
            users = {}

    if not users:
        users = _default_user_store()
        save_users_to_disk(users)
        return users

    normalized = {}
    for username, record in users.items():
        if isinstance(record, str):
            normalized[username] = {
                "password_hash": generate_password_hash(record),
                "role": "user",
                "created_at": datetime.now().isoformat(),
            }
            changed = True
            continue

        if not isinstance(record, dict):
            continue

        password_hash = record.get("password_hash")
        if not password_hash and record.get("password"):
            password_hash = generate_password_hash(str(record.get("password", "")))
            changed = True

        role = record.get("role") or ("admin" if username == ADMIN_USERNAME else "user")
        created_at = record.get("created_at") or datetime.now().isoformat()

        normalized[username] = {
            "password_hash": password_hash or "",
            "role": role,
            "created_at": created_at,
        }

    if ADMIN_USERNAME not in normalized or not check_password_hash(normalized[ADMIN_USERNAME].get("password_hash", ""), ADMIN_PASSWORD):
        normalized[ADMIN_USERNAME] = {
            "password_hash": generate_password_hash(ADMIN_PASSWORD),
            "role": "admin",
            "created_at": datetime.now().isoformat(),
        }
        changed = True

    for username, password in DEFAULT_USERS.items():
        if username not in normalized:
            normalized[username] = {
                "password_hash": generate_password_hash(password),
                "role": "user",
                "created_at": datetime.now().isoformat(),
            }
            changed = True

    if changed:
        save_users_to_disk(normalized)

    return normalized

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
        detections_log = []
        for d in data:
            if not isinstance(d, dict):
                continue
            row = dict(d)
            row.setdefault("status", "open")
            row.setdefault("resolved_at", "")
            row.setdefault("resolved_by", "")
            row.setdefault("resolution_note", "")
            detections_log.append(row)
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
    users = load_users_from_disk()

    record = users.get(username)
    if record and check_password_hash(record.get("password_hash", ""), password):
        session["username"] = username
        session["role"]     = record.get("role", "user")
        return jsonify({"role": session["role"]})
    return jsonify({"error": "Invalid username or password"}), 401


@app.route("/api/register", methods=["POST"])
def do_register():
    data     = request.json or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    confirm  = data.get("confirm_password", "").strip()

    if not username or not password or not confirm:
        return jsonify({"error": "Please fill in all fields"}), 400
    if username == ADMIN_USERNAME:
        return jsonify({"error": "That username is reserved"}), 400
    if len(username) < 3 or len(username) > 30:
        return jsonify({"error": "Username must be 3 to 30 characters"}), 400
    if not all(ch.isalnum() or ch in "._-" for ch in username):
        return jsonify({"error": "Username can only use letters, numbers, dot, underscore, and dash"}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400
    if password != confirm:
        return jsonify({"error": "Passwords do not match"}), 400

    with lock:
        users = load_users_from_disk()
        if username in users:
            return jsonify({"error": "Username already exists"}), 409

        users[username] = {
            "password_hash": generate_password_hash(password),
            "role": "user",
            "created_at": datetime.now().isoformat(),
        }
        save_users_to_disk(users)

    session["username"] = username
    session["role"] = "user"
    return jsonify({"role": "user", "username": username})

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
                "status":         "open",
                "resolved_at":    "",
                "resolved_by":    "",
                "resolution_note": "",
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

        data.sort(key=lambda d: (d.get("status") == "resolved", d.get("timestamp", "")), reverse=False)
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
        resolved = len([d for d in data if d.get("status") == "resolved"])
        open_issues = len(data) - resolved
    return jsonify({
        "total_events":     events,
        "total_detections": len(data),
        "open_issues":      open_issues,
        "resolved_issues":  resolved,
        "model":            MODEL_PATH,
        "role":             role,
        "username":         user,
    })

@app.route("/api/resolve/<int:detection_id>", methods=["POST"])
@login_required
@admin_required
def resolve_detection(detection_id):
    payload = request.json or {}
    note = (payload.get("note") or "").strip()
    now = datetime.now().isoformat()
    resolved = 0

    with lock:
        target = next((d for d in detections_log if d["id"] == detection_id), None)
        if not target:
            return jsonify({"error": "Not found"}), 404

        event_id = target.get("event")
        for row in detections_log:
            if row.get("event") == event_id:
                row["status"] = "resolved"
                row["resolved_at"] = now
                row["resolved_by"] = session.get("username", "admin")
                if note:
                    row["resolution_note"] = note
                resolved += 1

        save_log_to_disk()

    return jsonify({"ok": True, "event": event_id, "resolved": resolved})

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
              "boxes","damage_summary","damage_types","note","image","status",
              "resolved_at","resolved_by","resolution_note"]
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
        load_users_from_disk()

    port   = int(os.environ.get("PORT", 5000))
    lan_ip = get_lan_ip()
    print("\n" + "="*50)
    print("  SmartRoad AI — Server starting")
    print(f"  Admin : {ADMIN_USERNAME} / {ADMIN_PASSWORD}")
    print("  Users : default users are available, and new users can register")
    print(f"  Local : http://127.0.0.1:{port}")
    print(f"  LAN   : http://{lan_ip}:{port}")
    print("="*50 + "\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
