import ipaddress
import os
import socket
import time
import uuid
from collections import defaultdict
from functools import wraps
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    session,
    flash,
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

from database import (
    init_db,
    add_scan,
    get_user_scans,
    create_user,
    get_user_by_email,
    get_all_users_for_owner,
    add_feedback,
    get_recent_feedback,
)
from detector import analyze_video_pipeline, download_video_from_url

app = Flask(__name__)

# Environment-driven configuration
app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(32).hex()
max_upload_mb = int(os.environ.get("MAX_UPLOAD_MB", "100"))
app.config["MAX_CONTENT_LENGTH"] = max_upload_mb * 1024 * 1024
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("SESSION_COOKIE_SECURE", "0") == "1"

OWNER_EMAILS = {
    e.strip().lower()
    for e in os.environ.get("OWNER_EMAILS", "").split(",")
    if e.strip()
}

UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", os.path.join("uploads"))
STATIC_FOLDER = os.path.join("static")
OUTPUT_FOLDER = os.environ.get("OUTPUT_FOLDER", os.path.join(STATIC_FOLDER, "outputs"))

VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
ALLOWED_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs("models", exist_ok=True)

init_db()

_RATE_BUCKETS = defaultdict(list)


def is_rate_limited(bucket_name, max_requests=10, window_seconds=60):
    client_ip = request.remote_addr or "unknown"
    key = f"{bucket_name}:{client_ip}"
    now = time.time()
    recent = [ts for ts in _RATE_BUCKETS[key] if now - ts < window_seconds]
    if len(recent) >= max_requests:
        _RATE_BUCKETS[key] = recent
        return True
    recent.append(now)
    _RATE_BUCKETS[key] = recent
    return False


def is_owner_logged_in():
    email = (session.get("user_email") or "").strip().lower()
    return bool(email and email in OWNER_EMAILS)


def login_required_page(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please sign up or log in to access this page.", "error")
            return redirect(url_for("auth_page"))
        return fn(*args, **kwargs)

    return wrapper


def login_required_json(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"error": "Please log in first to analyze media."}), 401
        return fn(*args, **kwargs)

    return wrapper


def get_extension(filename):
    if not filename or "." not in filename:
        return ""
    return filename.rsplit(".", 1)[1].lower()


def allowed_file(filename):
    return get_extension(filename) in ALLOWED_EXTENSIONS


def detect_media_kind(path_or_name):
    ext = get_extension(path_or_name)
    return "image" if ext in IMAGE_EXTENSIONS else "video"


def is_safe_video_url(raw_url):
    try:
        parsed = urlparse(raw_url)
    except Exception:
        return False

    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return False

    hostname = (parsed.hostname or "").strip().lower()
    if not hostname or hostname in ("localhost", "127.0.0.1", "::1"):
        return False

    try:
        addr_info = socket.getaddrinfo(hostname, None)
        for info in addr_info:
            ip_str = info[4][0]
            ip_obj = ipaddress.ip_address(ip_str)
            if (
                ip_obj.is_private
                or ip_obj.is_loopback
                or ip_obj.is_link_local
                or ip_obj.is_multicast
                or ip_obj.is_reserved
                or ip_obj.is_unspecified
            ):
                return False
    except socket.gaierror:
        return False

    return True


def _build_default_ai_report(fake_prob, is_fake):
    return {
        "frame_count": 1 if is_fake else 0,
        "percentage": round(float(fake_prob or 0.0), 1) if is_fake else 0.0,
        "avg_fake_probability": round(float(fake_prob or 0.0), 1),
        "intervals": [],
    }


def _build_default_non_ai_report(fake_prob, is_fake):
    non_fake_pct = max(0.0, round(100.0 - float(fake_prob or 0.0), 1))
    return {
        "frame_count": 0 if is_fake else 1,
        "percentage": 0.0 if is_fake else non_fake_pct,
        "avg_fake_probability": round(float(fake_prob or 0.0), 1),
        "intervals": [],
    }


@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self' https://cdn.jsdelivr.net; "
        "img-src 'self' data: blob:; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net;"
    )
    return response


@app.errorhandler(413)
def file_too_large(_err):
    if request.path == "/analyze":
        return jsonify({"error": f"File exceeds the {max_upload_mb} MB upload limit."}), 413
    flash(f"File exceeds the {max_upload_mb} MB upload limit.", "error")
    return redirect(url_for("scanner_page"))


@app.errorhandler(500)
def internal_server_error(err):
    if request.path == "/analyze" or (
        request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html
    ):
        return jsonify({"error": f"Internal server error: {err}"}), 500
    flash("Something went wrong on the server.", "error")
    return redirect(url_for("home"))


# Page 1: Welcome Page
@app.route("/")
def home():
    return render_template(
        "index.html",
        page="welcome",
        current_user=session.get("user_name"),
        is_owner=is_owner_logged_in(),
    )


# Page 2: Sign Up & Log In
@app.route("/auth")
def auth_page():
    return render_template(
        "index.html",
        page="auth",
        current_user=session.get("user_name"),
        is_owner=is_owner_logged_in(),
    )


# Page 3: Media Analyzer (Photos & Videos From Files & From Social Media)
@app.route("/scanner")
@login_required_page
def scanner_page():
    return render_template(
        "index.html",
        page="scanner",
        current_user=session.get("user_name"),
        is_owner=is_owner_logged_in(),
        max_upload_mb=max_upload_mb,
    )


# Page 4: Output & User Scan History (+ Owner User Directory)
@app.route("/output")
@login_required_page
def output_page():
    user_id = session.get("user_id")
    user_scans = get_user_scans(user_id=user_id, limit=25)
    owner_users = get_all_users_for_owner() if is_owner_logged_in() else []
    return render_template(
        "index.html",
        page="output",
        current_user=session.get("user_name"),
        is_owner=is_owner_logged_in(),
        user_scans=user_scans,
        owner_users=owner_users,
    )


# Page 5: Feedback
@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    if request.method == "POST":
        if is_rate_limited("feedback", max_requests=10, window_seconds=60):
            flash("Too many feedback submissions. Please try again shortly.", "error")
            return redirect(url_for("feedback"))

        name = request.form.get("name", "").strip() or (session.get("user_name") or "")
        rating = request.form.get("rating", "5")
        comment = request.form.get("comment", "").strip()

        if not name or not comment:
            flash("Please enter your name and feedback comment.", "error")
            return redirect(url_for("feedback"))

        add_feedback(name, rating, comment)
        flash("Thank you! Your rating and feedback have been recorded.", "success")
        return redirect(url_for("feedback"))

    recent_feedback = get_recent_feedback(limit=10)
    return render_template(
        "index.html",
        page="feedback",
        current_user=session.get("user_name"),
        is_owner=is_owner_logged_in(),
        recent_feedback=recent_feedback,
    )


@app.route("/signup", methods=["POST"])
def signup():
    if is_rate_limited("signup", max_requests=8, window_seconds=60):
        flash("Too many sign-up attempts. Please wait a minute.", "error")
        return redirect(url_for("auth_page"))

    full_name = request.form.get("full_name", "").strip()
    email = request.form.get("email", "").strip()
    phone = request.form.get("phone", "").strip()
    password = request.form.get("password", "")

    if not full_name or not email or not phone or not password:
        flash("Name, email, phone number, and password are all required.", "error")
        return redirect(url_for("auth_page"))

    if len(password) < 8:
        flash("Password must be at least 8 characters long.", "error")
        return redirect(url_for("auth_page"))

    password_hash = generate_password_hash(password)
    created = create_user(full_name, email, phone, password_hash)

    if not created:
        flash("Could not create account. Check your details or try another email.", "error")
        return redirect(url_for("auth_page"))

    user = get_user_by_email(email)
    session.clear()
    session["user_id"] = user["id"]
    session["user_email"] = user["email"]
    session["user_name"] = user["full_name"]
    flash("Account created! Your scan history is now enabled.", "success")
    return redirect(url_for("scanner_page"))


@app.route("/login", methods=["POST"])
def login():
    if is_rate_limited("login", max_requests=10, window_seconds=60):
        flash("Too many login attempts. Please wait a minute.", "error")
        return redirect(url_for("auth_page"))

    email = request.form.get("email", "").strip()
    password = request.form.get("password", "")

    user = get_user_by_email(email)
    if not user or not check_password_hash(user["password_hash"], password):
        flash("Invalid email or password.", "error")
        return redirect(url_for("auth_page"))

    session.clear()
    session["user_id"] = user["id"]
    session["user_email"] = user["email"]
    session["user_name"] = user["full_name"]
    flash(f"Welcome back, {user['full_name']}!", "success")
    return redirect(url_for("scanner_page"))


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))


@app.route("/api/history")
@login_required_json
def api_history():
    scans = get_user_scans(user_id=session.get("user_id"), limit=25)
    return jsonify({"scans": scans})


@app.route("/analyze", methods=["POST"])
@login_required_json
def analyze():
    try:
        if is_rate_limited("analyze", max_requests=8, window_seconds=60):
            return jsonify({"error": "Rate limit reached. Please wait a moment before scanning again."}), 429

        start_time = time.perf_counter()
        source_mode = request.form.get("source_mode", "").strip()
        scan_mode = (request.form.get("scan_mode", "fast") or "fast").strip().lower()
        video_file = request.files.get("video_file") or request.files.get("media_file")
        video_url = request.form.get("video_url", "").strip()

        if source_mode == "file":
            video_url = ""
        elif source_mode == "social":
            video_file = None

        if (not video_file or video_file.filename == "") and not video_url:
            return jsonify({"error": "Please choose a photo or video from files or enter a social media URL."}), 400

        scan_id = uuid.uuid4().hex[:8]
        saved_path = None
        display_name = "unknown_media.mp4"
        media_type = "video"

        if video_file and video_file.filename != "":
            if not allowed_file(video_file.filename):
                return jsonify({
                    "error": (
                        "Unsupported file format. Please upload a video (.mp4, .mov, .avi, .mkv, .webm) "
                        "or photo (.jpg, .jpeg, .png, .webp)."
                    )
                }), 400
            safe_name = secure_filename(video_file.filename) or "uploaded_media.mp4"
            display_name = safe_name
            media_type = detect_media_kind(safe_name)
            saved_path = os.path.join(UPLOAD_FOLDER, f"{scan_id}_{safe_name}")
            video_file.save(saved_path)
        else:
            if not is_safe_video_url(video_url):
                return jsonify({
                    "error": "Invalid or blocked URL. Please provide a valid public http/https social media link."
                }), 400
            display_name = video_url[:200]
            media_type = "video"

        if not saved_path:
            saved_path = download_video_from_url(video_url, UPLOAD_FOLDER, scan_id)

        # Pass scan_mode only if analyze_video_pipeline accepts it; otherwise keep 3 args.
        try:
            result = analyze_video_pipeline(saved_path, OUTPUT_FOLDER, scan_id, scan_mode=scan_mode)
        except TypeError:
            result = analyze_video_pipeline(saved_path, OUTPUT_FOLDER, scan_id)

        processing_time = round(time.perf_counter() - start_time, 2)
        user_id = session.get("user_id")

        fake_prob = float(result.get("fake_probability", 0.0))
        is_fake = bool(result.get("is_fake", False))
        verdict_summary = str(result.get("verdict_summary", ""))

        add_scan(
            display_name,
            fake_prob,
            is_fake,
            verdict_summary,
            user_id=user_id,
            processing_time=processing_time,
            media_type=result.get("media_type") or media_type,
        )

        ai_report = result.get("ai_report") or _build_default_ai_report(fake_prob, is_fake)
        non_ai_report = result.get("non_ai_report") or _build_default_non_ai_report(fake_prob, is_fake)
        segments = result.get("segments") or []

        return jsonify({
            "scan_id": scan_id,
            "filename": display_name,
            "media_type": result.get("media_type") or media_type,
            "scan_mode": scan_mode,
            "processing_time": processing_time,
            "fake_probability": fake_prob,
            "is_fake": is_fake,
            "verdict_title": result.get("verdict_title", "Manipulation Analysis Complete"),
            "verdict_summary": verdict_summary,
            "ai_report": ai_report,
            "non_ai_report": non_ai_report,
            "segments": segments,
            "gradcam_images": result.get("gradcam_images", []),
            "spectrogram_url": result.get("spectrogram_url"),
            "timeline_url": result.get("timeline_url"),
            "crypto_traps": result.get("crypto_traps", []),
            "recent_scans": get_user_scans(user_id=user_id, limit=25),
        })
    except Exception as exc:
        app.logger.exception("Analyze route failed")
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


if __name__ == "__main__":
    debug_enabled = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_enabled, host="127.0.0.1", port=5000)
