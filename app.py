from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import io
from PIL import Image
import os

app = Flask(__name__)

masks = {
    "white": {"varnished": ([100, 20, 150], [120, 80, 255]), "unvarnished": ([100, 100, 30], [120, 200, 90])},
    "gray": {"varnished": ([100, 60, 200], [115, 140, 255]), "unvarnished": ([100, 40, 70], [115, 100, 160])},
    "black": {"varnished": ([0, 0, 30], [180, 50, 80]), "unvarnished": ([0, 0, 10], [180, 80, 40])},
    "red": {"varnished": ([0, 100, 100], [10, 255, 255]), "unvarnished": ([5, 140, 120], [20, 255, 255])},
    "blue": {"varnished": ([115, 80, 60], [130, 180, 200]), "unvarnished": ([110, 60, 20], [130, 150, 100])}
}

def detect_varnish_status(hsv, lower, upper, threshold=170):
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    mean_v = cv2.mean(hsv[:, :, 2], mask=mask)[0]
    return "varnished" if mean_v >= threshold else "unvarnished"

def detect_scratches_filtered(gray_img, mask):
    blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.bitwise_and(edges, edges, mask=mask)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(edges)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h) if h != 0 else 0
        area = cv2.contourArea(cnt)
        if 10 < area < 300 and aspect_ratio < 5:
            cv2.drawContours(filtered, [cnt], -1, 255, -1)
    return filtered

@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files or "color" not in request.form:
        return jsonify({"error": "Eksik veri"}), 400

    file = request.files["image"]
    color = request.form["color"]

    if color not in masks:
        return jsonify({"error": "Geçersiz renk"}), 400

    # Görseli oku
    img = Image.open(file.stream).convert("RGB")
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lower, upper = masks[color]["varnished"]
    status = detect_varnish_status(hsv, lower, upper)
    lower, upper = masks[color][status]

    # Putty Tespiti
    putty_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    putty_contours, _ = cv2.findContours(putty_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Scratch Tespiti
    scratch_mask = detect_scratches_filtered(gray, putty_mask)
    scratch_contours, _ = cv2.findContours(scratch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = img.copy()
    cv2.drawContours(output, putty_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(output, scratch_contours, -1, (0, 0, 255), 1)

    # Encode base64 for mobile
    _, buffer = cv2.imencode(".jpg", output)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "damage_detected": len(putty_contours) > 0 or len(scratch_contours) > 0,
        "putty_regions": len(putty_contours),
        "scratch_regions": len(scratch_contours),
        "image_base64": img_base64
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
