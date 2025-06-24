import cv2
import numpy as np
from tkinter import Tk, filedialog, simpledialog

# HSV renk aralıkları
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

        # Filtreleme: çok uzun ya da düz çizgileri çıkar
        if 10 < area < 300 and aspect_ratio < 5:
            cv2.drawContours(filtered, [cnt], -1, 255, -1)

    return filtered

def draw_freehand_mask(image):
    clone = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    drawing = [False]
    points = []

    def draw(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing[0] = True
            points.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE and drawing[0]:
            points.append((x, y))
            cv2.line(clone, points[-2], points[-1], (0, 255, 0), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing[0] = False
            if len(points) > 2:
                cv2.fillPoly(mask, [np.array(points)], 255)
            cv2.destroyWindow("Draw Region")

    cv2.namedWindow("Draw Region")
    cv2.setMouseCallback("Draw Region", draw)
    while True:
        cv2.imshow("Draw Region", clone)
        key = cv2.waitKey(1) & 0xFF
        if key == 13: break  # ENTER
        elif key == 27: break  # ESC
    cv2.destroyAllWindows()
    return mask

# MAIN
Tk().withdraw()
color = simpledialog.askstring("Vehicle Color", "Enter color (white, gray, black, red, blue):")
if not color or color.lower() not in masks:
    print("Invalid color selected.")
    exit()

path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
if not path:
    print("No image selected.")
    exit()

img = cv2.imread(path)
if img is None:
    print("Could not load image.")
    exit()

mask = draw_freehand_mask(img)
if np.count_nonzero(mask) == 0:
    print("No region selected.")
    exit()

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

lower, upper = masks[color]["varnished"]
status = detect_varnish_status(hsv, lower, upper)
lower, upper = masks[color][status]

# Putty Detection
putty_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
putty_mask = cv2.bitwise_and(putty_mask, putty_mask, mask=mask)
putty_contours, _ = cv2.findContours(putty_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Scratch Detection (filtered)
scratch_mask = detect_scratches_filtered(gray, mask)
scratch_contours, _ = cv2.findContours(scratch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output = img.copy()
cv2.drawContours(output, putty_contours, -1, (0, 255, 0), 2)
cv2.drawContours(output, scratch_contours, -1, (0, 0, 255), 1)

# Save
save_path = path.replace(".jpg", "_output.jpg").replace(".png", "_output.png")
cv2.imwrite(save_path, output)
print("✅ Done. Saved to:", save_path)
