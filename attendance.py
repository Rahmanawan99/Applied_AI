from datetime import datetime
import os

def mark_attendance(name):

    print(f"[DEBUG] Attempting to mark attendance for: {name}")

    if name.lower() == "unknown":
        print("[DEBUG] Skipping: User is Unknown")
        return

    os.makedirs("logs", exist_ok=True)

    file_path = "logs/attendance_log.txt"

    today = datetime.now().strftime("%Y-%m-%d")

    if os.path.exists(file_path):

        with open(file_path, "r") as f:
            for line in f:
                if name in line and today in line:
                    print("Already marked today")
                    return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(file_path, "a") as f:
        f.write(f"{timestamp} | {name} | PRESENT\n")

    print(f"Attendance marked for {name}")