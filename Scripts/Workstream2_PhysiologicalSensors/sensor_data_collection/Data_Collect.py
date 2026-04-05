import serial
import csv
from datetime import datetime, timezone, timedelta

PORT = 'COM11'      # 改成你的串口
BAUD = 9600

beijing_tz = timezone(timedelta(hours=8))
filename = datetime.now(beijing_tz).strftime("HR_GSR_Beijing_%Y%m%d_%H%M%S.csv")

ser = serial.Serial(PORT, BAUD, timeout=1)

print("Start collecting HR + GSR data...")
print("Saving to:", filename)

with open(filename, mode='w', newline='', encoding='utf-8-sig') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp_Beijing", "HR", "GSR"])

    try:
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()

            if line:
                try:
                    parts = line.split(',')
                    hr = None
                    gsr = None

                    for part in parts:
                        part = part.strip()
                        if part.startswith("BPM:"):
                            hr = int(part.split(":")[1].strip())
                        elif part.startswith("GSR:"):
                            gsr = int(part.split(":")[1].strip())

                    if hr is not None and gsr is not None:
                        timestamp = datetime.now(beijing_tz).strftime('%Y-%m-%d %H:%M:%S')
                        writer.writerow([f"'{timestamp}", hr, gsr])   # 前面加 ' ，让 Excel 按文本保存
                        file.flush()
                        print(timestamp, hr, gsr)

                except Exception:
                    pass

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        ser.close()
        print("Serial port closed.")
