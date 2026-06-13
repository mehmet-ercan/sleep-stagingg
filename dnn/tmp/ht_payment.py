import requests
import time
import tkinter as tk
from tkinter import messagebox

URL = "https://hisarturizm.com.tr/api/payment/balances?lastName=akcan&companyId=624&passportNoOrCitizenshipNumber=40007242942"

def show_popup(amount):
    root = tk.Tk()
    root.withdraw()  # ana pencereyi gizle

    messagebox.showwarning(
        "KRİTİK UYARI",
        f"Amount 2850 altına düştü!\nDeğer: {amount}"
    )

    root.destroy()

alert_shown = False  # spam olmasın diye

while True:
    try:
        response = requests.get(URL)
        data = response.json()

        amount = float(data["payment"]["total"]["amount"])
        print("Amount:", amount)

        if amount < 2850.00 and not alert_shown:
            show_popup(amount)
            alert_shown = True  # sadece 1 kez göster

    except Exception as e:
        print("Hata:", e)

    time.sleep(60)
    try:
        response = requests.get(URL)
        data = response.json()

        amount = float(data["payment"]["total"]["amount"])

        print("Amount:", amount)

        # Şart
        if amount < 2850.00:
            print("Amount 2850'den küçük! Döngü durduruluyor...")
            break

    except Exception as e:
        print("Hata:", e)

    time.sleep(60)  # 10 dk