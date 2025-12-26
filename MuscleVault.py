import sys
import os
import sqlite3
import datetime
import re
import urllib.request
import cv2
import numpy as np
import onnxruntime as ort
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QListWidget, QLabel, QPushButton, 
                             QFileDialog, QTabWidget, QScrollArea, QDialog, QCompleter,
                             QGridLayout, QCheckBox, QDateEdit, QLayout, QSizePolicy,
                             QSlider, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                             QComboBox, QFrame, QGroupBox, QMenu, QStatusBar, QSpinBox, QFormLayout, QLineEdit, QDoubleSpinBox, QInputDialog, QListWidgetItem, QMessageBox) 
from PyQt6.QtCore import Qt, QDate, QLocale, QPointF, pyqtSignal, QSize ,QTimer, QEvent
from PyQt6.QtGui import QPixmap, QPainter, QIcon, QFont, QKeySequence, QShortcut, QAction, QDragEnterEvent, QDropEvent, QImage
import uuid

# ============================================================================
# AYARLAR: PUANLAMA SİSTEMİ VE KATSAYILAR
# ============================================================================
SCORING_SYSTEM = {
    "Bikini": {},
    "Wellness": {},
    "Figure": {},
    "Physique": {},
    "Bodybuilding": {}
}

_GLOBAL_ORT_SESSION = None

model_path = "Real-ESRGAN-General-x4v3_fp16.onnx"  # dosya adın neyse

class RealESRGANOnnxUpscaler:
    def __init__(self, model_path: str, target_scale: int = 3, ai_strength: float = 0.75):
        """
        ai_strength: 0.0 ile 1.0 arası. 
        0.75 (%75) demek: %75 AI görüntüsü, %25 Orijinal doku. 
        Daha doğal olması için bu değeri 0.60 - 0.80 arasında tutabilirsin.
        """
        self.model_path = model_path
        self.tile = 128         
        self.model_scale = 4    
        self.target_scale = target_scale
        self.ai_strength = ai_strength # Doğallık ayarı

        global _GLOBAL_ORT_SESSION
        
        if _GLOBAL_ORT_SESSION is None:
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
            print(f"Model yükleniyor... (Hedef: x{self.target_scale}, Doğallık: %{int(ai_strength*100)})")
            try:
                _GLOBAL_ORT_SESSION = ort.InferenceSession(model_path, sess_options=so, providers=providers)
                print(f"Aktif Cihaz: {_GLOBAL_ORT_SESSION.get_providers()[0]}")
            except Exception as e:
                print(f"HATA: {e}")
                _GLOBAL_ORT_SESSION = None
                return

        self.sess = _GLOBAL_ORT_SESSION
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

    def _run_tile(self, tile_rgb_u8: np.ndarray) -> np.ndarray:
        if tile_rgb_u8.shape[0] != 128 or tile_rgb_u8.shape[1] != 128:
            tile_rgb_u8, _, _ = self._pad_image_to_fixed(tile_rgb_u8, 128)

        inp = tile_rgb_u8.astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))
        inp = np.expand_dims(inp, axis=0)

        result = self.sess.run([self.out_name], {self.in_name: inp})
        out = result[0]

        if out.ndim == 4: out = out[0]
        if out.ndim == 3 and out.shape[0] == 3: out = np.transpose(out, (1, 2, 0))
        
        out = np.clip(out, 0.0, 1.0)
        out = (out * 255.0).round().astype(np.uint8)
        return out

    def _pad_image_whole(self, img: np.ndarray) -> tuple[np.ndarray, int, int]:
        h, w, c = img.shape
        pad_h = (self.tile - (h % self.tile)) % self.tile
        pad_w = (self.tile - (w % self.tile)) % self.tile
        if pad_h == 0 and pad_w == 0: return img, 0, 0
        padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        return padded, pad_h, pad_w

    def _pad_image_to_fixed(self, img: np.ndarray, target_size: int):
        h, w, c = img.shape
        pad_h = target_size - h
        pad_w = target_size - w
        if pad_h < 0 or pad_w < 0: return img, 0, 0
        if pad_h == 0 and pad_w == 0: return img, 0, 0
        padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        return padded, pad_h, pad_w
    
    def _apply_unsharp_mask(self, image, kernel_size=(5, 5), sigma=1.0, amount=0.5, threshold=0):
        """Resme hafif bir keskinlik vererek detayları patlatır."""
        gaussian = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = float(amount + 1) * image - float(amount) * gaussian
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        return sharpened.round().astype(np.uint8)

    def upscale_pixmap(self, pixmap: QPixmap):
        if pixmap is None or pixmap.isNull(): return None

        try:
            # 1. Hazırlık
            qimg = pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
            orig_w, orig_h = qimg.width(), qimg.height()
            
            ptr = qimg.bits()
            ptr.setsize(orig_h * orig_w * 4)
            arr = np.frombuffer(ptr, np.uint8).reshape((orig_h, orig_w, 4))
            rgb = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
            
            # --- ADIM 1: AI UPSCALE (x4) ---
            padded, pad_h, pad_w = self._pad_image_whole(rgb)
            H, W, _ = padded.shape
            out_ai = np.zeros((H * self.model_scale, W * self.model_scale, 3), dtype=np.uint8)

            for y in range(0, H, self.tile):
                for x in range(0, W, self.tile):
                    tile = padded[y:y+self.tile, x:x+self.tile, :]
                    up_tile = self._run_tile(tile)
                    oy, ox = y * self.model_scale, x * self.model_scale
                    th, tw, _ = up_tile.shape
                    out_ai[oy:oy+th, ox:ox+tw, :] = up_tile

            final_h_x4 = orig_h * self.model_scale
            final_w_x4 = orig_w * self.model_scale
            out_ai = out_ai[:final_h_x4, :final_w_x4, :]

            # --- ADIM 2: HEDEF BOYUTA İNDİRME (Resize) ---
            final_w_target = orig_w * self.target_scale
            final_h_target = orig_h * self.target_scale
            
            # AI sonucunu hedef boyuta getir
            ai_resized = cv2.resize(out_ai, (final_w_target, final_h_target), interpolation=cv2.INTER_AREA)

            # --- ADIM 3: TEXTURE BLENDING (DOĞALLIK İÇİN) ---
            # Orijinal resmi de klasik yöntemle (LANCZOS) aynı boyuta getir
            original_resized = cv2.resize(rgb, (final_w_target, final_h_target), interpolation=cv2.INTER_LANCZOS4)
            
            # İkisini karıştır: (AI * strength) + (Original * (1-strength))
            # ai_strength 1.0 ise tamamen yapay, 0.0 ise tamamen orijinal olur.
            # 0.70 civarı en iyi "doğal ama net" görüntüyü verir.
            blended = cv2.addWeighted(ai_resized, self.ai_strength, original_resized, 1 - self.ai_strength, 0)

            # --- ADIM 4: HAFİF KESKİNLİK (UNSHARP MASK) ---
            # Harmanlama sonrası hafif yumuşama olabilir, bunu telafi et
            final_output = self._apply_unsharp_mask(blended, amount=0.5)

            # 5. Kaydet
            final_output = np.ascontiguousarray(final_output)
            h_final, w_final, ch = final_output.shape
            bytes_per_line = ch * w_final
            qimg2 = QImage(final_output.tobytes(), w_final, h_final, bytes_per_line, QImage.Format.Format_RGB888)
            return QPixmap.fromImage(qimg2.copy())
            
        except Exception as e:
            print("Upscale Error:", e)
            import traceback
            traceback.print_exc()
            return None

class SilentConfirmDialog(QDialog):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Onay")
        self.setModal(True)
        self.resize(300, 120)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(text))

        btns = QHBoxLayout()
        btn_no = QPushButton("İptal")
        btn_yes = QPushButton("Sil")
        btn_yes.setStyleSheet("background:#c62828;color:white;")

        btns.addStretch()
        btns.addWidget(btn_no)
        btns.addWidget(btn_yes)
        layout.addLayout(btns)

        btn_no.clicked.connect(self.reject)
        btn_yes.clicked.connect(self.accept)

class CompetitionManagerDialog(QDialog):
    def __init__(self, db, parent=None):
        super().__init__(parent)
        self.db = db
        self.setWindowTitle("🏆 Yarışma Yöneticisi")
        self.setFixedSize(500, 600)
        self.init_ui()
        self.load_competitions()

    def init_ui(self):
        layout = QVBoxLayout()

        # Bilgi
        info = QLabel("Veritabanındaki kayıtlı yarışmalar. Fotoğraf yüklerken bu liste önerilir.")
        info.setWordWrap(True)
        layout.addWidget(info)

        # Liste
        self.list_comps = QListWidget()
        layout.addWidget(self.list_comps)

        # Butonlar
        btn_layout = QHBoxLayout()
        
        self.btn_import = QPushButton("📥 CSV İçe Aktar")
        self.btn_import.clicked.connect(self.import_csv)
        self.btn_import.setStyleSheet("background-color: #00897B; color: white;")
        
        self.btn_export = QPushButton("📤 CSV Dışa Aktar")
        self.btn_export.clicked.connect(self.export_csv)
        self.btn_export.setStyleSheet("background-color: #555; color: white;")

        btn_layout.addWidget(self.btn_import)
        btn_layout.addWidget(self.btn_export)
        layout.addLayout(btn_layout)
        
        # Silme Butonu
        self.btn_del = QPushButton("🗑️ Seçiliyi Sil")
        self.btn_del.clicked.connect(self.delete_selection)
        layout.addWidget(self.btn_del)

        self.setLayout(layout)

    def load_competitions(self):
        self.list_comps.clear()
        # Veritabanında competitions tablosu yoksa oluşturmayı dene
        try:
            cur = self.db.conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS competitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    date_str TEXT,
                    UNIQUE(name, date_str)
                )
            """)
            self.db.conn.commit()
            
            # Listeyi Çek
            cur.execute("SELECT name, date_str FROM competitions ORDER BY date_str DESC")
            rows = cur.fetchall()
            for name, date_str in rows:
                item = QListWidgetItem(f"{date_str} - {name}")
                item.setData(Qt.ItemDataRole.UserRole, (name, date_str)) # Veriyi sakla
                self.list_comps.addItem(item)
                
        except Exception as e:
            print(f"DB Hatası: {e}")

    def import_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Yarışma Listesi Yükle", "", "CSV (*.csv);;TXT (*.txt)")
        if not path: return

        import csv
        count = 0
        try:
            cur = self.db.conn.cursor()
            with open(path, mode='r', encoding='utf-8-sig') as f:
                reader = csv.reader(f, delimiter=';') # Noktalı virgül varsayalım
                next(reader, None) # Başlığı atla
                
                for row in reader:
                    if len(row) < 2: continue
                    name = row[0].strip()
                    date_str = row[1].strip() # Format: YYYY-AA-GG olmalı
                    
                    if name and date_str:
                        try:
                            cur.execute("INSERT OR IGNORE INTO competitions (name, date_str) VALUES (?, ?)", (name, date_str))
                            count += 1
                        except: pass
            
            self.db.conn.commit()
            self.load_competitions()
            QMessageBox.information(self, "Başarılı", f"{count} yeni yarışma eklendi.")

        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Yükleme başarısız:\n{e}")

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Listeyi Kaydet", "Yarismalar.csv", "CSV (*.csv)")
        if not path: return

        import csv
        try:
            cur = self.db.conn.cursor()
            cur.execute("SELECT name, date_str FROM competitions ORDER BY date_str DESC")
            rows = cur.fetchall()

            with open(path, mode='w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(["Yarisma Adi", "Tarih"])
                for name, date in rows:
                    writer.writerow([name, date])
            
            QMessageBox.information(self, "Başarılı", "Liste kaydedildi.")

        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Kaydetme başarısız:\n{e}")

    def delete_selection(self):
        item = self.list_comps.currentItem()
        if not item: return
        
        name, date_str = item.data(Qt.ItemDataRole.UserRole)
        
        res = QMessageBox.question(self, "Sil", f"{name} ({date_str}) silinsin mi?\n(Fotoğraflar silinmez, sadece listeden kalkar.)", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if res == QMessageBox.StandardButton.Yes:
            cur = self.db.conn.cursor()
            cur.execute("DELETE FROM competitions WHERE name=? AND date_str=?", (name, date_str))
            self.db.conn.commit()
            self.load_competitions()
        

class SettingsDialog(QDialog):
    def __init__(self, db, parent=None):
        super().__init__(parent)
        self.db = db
        self.setWindowTitle("⚙️ Program Ayarları & AI")
        self.resize(400, 350)
        
        layout = QVBoxLayout(self)
        
        # --- AI AYARLARI GRUBU ---
        grp_ai = QGroupBox("🤖 Yapay Zeka (AI) Ayarları")
        layout_ai = QVBoxLayout()
        
        # 1. Otomatik Uygulama
        self.chk_auto = QCheckBox("Otomatik AI Upscale (Yüklerken Uygula)")
        self.chk_auto.setToolTip("Fotoğrafları yüklerken otomatik olarak kaliteyi artırır.")
        layout_ai.addWidget(self.chk_auto)
        layout_ai.addWidget(QLabel("<small style='color:gray'>RTX 4070 Ti ile süper hızlıdır.</small>"))
        
        # 2. Scale (Büyütme Oranı)
        layout_scale = QHBoxLayout()
        layout_scale.addWidget(QLabel("Büyütme Hedefi:"))
        self.cmb_scale = QComboBox()
        self.cmb_scale.addItems(["x2 (Hafif Büyütme)", "x3 (Önerilen)", "x4 (Maksimum)"])
        layout_scale.addWidget(self.cmb_scale)
        layout_ai.addLayout(layout_scale)
        
        # 3. Strength (Doğallık Ayarı)
        layout_str = QVBoxLayout()
        self.lbl_str_val = QLabel("Karışım Oranı: %70 AI (Daha Doğal)")
        layout_str.addWidget(self.lbl_str_val)
        
        self.slider_str = QSlider(Qt.Orientation.Horizontal)
        self.slider_str.setRange(0, 100) # 0 - 100 arası
        self.slider_str.valueChanged.connect(self.update_str_label)
        layout_str.addWidget(self.slider_str)
        
        layout_ai.addLayout(layout_str)
        grp_ai.setLayout(layout_ai)
        layout.addWidget(grp_ai)

        # --- FOTOĞRAF KLASÖRÜ ---
        grp_paths = QGroupBox("📂 Fotoğraf Klasörü")
        layout_paths = QHBoxLayout()
        self.txt_image_dir = QLineEdit()
        self.txt_image_dir.setReadOnly(True)
        btn_browse = QPushButton("Klasör Seç")
        btn_browse.clicked.connect(self.browse_image_dir)
        layout_paths.addWidget(self.txt_image_dir, 1)
        layout_paths.addWidget(btn_browse)
        grp_paths.setLayout(layout_paths)
        layout.addWidget(grp_paths)

        # Butonlar
        btn_layout = QHBoxLayout()
        btn_save = QPushButton("💾 Kaydet")
        btn_save.clicked.connect(self.save_settings)
        btn_save.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        
        btn_cancel = QPushButton("İptal")
        btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_save)
        layout.addLayout(btn_layout)
        
        self.load_current_values()

    def update_str_label(self, val):
        self.lbl_str_val.setText(f"Karışım Oranı: %{val} AI - %{100-val} Orijinal")

    def load_current_values(self):
        # DB'den oku ve UI'ya yansıt
        auto = self.db.get_setting("ai_auto_apply", "0")
        scale = self.db.get_setting("ai_scale", "3")
        strength = self.db.get_setting("ai_strength", "0.70")
        image_dir = self.db.get_setting("image_base_dir", os.path.abspath("images"))

        self.chk_auto.setChecked(True if auto == "1" else False)
        
        # Scale (x2->0, x3->1, x4->2)
        if scale == "2": self.cmb_scale.setCurrentIndex(0)
        elif scale == "3": self.cmb_scale.setCurrentIndex(1)
        else: self.cmb_scale.setCurrentIndex(2)
        
        # Strength (0.70 -> 70)
        str_val = int(float(strength) * 100)
        self.slider_str.setValue(str_val)
        self.txt_image_dir.setText(image_dir)

    def save_settings(self):
        # UI'dan al DB'ye yaz
        auto_val = "1" if self.chk_auto.isChecked() else "0"
        
        idx = self.cmb_scale.currentIndex()
        scale_val = "2" if idx == 0 else ("3" if idx == 1 else "4")
        
        str_val = self.slider_str.value() / 100.0 # 70 -> 0.70

        self.db.update_setting("ai_auto_apply", auto_val)
        self.db.update_setting("ai_scale", scale_val)
        self.db.update_setting("ai_strength", str(str_val))
        img_dir = self.txt_image_dir.text().strip()
        if img_dir:
            self.db.update_setting("image_base_dir", os.path.abspath(img_dir))

        self.accept()

    def browse_image_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Fotoğraf Klasörü Seç")
        if path:
            self.txt_image_dir.setText(path)


# --- VERİTABANI YÖNETİCİSİ (YEREL - SQLITE) ---
class DatabaseManager:
    def __init__(self):
        self.conn = sqlite3.connect("fitness_app_v13_external.db") # Yeni DB ismi (çakışmasın diye)
        self.create_tables()
        self.seed_default_criteria_if_empty()

        # Varsayılan Ayarlar
        self.set_default_setting("ai_scale", "3")
        self.set_default_setting("ai_strength", "0.70")
        self.set_default_setting("ai_auto_apply", "0")
        self.set_default_setting("image_base_dir", os.path.abspath("images"))

        # Klasör yapılarını oluştur
        self.ensure_image_dirs()

    def close(self):
        if self.conn:
            self.conn.close()

    def seed_default_criteria_if_empty(self):
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM scoring_criteria")
        if cur.fetchone()[0] > 0:
            return

        defaults = {
            "Bikini": [
                ("Arms (Triceps, Forearms, Biceps)", 3, 1),
                ("Shoulders (Size, Roundness)", 5, 2),
                ("Abs/Midsection (Conditioning, Visible Abs, Tiny Waist)", 10, 3),
                ("Quads (Striation, Size, Sweep)", 7, 4),
                ("Hamstrings (Striation)", 7, 5),
                ("Glutes (Size, Shape, Striation)", 10, 6),
                ("Calves (Size, Shape, Striation)", 7, 7),
                ("Overall Shape (X Shape), Symmetry / Proportion", 10, 8),
                ("Presentation / Stage Presence / Skin Tone / Look", 10, 9),
                ("Conditioning / Striation", 8, 10),
            ],

            # Max Total = 860
            "Wellness": [
                ("Arms (Triceps, Forearms, Biceps)", 5, 1),
                ("Shoulders (Size, Roundness)", 7, 2),
                ("Abs/Midsection (Conditioning, Visible Abs, Tiny Waist)", 9, 3),
                ("Quads (Striation, Size, Sweep)", 10, 4),
                ("Hamstrings (Striation)", 10, 5),
                ("Glutes (Size, Shape, Striation)", 10, 6),
                ("Calves (Size, Shape, Striation)", 7, 7),
                ("Overall Shape (X Shape), Symmetry / Proportion", 10, 8),
                ("Presentation / Stage Presence / Skin Tone / Look", 10, 9),
                ("Conditioning / Striation", 8, 10),
            ],

            # Max Total = 1070
            "Figure": [
                ("Arms (Triceps, Forearms, Biceps)", 6, 1),
                ("Shoulders (Size, Roundness)", 8, 2),
                ("Front Lats (Wideness)", 8, 3),
                ("Abs/Midsection (Conditioning, Visible Abs, Tiny Waist)", 9, 4),
                ("Quads (Striation, Size, Sweep)", 9, 5),
                ("Back Lats (Wideness)", 8, 6),
                ("Traps, Rear Delts, Mid-Back (Thickness)", 9, 7),
                ("Hamstrings (Striation)", 8, 8),
                ("Glutes (Size, Shape, Striation)", 8, 9),
                ("Calves (Size, Shape, Striation)", 7, 10),
                ("Overall Shape (X Shape) Size, Symmetry / Proportion", 10, 11),
                ("Presentation / Stage Presence / Skin Tone / Look", 8, 12),
                ("Conditioning / Striation", 9, 13),
            ],

            # Max Total = 1070
            "Physique": [
                ("Arms (Triceps, Forearms, Biceps)", 6, 1),
                ("Shoulders (Size, Roundness)", 8, 2),
                ("Front Lats (Wideness)", 8, 3),
                ("Abs/Midsection (Conditioning, Visible Abs, Tiny Waist)", 9, 4),
                ("Quads (Striation, Size, Sweep)", 9, 5),
                ("Back Lats (Wideness)", 8, 6),
                ("Traps, Rear Delts, Mid-Back (Thickness)", 9, 7),
                ("Hamstrings (Striation)", 8, 8),
                ("Glutes (Size, Shape, Striation)", 8, 9),
                ("Calves (Size, Shape, Striation)", 7, 10),
                ("Overall Shape (X Shape) Size, Symmetry / Proportion", 10, 11),
                ("Presentation / Stage Presence / Skin Tone / Look", 8, 12),
                ("Conditioning / Striation", 9, 13),
            ],

            # Max Total = 1070
            "Bodybuilding": [
                ("Arms (Triceps, Forearms, Biceps)", 6, 1),
                ("Shoulders (Size, Roundness)", 8, 2),
                ("Front Lats (Wideness)", 8, 3),
                ("Abs/Midsection (Conditioning, Visible Abs, Tiny Waist)", 9, 4),
                ("Quads (Striation, Size, Sweep)", 9, 5),
                ("Back Lats (Wideness)", 8, 6),
                ("Traps, Rear Delts, Mid-Back (Thickness)", 9, 7),
                ("Hamstrings (Striation)", 8, 8),
                ("Glutes (Size, Shape, Striation)", 8, 9),
                ("Calves (Size, Shape, Striation)", 7, 10),
                ("Overall Shape (X Shape) Size, Symmetry / Proportion", 10, 11),
                ("Presentation / Stage Presence / Skin Tone / Look", 8, 12),
                ("Conditioning / Striation", 9, 13),
            ],
        }
        
        for div, items in defaults.items():
            for crit, weight, order in items:
                cur.execute("""INSERT INTO scoring_criteria(division, criterion, weight, sort_order)
                            VALUES (?, ?, ?, ?)""", (div, crit, float(weight), int(order)))
        self.conn.commit()

    def set_setting(self, key, value):
        """Genel ayarları veya son seçilen durumları veritabanına kaydeder."""
        try:
            cur = self.conn.cursor()
            # Önce tablonun olduğundan emin olalım (Garantiye almak için)
            cur.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
            
            # Değeri kaydet veya varsa güncelle (REPLACE)
            cur.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, str(value)))
            self.conn.commit()
        except Exception as e:
            print(f"Ayar kaydedilirken hata oluştu: {e}")

    def get_criteria(self, division):
        cur = self.conn.cursor()
        cur.execute("""SELECT criterion, weight FROM scoring_criteria
                    WHERE division=?
                    ORDER BY sort_order ASC, criterion ASC""", (division,))
        return cur.fetchall()

    def upsert_criteria(self, division, items):
        """
        items: list of tuples (criterion:str, weight:float, sort_order:int)
        """
        cur = self.conn.cursor()
        # division için mevcutları temizleyip yeniden yazmak (basit ve sağlam)
        cur.execute("DELETE FROM scoring_criteria WHERE division=?", (division,))
        for crit, w, order in items:
            crit = crit.strip()
            if not crit:
                continue
            cur.execute("""INSERT INTO scoring_criteria(division, criterion, weight, sort_order)
                        VALUES (?, ?, ?, ?)""", (division, crit, float(w), int(order)))
        self.conn.commit()


    def create_tables(self):
        cur = self.conn.cursor()
        # image_data BLOB olarak saklanacak (Resim dosyasının kendisi)
        cur.execute("PRAGMA foreign_keys = ON;")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        cur.execute('''CREATE TABLE IF NOT EXISTS global_competitions (
            name TEXT PRIMARY KEY,
            last_date TEXT
        )''')

        cur.execute('''CREATE TABLE IF NOT EXISTS scoring_criteria (
                        division TEXT NOT NULL,
                        criterion TEXT NOT NULL,
                        weight REAL NOT NULL DEFAULT 1,
                        sort_order INTEGER NOT NULL DEFAULT 0,
                        PRIMARY KEY (division, criterion)
                    )''')

        cur.execute('''CREATE TABLE IF NOT EXISTS athletes (
                        id INTEGER PRIMARY KEY,
                        name TEXT,
                        division TEXT DEFAULT 'Bikini',
                        total_score REAL DEFAULT 0)''')

        #  PHOTOS TABLOSU
        cur.execute('''CREATE TABLE IF NOT EXISTS photos (
                        id INTEGER PRIMARY KEY,
                        athlete_id INTEGER,
                        date TEXT,
                        competition TEXT,
                        image_path TEXT,      -- Orijinal Dosya Yolu
                        thumb_path TEXT,      -- Küçük Resim Dosya Yolu
                        rank INTEGER DEFAULT 1,
                        FOREIGN KEY (athlete_id) REFERENCES athletes(id) ON DELETE CASCADE
                       )''')

        cur.execute('''CREATE TABLE IF NOT EXISTS favorite_athletes (
                        athlete_id INTEGER PRIMARY KEY,
                        pinned_at TEXT DEFAULT (datetime('now')),
                        FOREIGN KEY (athlete_id) REFERENCES athletes(id) ON DELETE CASCADE
                    )''')

        cur.execute('''CREATE TABLE IF NOT EXISTS favorite_competitions (
                        athlete_id INTEGER NOT NULL,
                        year TEXT NOT NULL,
                        competition TEXT NOT NULL,
                        PRIMARY KEY (athlete_id, year),
                        FOREIGN KEY (athlete_id) REFERENCES athletes(id) ON DELETE CASCADE
                    )''')

        self.conn.commit()

        cur.execute("CREATE INDEX IF NOT EXISTS idx_photos_lookup ON photos(athlete_id, date, competition, rank)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_athletes_div_score ON athletes(division, total_score)")
        self.conn.commit()

    def upsert_global_competition(self, comp_name, date_str):
        comp_name = (comp_name or "").strip()
        if not comp_name:
            return

        cur = self.conn.cursor()
        cur.execute(
            """INSERT INTO global_competitions(name, last_date)
            VALUES(?, ?)
            ON CONFLICT(name) DO UPDATE SET last_date=excluded.last_date""",
            (comp_name, date_str)
        )
        self.conn.commit()


    def ensure_competition_record(self, comp_name, date_str):
        comp_name = (comp_name or "").strip()
        if not comp_name:
            return
        cur = self.conn.cursor()
        try:
            cur.execute("INSERT OR IGNORE INTO competitions (name, date_str) VALUES (?, ?)", (comp_name, date_str))
            cur.execute("UPDATE competitions SET date_str=? WHERE name=? AND (date_str IS NULL OR date_str='')", (date_str, comp_name))
            self.conn.commit()
        except Exception:
            pass

    def set_default_setting(self, key, value):
        """Eğer ayar yoksa ekler, varsa dokunmaz."""
        cur = self.conn.cursor()
        cur.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (key, str(value)))
        self.conn.commit()

    def get_setting(self, key, default=None):
        cur = self.conn.cursor()
        cur.execute("SELECT value FROM settings WHERE key=?", (key,))
        row = cur.fetchone()
        return row[0] if row else default

    def update_setting(self, key, value):
        cur = self.conn.cursor()
        cur.execute("REPLACE INTO settings (key, value) VALUES (?, ?)", (key, str(value)))
        self.conn.commit()

    def store_path_for_db(self, path):
        """Kayıt için yolu normalize eder (mümkünse base klasöre göre relatif)."""
        if not path:
            return path
        base_dir, _, _ = self.get_image_dirs()
        abs_path = os.path.abspath(path)
        try:
            if os.path.commonpath([abs_path, base_dir]) == base_dir:
                return os.path.relpath(abs_path, base_dir)
        except Exception:
            pass
        return abs_path

    def get_image_dirs(self):
        """Seçilen fotoğraf klasörü ve alt dizinlerini döndürür, yoksa oluşturur."""
        stored_dir = self.get_setting("image_base_dir", os.path.abspath("images"))
        base_dir = os.path.abspath(stored_dir)
        default_dir = os.path.abspath("images")

        def ensure_dirs(target_base):
            target_base = os.path.abspath(target_base)
            orig = os.path.join(target_base, "originals")
            thumb = os.path.join(target_base, "thumbnails")
            os.makedirs(orig, exist_ok=True)
            os.makedirs(thumb, exist_ok=True)
            return target_base, orig, thumb

        try:
            base_dir, orig_dir, thumb_dir = ensure_dirs(base_dir)
        except OSError:
            # Fall back if the saved path was from another machine/user and not writable
            base_dir, orig_dir, thumb_dir = ensure_dirs(default_dir)

        # Normalize edilmiş yolu sakla
        self.update_setting("image_base_dir", base_dir)
        return base_dir, orig_dir, thumb_dir

    def ensure_image_dirs(self):
        self.get_image_dirs()

    def resolve_path_from_db(self, path, prefer_thumb=False):
        """DB'deki yolu mevcut base klasöre göre çözer, taşınmışsa düzeltir."""
        if not path:
            return ""

        base_dir, orig_dir, thumb_dir = self.get_image_dirs()

        # 1) Relative ise doğrudan base'e bağla
        if not os.path.isabs(path):
            candidate = os.path.abspath(os.path.join(base_dir, path))
            if os.path.exists(candidate):
                return candidate

        # 2) Absolute ve mevcutsa
        if os.path.isabs(path) and os.path.exists(path):
            return path

        # 3) Fallback: dosya adına göre base içinde ara
        fname = os.path.basename(path)
        search_dirs = [thumb_dir, orig_dir, base_dir] if prefer_thumb else [orig_dir, thumb_dir, base_dir]
        for d in search_dirs:
            candidate = os.path.join(d, fname)
            if os.path.exists(candidate):
                return candidate

        # Bulunamadı, olduğu gibi döndür
        return path

    def resolve_photo_paths(self, photo_id, image_path, thumb_path):
        """Yolları çözer; değiştiyse DB'yi günceller."""
        new_img = self.resolve_path_from_db(image_path, prefer_thumb=False)
        new_thumb = self.resolve_path_from_db(thumb_path, prefer_thumb=True)
        if new_img != image_path or new_thumb != thumb_path:
            cur = self.conn.cursor()
            cur.execute(
                "UPDATE photos SET image_path=?, thumb_path=? WHERE id=?",
                (self.store_path_for_db(new_img), self.store_path_for_db(new_thumb), photo_id)
            )
            self.conn.commit()
        return new_img, new_thumb

    def get_global_comp_date_mapping(self):
        cur = self.conn.cursor()
        cur.execute("SELECT name, last_date FROM global_competitions")
        rows = cur.fetchall()
        comp_to_date = {name: date for name, date in rows if name and date}
        return comp_to_date

    def get_global_competitions(self):
        cur = self.conn.cursor()
        cur.execute("SELECT name FROM global_competitions ORDER BY name COLLATE NOCASE ASC")
        return [r[0] for r in cur.fetchall()]

    def get_all_competitions(self):
        cur = self.conn.cursor()
        cur.execute("""
            SELECT name FROM competitions WHERE name IS NOT NULL
            UNION
            SELECT name FROM global_competitions WHERE name IS NOT NULL
            ORDER BY name COLLATE NOCASE ASC
        """)
        return [r[0] for r in cur.fetchall()]

    def delete_global_competition(self, comp_name):
        """Global yarışma listesinden (önerilerden) ismi siler."""
        cur = self.conn.cursor()
        cur.execute("DELETE FROM global_competitions WHERE name=?", (comp_name,))
        self.conn.commit()

    def update_athlete_name(self, athlete_id, new_name):
        cur = self.conn.cursor()
        cur.execute("UPDATE athletes SET name=? WHERE id=?", (new_name.strip(), athlete_id))
        self.conn.commit()

    def add_athlete(self, name, division):
        cur = self.conn.cursor()
        cur.execute("INSERT INTO athletes (name, division, total_score) VALUES (?, ?, 0)", (name, division))
        self.conn.commit()

    def update_athlete_score(self, athlete_id, score):
        cur = self.conn.cursor()
        cur.execute("UPDATE athletes SET total_score=? WHERE id=?", (score, athlete_id))
        self.conn.commit()

    def delete_athlete(self, athlete_id):
        cur = self.conn.cursor()
        cur.execute("DELETE FROM favorite_athletes WHERE athlete_id=?", (athlete_id,))
        cur.execute("DELETE FROM favorite_competitions WHERE athlete_id=?", (athlete_id,))
        cur.execute("DELETE FROM photos WHERE athlete_id=?", (athlete_id,))
        cur.execute("DELETE FROM athletes WHERE id=?", (athlete_id,))
        self.conn.commit()

    def set_favorite_competition(self, athlete_id, year, competition):
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO favorite_competitions(athlete_id, year, competition)
            VALUES (?, ?, ?)
            ON CONFLICT(athlete_id, year) DO UPDATE SET competition=excluded.competition
        """, (athlete_id, str(year), competition))
        self.conn.commit()

    def remove_favorite_competition(self, athlete_id, year):
        cur = self.conn.cursor()
        cur.execute("DELETE FROM favorite_competitions WHERE athlete_id=? AND year=?", (athlete_id, str(year)))
        self.conn.commit()

    def get_favorite_competitions(self, athlete_id):
        cur = self.conn.cursor()
        cur.execute("SELECT year, competition FROM favorite_competitions WHERE athlete_id=?", (athlete_id,))
        return {row[0]: row[1] for row in cur.fetchall()}

    def is_favorite_competition(self, athlete_id, year, competition):
        cur = self.conn.cursor()
        cur.execute("""SELECT 1 FROM favorite_competitions 
                       WHERE athlete_id=? AND year=? AND competition=?""",
                    (athlete_id, str(year), competition))
        return cur.fetchone() is not None

    def add_favorite_athlete(self, athlete_id):
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO favorite_athletes(athlete_id, pinned_at)
            VALUES (?, datetime('now'))
            ON CONFLICT(athlete_id) DO UPDATE SET pinned_at=datetime('now')
        """, (athlete_id,))
        self.conn.commit()

    def remove_favorite_athlete(self, athlete_id):
        cur = self.conn.cursor()
        cur.execute("DELETE FROM favorite_athletes WHERE athlete_id=?", (athlete_id,))
        self.conn.commit()

    def get_favorite_athletes(self, division=None):
        cur = self.conn.cursor()
        base_query = """
            SELECT a.id, a.name, a.division, a.total_score
            FROM favorite_athletes f
            JOIN athletes a ON a.id = f.athlete_id
        """
        if division:
            cur.execute(base_query + " WHERE a.division=? ORDER BY f.pinned_at DESC", (division,))
        else:
            cur.execute(base_query + " ORDER BY f.pinned_at DESC")
        return cur.fetchall()

    def is_favorite_athlete(self, athlete_id):
        cur = self.conn.cursor()
        cur.execute("SELECT 1 FROM favorite_athletes WHERE athlete_id=?", (athlete_id,))
        return cur.fetchone() is not None

    def get_comp_dates(self, athlete_id, competition):
        cur = self.conn.cursor()
        cur.execute("""SELECT DISTINCT date FROM photos
                       WHERE athlete_id=? AND competition=?
                       ORDER BY date DESC""",
                    (athlete_id, competition))
        return [r[0] for r in cur.fetchall()]

    def get_athlete(self, athlete_id):
        cur = self.conn.cursor()
        cur.execute("SELECT id, name, division, total_score FROM athletes WHERE id=?", (athlete_id,))
        return cur.fetchone()

    def get_athletes_by_division(self, division):
        cur = self.conn.cursor()
        cur.execute("SELECT id, name, total_score FROM athletes WHERE division=? ORDER BY total_score DESC", (division,))
        return cur.fetchall()

    def get_all_athletes(self):
        cur = self.conn.cursor()
        cur.execute("SELECT id, name, division FROM athletes ORDER BY division ASC, name ASC")
        return cur.fetchall()

    def normalize_ranks(self, athlete_id, date_str, competition):
        cur = self.conn.cursor()
        cur.execute("""SELECT id FROM photos
                       WHERE athlete_id=? AND date=? AND competition=?
                       ORDER BY rank ASC, id ASC""",
                    (athlete_id, date_str, competition))
        ids = [r[0] for r in cur.fetchall()]
        for i, pid in enumerate(ids, start=1):
            cur.execute("UPDATE photos SET rank=? WHERE id=?", (i, pid))
        self.conn.commit()

    def get_next_rank(self, athlete_id, date_str, competition):
        cur = self.conn.cursor()
        cur.execute("""SELECT MAX(rank) FROM photos
                       WHERE athlete_id=? AND date=? AND competition=?""",
                    (athlete_id, date_str, competition))
        max_rank = cur.fetchone()[0]
        return 1 if max_rank is None else max_rank + 1

    def add_photo(self, athlete_id, date, competition, file_path=None, image_data=None, image_path=None, thumb_path=None):
        """
        Resmi diske kaydeder (Orijinal + Thumbnail) ve yollarını DB'ye yazar.
        """
        
        # Yeni şema: Eğer yollar hazır geldiyse doğrudan DB'ye yaz.
        if image_path and thumb_path:
            abs_img_path = os.path.abspath(image_path)
            abs_thumb_path = os.path.abspath(thumb_path)
            try:
                cur = self.conn.cursor()
                cur.execute(
                    """INSERT INTO photos (athlete_id, date, competition, image_path, thumb_path)
                       VALUES (?, ?, ?, ?, ?)""",
                    (athlete_id, date, competition, self.store_path_for_db(abs_img_path), self.store_path_for_db(abs_thumb_path))
                )
                self.conn.commit()
                return True
            except Exception as e:
                print(f"Kayıt Hatası: {e}")
                return False

        from PyQt6.QtGui import QPixmap
        from PyQt6.QtCore import Qt

        # Eski akış: dosyadan/bytes'tan oku, kaydet ve thumbnail üret.
        pix = QPixmap()
        if image_data:
            pix.loadFromData(image_data)
        elif file_path:
            pix.load(file_path)
        
        if pix.isNull(): return False

        unique_name = f"{athlete_id}_{uuid.uuid4().hex}.png"
        
        _, orig_dir, thumb_dir = self.get_image_dirs()

        abs_img_path = os.path.abspath(os.path.join(orig_dir, unique_name))
        abs_thumb_path = os.path.abspath(os.path.join(thumb_dir, unique_name))

        try:
            pix.save(abs_img_path, "PNG")

            thumb = pix.scaled(600, 900, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            thumb.save(abs_thumb_path, "PNG")

            cur = self.conn.cursor()
            cur.execute("""INSERT INTO photos (athlete_id, date, competition, image_path, thumb_path)
                           VALUES (?, ?, ?, ?, ?)""",
                        (athlete_id, date, competition, self.store_path_for_db(abs_img_path), self.store_path_for_db(abs_thumb_path)))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Kayıt Hatası: {e}")
            return False


    def delete_photo(self, photo_id):
        """
        Hem veritabanından hem de diskten siler.
        """
        cur = self.conn.cursor()
        
        # Önce dosya yollarını al
        cur.execute("SELECT image_path, thumb_path FROM photos WHERE id=?", (photo_id,))
        row = cur.fetchone()

        if row:
            img_path, thumb_path = row
            img_path = self.resolve_path_from_db(img_path, prefer_thumb=False)
            thumb_path = self.resolve_path_from_db(thumb_path, prefer_thumb=True)
            # Dosyaları diskten sil
            try:
                if os.path.exists(img_path): os.remove(img_path)
                if os.path.exists(thumb_path): os.remove(thumb_path)
            except Exception as e:
                print(f"Dosya silme hatası: {e}")

        # DB'den sil
        cur.execute("DELETE FROM photos WHERE id=?", (photo_id,))
        self.conn.commit()

    def update_photo_info(self, photo_id, new_date, new_comp):
        cur = self.conn.cursor()

        # mevcut bilgileri al
        cur.execute("SELECT athlete_id, date, competition FROM photos WHERE id=?", (photo_id,))
        row = cur.fetchone()
        if not row:
            return
        athlete_id, old_date, old_comp = row

        # hedef grupta en sona rank ver
        new_rank = self.get_next_rank(athlete_id, new_date, new_comp)

        cur.execute("""UPDATE photos
                       SET date=?, competition=?, rank=?
                       WHERE id=?""",
                    (new_date, new_comp, new_rank, photo_id))
        self.conn.commit()

        # iki grubu da normalize et
        self.normalize_ranks(athlete_id, old_date, old_comp)
        self.normalize_ranks(athlete_id, new_date, new_comp)

    def move_photo_to_athlete(self, photo_id, new_athlete_id, new_date, new_comp):
        cur = self.conn.cursor()
        cur.execute("SELECT athlete_id, date, competition FROM photos WHERE id=?", (photo_id,))
        row = cur.fetchone()
        if not row:
            return False

        old_athlete, old_date, old_comp = row
        target_date = new_date or old_date
        target_comp = new_comp or old_comp

        new_rank = self.get_next_rank(new_athlete_id, target_date, target_comp)
        cur.execute("""UPDATE photos
                       SET athlete_id=?, date=?, competition=?, rank=?
                       WHERE id=?""",
                    (new_athlete_id, target_date, target_comp, new_rank, photo_id))
        self.conn.commit()

        # Yarışma bilgisini global listede güncel tut
        self.upsert_global_competition(target_comp, target_date)
        self.ensure_competition_record(target_comp, target_date)

        # Normalize eski ve yeni gruplar
        self.normalize_ranks(old_athlete, old_date, old_comp)
        self.normalize_ranks(new_athlete_id, target_date, target_comp)
        return True


    def get_photos(self, athlete_id):
        cur = self.conn.cursor()
        # image_data YERİNE image_path ve thumb_path çekiyoruz
        cur.execute("SELECT date, competition, thumb_path, id, rank, image_path FROM photos WHERE athlete_id=?", (athlete_id,))
        rows = []
        for date_str, comp, thumb_path, pid, rank, image_path in cur.fetchall():
            resolved_img, resolved_thumb = self.resolve_photo_paths(pid, image_path, thumb_path)
            rows.append((date_str, comp, resolved_thumb, pid, rank, resolved_img))
        return rows
    
    def get_comp_date_mapping(self, athlete_id):
        cur = self.conn.cursor()
        cur.execute("SELECT competition, date FROM photos WHERE athlete_id=?", (athlete_id,))
        rows = cur.fetchall()
        return {r[0]:r[1] for r in rows if r[0]}, {r[1]:r[0] for r in rows if r[1]}

    def move_photo(self, photo_id, direction, athlete_id, date_str, competition):
        cur = self.conn.cursor()
        cur.execute("""SELECT id, rank FROM photos
                       WHERE athlete_id=? AND date=? AND competition=?
                       ORDER BY rank ASC, id ASC""",
                    (athlete_id, date_str, competition))
        items = list(cur.fetchall())

        current_index = next((i for i, (pid, _) in enumerate(items) if pid == photo_id), -1)
        if current_index == -1:
            return

        target = current_index + direction
        if target < 0 or target >= len(items):
            return

        items[current_index], items[target] = items[target], items[current_index]

        # 1-based rank
        for new_rank, (pid, _) in enumerate(items, start=1):
            cur.execute("UPDATE photos SET rank=? WHERE id=?", (new_rank, pid))

        self.conn.commit()


# --- PUANLAMA DİYALOĞU ---
class ScoringDialog(QDialog):
    score_saved = pyqtSignal()

    def __init__(self, db, athlete_id, athlete_name, division, parent=None):
        super().__init__(parent)
        self.db = db
        self.athlete_id = athlete_id
        self.division = division
        rows = self.db.get_criteria(division)  # [(criterion, weight), ...]
        self.criteria = {crit: float(w) for crit, w in rows}
        self.inputs = {}
        
        self.setWindowTitle(f"Puanla: {athlete_name} ({division})")
        self.resize(400, 500)
        
        layout = QVBoxLayout()
        form_layout = QFormLayout()
        
        layout.addWidget(QLabel("Değerlendirme Kriterleri (1-10)"))
        
        for crit, weight in self.criteria.items():
            spin = QSpinBox()
            spin.setRange(0, 10)
            spin.setValue(5)
            self.inputs[crit] = (spin, weight)
            form_layout.addRow(f"{crit} (x{weight}):", spin)
            
        layout.addLayout(form_layout)
        
        self.lbl_result = QLabel("Skor: 0.0")
        self.lbl_result.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.lbl_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_result)
        
        btn_calc = QPushButton("Hesapla")
        btn_calc.clicked.connect(self.calc)
        layout.addWidget(btn_calc)
        
        btn_save = QPushButton("Kaydet ve Çık")
        btn_save.clicked.connect(self.save)
        btn_save.setStyleSheet("background-color: #4CAF50; color: white;")
        layout.addWidget(btn_save)
        
        self.setLayout(layout)
        self.calc()

    def calc(self):
        total = 0.0
        max_total = 0.0

        for crit, (spin, weight) in self.inputs.items():
            total += spin.value() * weight
            max_total += 10 * weight

        self.total_raw = int(round(total))
        self.max_total = int(round(max_total))
        norm10 = (total / max_total * 10) if max_total > 0 else 0
        self.final_val = round(norm10, 2)

        # İstersen sadece 599/770 göster; ben ikisini birlikte veriyorum:
        self.lbl_result.setText(f"Total: {self.total_raw} / {self.max_total}   (Norm: {self.final_val} / 10)")


    def save(self):
        self.calc()
        self.db.update_athlete_score(self.athlete_id, self.total_raw)
        self.score_saved.emit()
        self.accept()


# --- FOTOĞRAF DETAY DİYALOĞU ---
# --- FOTOĞRAF DETAY DİYALOĞU (GÜNCELLENMİŞ) ---
class PhotoDetailsDialog(QDialog):
    saved_successfully = pyqtSignal()

    def __init__(self, db, athlete_id, athlete_name="", file_paths=None, current_date=None, current_comp=None, parent=None):
        super().__init__(parent)
        self.db = db
        self.athlete_id = athlete_id
        self.athlete_name = athlete_name
        self.file_paths = file_paths
        
        self.is_manual_mode = (file_paths is None)

        self.setWindowTitle(f"Toplu Fotoğraf Yükle - {self.athlete_name}")
        self.resize(600, 750)
        
        # Eski harita sistemini koruyoruz ama asıl veriyi tablodan çekeceğiz
        self.comp_map = self.db.get_global_comp_date_mapping()
    
        self.init_ui()
        
        # --- YENİ: Yarışma Listesini Yükle ---
        self.load_competitions() 

        # Eğer düzenleme moduysa ve mevcut veri geldiyse kutuları doldur
        if current_comp: 
            self.comp_combo.setCurrentText(current_comp)
        
        if current_date:
            self.date_edit.setDate(QDate.fromString(current_date, "yyyy-MM-dd"))
        elif current_comp: # Tarih yok ama yarışma varsa, veritabanından bulmayı dene
            self.on_comp_text_changed(current_comp)

    def init_ui(self):
        layout = QVBoxLayout()
        
        # --- 1. SPORCU İSMİ ---
        lbl_athlete = QLabel(f"👤 {self.athlete_name}")
        lbl_athlete.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_athlete.setStyleSheet("background-color: #00AEEF; color: white; font-weight: bold; font-size: 16px; padding: 10px; border-radius: 5px;")
        layout.addWidget(lbl_athlete)
        
        # --- 2. FOTOĞRAF KUYRUĞU ---
        self.lbl_status = QLabel(f"📸 Yüklenecek Fotoğraflar (Sürükle-Bırak veya Yapıştır)")
        layout.addWidget(self.lbl_status)
        
        self.list_photos = QListWidget()
        self.list_photos.setViewMode(QListWidget.ViewMode.IconMode)
        self.list_photos.setIconSize(QSize(180, 180))
        self.list_photos.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list_photos.setSpacing(10)
        self.list_photos.setMinimumHeight(250)
        self.list_photos.setStyleSheet("""
            QListWidget { background-color: #222; border: 2px dashed #555; }
            QListWidget::item { border: 1px solid #444; background-color: #333; }
            QListWidget::item:selected { border: 2px solid #00AEEF; }
        """)
        self.list_photos.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list_photos.customContextMenuRequested.connect(self.on_list_photos_context_menu)
        layout.addWidget(self.list_photos)

        # Kontrol Butonları
        list_tools = QHBoxLayout()
        btn_remove = QPushButton("❌ Seçileni Çıkar")
        btn_remove.clicked.connect(self.remove_selected_photo)
        
        btn_clear = QPushButton("🗑️ Listeyi Temizle")
        btn_clear.clicked.connect(self.list_photos.clear)

        list_tools.addStretch()
        list_tools.addWidget(btn_remove)
        list_tools.addWidget(btn_clear)
        layout.addLayout(list_tools)

        # Manuel Mod Butonları
        img_btn_layout = QHBoxLayout()
        if self.is_manual_mode:
            self.btn_paste = QPushButton("📋 Yapıştır (Ctrl+V)")
            self.btn_paste.clicked.connect(self.paste_from_clipboard)
            self.btn_paste.setStyleSheet("padding: 8px; font-weight: bold;")
            
            self.btn_browse = QPushButton("📂 Dosya Seç (Çoklu)")
            self.btn_browse.clicked.connect(self.browse_file)
            self.btn_browse.setStyleSheet("padding: 8px; font-weight: bold;")
            
            img_btn_layout.addWidget(self.btn_paste)
            img_btn_layout.addWidget(self.btn_browse)
        layout.addLayout(img_btn_layout)

        # --- 3. YARIŞMA VE TARİH (GÜNCELLENMİŞ) ---
        layout.addWidget(QLabel("🏆 Yarışma / Kategori Adı:"))
        
        comp_layout = QHBoxLayout()
        
        # AKILLI COMBOBOX
        self.comp_combo = QComboBox()
        self.comp_combo.setEditable(True)
        self.comp_combo.setPlaceholderText("Yarışma adı arayın...")
        self.comp_combo.setStyleSheet("padding: 6px;")
        
        # Completer (Otomatik Tamamlama) Ayarları
        completer = QCompleter(self.comp_combo.model(), self.comp_combo)
        completer.setFilterMode(Qt.MatchFlag.MatchContains) # İçinde geçeni bul
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.comp_combo.setCompleter(completer)
        
        # Sinyaller
        # 1. Listeden bir şey seçilirse -> Tarihi güncelle
        self.comp_combo.currentIndexChanged.connect(self.on_comp_selected_from_list)
        # 2. Yazı elle değiştirilirse -> Belki eski map'te vardır diye kontrol et
        self.comp_combo.editTextChanged.connect(self.on_comp_text_changed)
        
        comp_layout.addWidget(self.comp_combo, 1)
        
        self.btn_del_comp_name = QPushButton("🗑️")
        self.btn_del_comp_name.setFixedWidth(40)
        self.btn_del_comp_name.setToolTip("Seçili yarışmayı veritabanından siler")
        self.btn_del_comp_name.clicked.connect(self.delete_global_comp_name)
        comp_layout.addWidget(self.btn_del_comp_name)
        
        layout.addLayout(comp_layout)

        layout.addWidget(QLabel("📅 Tarih:"))
        date_row = QHBoxLayout()

        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("yyyy-MM-dd")
        self.date_edit.setDate(QDate.currentDate())
        line_edit = self.date_edit.lineEdit()
        if line_edit:
            line_edit.textChanged.connect(self.on_date_text_changed)
        self.date_edit.dateChanged.connect(self.sync_date_to_text)
        try:
            # Force the calendar widget to English regardless of OS locale
            self.date_edit.calendarWidget().setLocale(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))
        except Exception:
            pass
        date_row.addWidget(self.date_edit)

        self.date_text_input = QLineEdit()
        self.date_text_input.setPlaceholderText("Örn: July 20, 2021 veya 2021-07-20")
        self.date_text_input.setText(self.date_edit.date().toString("yyyy-MM-dd"))
        self.date_text_input.textChanged.connect(self.on_free_text_date_changed)
        date_row.addWidget(self.date_text_input)

        layout.addLayout(date_row)

        self.updating = False

        # --- 4. KAYDETME BUTONLARI ---
        btn_layout = QHBoxLayout()
        
        btn_cancel = QPushButton("Kapat")
        btn_cancel.clicked.connect(self.reject)

        self.btn_save_next = QPushButton("💾 Kaydet ve Devam Et")
        self.btn_save_next.clicked.connect(self.save_and_next)
        self.btn_save_next.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 12px; font-size: 13px;")

        self.btn_save = QPushButton("✅ Kaydet ve Kapat")
        self.btn_save.clicked.connect(self.save_and_close)
        self.btn_save.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 12px; font-size: 13px;")
        
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(self.btn_save_next)
        btn_layout.addWidget(self.btn_save)
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        
        self.final_date = None
        self.final_comp = None

        if self.file_paths:
            for f in self.file_paths:
                self.add_photo_to_list(f)

    # --- YENİ: YARIŞMA VERİLERİNİ YÜKLEME ---
    def load_competitions(self):
        """Veritabanındaki competitions tablosundan ve eski map'ten verileri çeker."""
        self.comp_combo.clear()
        self.comp_combo.addItem("", "") # Boş seçenek

        # 1. Önce CSV ile yüklenen Ana Listeyi Çek (competitions tablosu)
        try:
            cur = self.db.conn.cursor()
            # Tablo yoksa oluştur (Garanti olsun)
            cur.execute("""CREATE TABLE IF NOT EXISTS competitions (id INTEGER PRIMARY KEY, name TEXT, date_str TEXT)""")
            
            cur.execute("SELECT name, date_str FROM competitions ORDER BY date_str DESC")
            rows = cur.fetchall()
            for name, date_str in rows:
                self.comp_combo.addItem(name, date_str) # date_str'yi gizli veri olarak ekle
        except: pass

        # 2. Eski sistemdeki kayıtları da ekle (Eğer ana listede yoksa)
        # Bu sayede kullanıcının elle girdiği eski yarışmalar kaybolmaz
        existing_items = set([self.comp_combo.itemText(i) for i in range(self.comp_combo.count())])
        
        for name, date_str in self.comp_map.items():
            if name not in existing_items:
                self.comp_combo.addItem(name, date_str)

    # --- YENİ: SEÇİM OLAYLARI ---
    def on_comp_selected_from_list(self, index):
        """Listeden seçilince tarihi otomatik getir."""
        if self.updating: return
        
        # Combobox'ın gizli verisinde (UserRole) tarih var mı?
        date_str = self.comp_combo.itemData(index)
        
        if date_str:
            self.updating = True
            self.date_edit.setDate(QDate.fromString(date_str, "yyyy-MM-dd"))
            self.updating = False

    def on_comp_text_changed(self, text):
        """Yazı elle değişince eski map'i kontrol et."""
        if self.updating: return
        text = text.strip()

        # Eski map'te var mı?
        if text in self.comp_map:
            self.updating = True
            self.date_edit.setDate(QDate.fromString(self.comp_map[text], "yyyy-MM-dd"))
            self.updating = False

    def on_date_text_changed(self, text):
        """Serbest tarih metnini (örn. 'July 20, 2021') otomatik olarak yyyy-MM-dd'e çevir."""
        self._apply_freeform_date(text)

    def on_free_text_date_changed(self, text):
        """Yan taraftaki metin kutusundan gelen tarihi işle."""
        self._apply_freeform_date(text)

    def sync_date_to_text(self, qdate):
        """Takvimdeki değişikliği metin kutusuna yansıt."""
        if self.updating:
            return
        self.updating = True
        self.date_text_input.setText(qdate.toString("yyyy-MM-dd"))
        self.updating = False

    def _apply_freeform_date(self, text):
        if self.updating:
            return
        parsed = self._parse_freeform_date(text)
        if parsed and parsed.isValid():
            self.updating = True
            self.date_edit.setDate(parsed)
            self.date_text_input.setText(parsed.toString("yyyy-MM-dd"))
            self.updating = False

    def _parse_freeform_date(self, text):
        """İngilizce ay isimleri ve yaygın ayraçlarla girilen tarihleri yakalar."""
        if not text:
            return None
        txt = text.strip()
        if not txt:
            return None

        # Önce doğrudan yaygın formatları dene
        for fmt in ("yyyy-MM-dd", "dd.MM.yyyy", "dd/MM/yyyy", "MM/dd/yyyy", "MM-dd-yyyy", "dd-MM-yyyy"):
            qd = QDate.fromString(txt, fmt)
            if qd.isValid():
                return qd

        cleaned = re.sub(r"[.,/\\-]", " ", txt)
        tokens = [t for t in cleaned.split() if t]
        if len(tokens) != 3:
            return None

        month_map = {
            "january": 1, "jan": 1,
            "february": 2, "feb": 2,
            "march": 3, "mar": 3,
            "april": 4, "apr": 4,
            "may": 5,
            "june": 6, "jun": 6,
            "july": 7, "jul": 7,
            "august": 8, "aug": 8,
            "september": 9, "sep": 9, "sept": 9,
            "october": 10, "oct": 10,
            "november": 11, "nov": 11,
            "december": 12, "dec": 12,
        }

        def parse_day(tok):
            tok = re.sub(r"(st|nd|rd|th)$", "", tok.lower())
            return int(tok) if tok.isdigit() else None

        def parse_year(tok):
            if not tok.isdigit():
                return None
            y = int(tok)
            if y < 100:  # 21 -> 2021
                y += 2000 if y < 50 else 1900
            return y

        def build(m_tok, d_tok, y_tok):
            m = month_map.get(m_tok.lower())
            d = parse_day(d_tok)
            y = parse_year(y_tok)
            if m and d and y:
                candidate = QDate(y, m, d)
                if candidate.isValid():
                    return candidate
            return None

        # Ay başta (July 20 2021) veya ortada (20 July 2021)
        return build(tokens[0], tokens[1], tokens[2]) or build(tokens[1], tokens[0], tokens[2])

    # --- ESKİ KODLARIN DEVAMI (Aynen Korumalıyız) ---
    def process_saving_batch(self):
            count = self.list_photos.count()
            if count == 0: return False

            self.final_date = self.date_edit.date().toString("yyyy-MM-dd")
            self.final_comp = self.comp_combo.currentText().strip()
            if not self.final_comp:
                QMessageBox.warning(self, "Uyarı", "Lütfen yarışma adı girin.")
                return False
            
            # Hem eski map'e kaydet (Hızlı erişim için) hem de competitions tablosuna (Gelecek için)
            self.db.upsert_global_competition(self.final_comp, self.final_date)
            
            # Yeni bir yarışma ismi girdiysek, bir dahaki sefere listede çıksın diye competitions tablosuna ekleyelim
            try:
                cur = self.db.conn.cursor()
                cur.execute("INSERT OR IGNORE INTO competitions (name, date_str) VALUES (?, ?)", (self.final_comp, self.final_date))
                self.db.conn.commit()
            except: pass

            # --- AI ve Kayıt İşlemleri (BURASI AYNI) ---
            setting_val = self.db.get_setting("ai_auto_apply", "0")
            auto_upscale = (setting_val == "1")
            
            # Model yolu bulma
            import os
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "Real-ESRGAN-x4plus.onnx")

            # Fotoğraf klasörleri (kullanıcının seçtiği)
            img_base, orig_dir, thumb_dir = self.db.get_image_dirs()

            upscaler = None
            if auto_upscale:
                sc = int(self.db.get_setting("ai_scale", "3"))
                st = float(self.db.get_setting("ai_strength", "0.70"))
                if os.path.exists(model_path):
                    self.lbl_status.setText(f"🚀 AI Hazırlanıyor...")
                    QApplication.processEvents()
                    try:
                        upscaler = RealESRGANOnnxUpscaler(model_path, target_scale=sc, ai_strength=st)
                    except: pass

            saved_count = 0
            for i in range(count):
                item = self.list_photos.item(i)
                file_path = item.data(Qt.ItemDataRole.UserRole)
                
                if file_path:
                    # AI Kısmı
                    processed = False
                    if upscaler:
                        self.lbl_status.setText(f"✨ AI İşleniyor ({i+1}/{count})...")
                        self.lbl_status.setStyleSheet("color: #E91E63; font-weight: bold;") 
                        QApplication.processEvents()
                        try:
                            pix = QPixmap(file_path)
                            new_pix = upscaler.upscale_pixmap(pix)
                            if new_pix:
                                # Burada veritabanına değil DOSYAYA kaydedeceğiz artık (Yeni sistem)
                                # Ama senin kodda DB'ye binary atıyordu, onu yeni sisteme (Path bazlı) çevirelim mi?
                                # Sen "Path bazlı" sisteme geçtin. O yüzden burayı güncellemeliyiz.
                                
                                # 1. Kalıcı Klasöre Kaydet
                                import uuid, shutil
                                os.makedirs(orig_dir, exist_ok=True)
                                os.makedirs(thumb_dir, exist_ok=True)
                                
                                ext = ".png" # AI çıktısı PNG olur
                                unique_name = f"{uuid.uuid4()}{ext}"
                                target_orig = os.path.join(orig_dir, unique_name)
                                target_thumb = os.path.join(thumb_dir, unique_name)

                                # Orijinali (AI'lı) kaydet
                                new_pix.save(target_orig, "PNG")
                                
                                # Thumbnail oluştur
                                thumb = new_pix.scaled(600, 900, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                                thumb.save(target_thumb, "PNG")
                                
                                # DB'ye Path Kaydet
                                self.db.add_photo(self.athlete_id, self.final_date, self.final_comp, thumb_path=target_thumb, image_path=target_orig)
                                processed = True
                        except Exception as e:
                            print(f"AI Hatası: {e}")

                    if not processed:
                        # Normal Kayıt (AI Yoksa)
                        import uuid, shutil
                        os.makedirs(orig_dir, exist_ok=True)
                        os.makedirs(thumb_dir, exist_ok=True)
                        
                        ext = os.path.splitext(file_path)[1]
                        unique_name = f"{uuid.uuid4()}{ext}"
                        target_orig = os.path.join(orig_dir, unique_name)
                        target_thumb = os.path.join(thumb_dir, unique_name)
                        
                        shutil.copy2(file_path, target_orig)
                        
                        pix = QPixmap(target_orig)
                        thumb = pix.scaled(600, 900, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                        thumb.save(target_thumb)
                        
                        self.db.add_photo(self.athlete_id, self.final_date, self.final_comp, thumb_path=target_thumb, image_path=target_orig)
                    
                    saved_count += 1
            
            self.lbl_status.setStyleSheet("color: #ddd;")
            return saved_count > 0

    def save_and_close(self):
        if self.process_saving_batch():
            self.saved_successfully.emit()
            self.accept()

    def save_and_next(self):
        if self.process_saving_batch():
            self.saved_successfully.emit()
            self.list_photos.clear()
            self.lbl_status.setText(f"✅ Kaydedildi! Sıradaki fotoları yapıştırın... (Yarışma: {self.final_comp})")
            self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")

    # --- DİĞER FONKSİYONLAR (Aynı) ---
    def on_list_photos_context_menu(self, pos):
        menu = QMenu(self)
        act_paste = menu.addAction("Yapıştır")
        chosen = menu.exec(self.list_photos.mapToGlobal(pos))
        if chosen == act_paste:
            self.paste_from_clipboard()

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.StandardKey.Paste):
            self.paste_from_clipboard()
        else:
            super().keyPressEvent(event)

    def paste_from_clipboard(self):
        clipboard = QApplication.clipboard()
        mime = clipboard.mimeData()
        if mime.hasImage():
            self.save_temp_and_add(clipboard.pixmap())
        elif mime.hasUrls():
            for url in mime.urls():
                if url.isLocalFile():
                    self.add_photo_to_list(url.toLocalFile())

    def browse_file(self):
        fpaths, _ = QFileDialog.getOpenFileNames(self, "Fotoğrafları Seç", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if fpaths:
            for f in fpaths:
                self.add_photo_to_list(f)

    def save_temp_and_add(self, pixmap):
        if pixmap.isNull(): return
        if not os.path.exists("temp_cache"): os.makedirs("temp_cache")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join("temp_cache", f"paste_{ts}.png")
        pixmap.save(path, "PNG")
        self.add_photo_to_list(path)

    def add_photo_to_list(self, file_path):
        pixmap = QPixmap(file_path)
        if pixmap.isNull(): return
        icon = QIcon(pixmap)
        item = QListWidgetItem(icon, "")
        item.setData(Qt.ItemDataRole.UserRole, file_path)
        self.list_photos.addItem(item)
        self.list_photos.scrollToBottom()
        self.lbl_status.setText("📸 Yüklenecek Fotoğraflar (Sürükle-Bırak veya Yapıştır)")
        self.lbl_status.setStyleSheet("color: #ddd;")

    def remove_selected_photo(self):
        for item in self.list_photos.selectedItems():
            row = self.list_photos.row(item)
            self.list_photos.takeItem(row)

    def delete_global_comp_name(self):
        current_text = self.comp_combo.currentText().strip()
        if not current_text: return
        
        # Hem yeni tablodan hem eski map'ten silmeyi teklif et
        from PyQt6.QtWidgets import QMessageBox
        reply = QMessageBox.question(self, 'Sil', f"'{current_text}' veritabanından silinsin mi?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            # 1. Competitions tablosundan sil
            try:
                cur = self.db.conn.cursor()
                cur.execute("DELETE FROM competitions WHERE name = ?", (current_text,))
                self.db.conn.commit()
            except: pass
            
            # 2. Eski map'ten sil
            self.db.delete_global_competition(current_text)
            
            # 3. Listeden kaldır
            idx = self.comp_combo.findText(current_text)
            if idx >= 0:
                self.comp_combo.removeItem(idx)
            
            self.comp_combo.setCurrentIndex(0)

class FullImageViewer(QDialog):
    def __init__(self, image_path, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1000, 800)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        
        self.view = ZoomableGraphicsView(QPixmap(image_path))
        self.view.setStyleSheet("background-color: #000;")
        layout.addWidget(self.view)
        
        btn_close = QPushButton("X")
        btn_close.setFixedSize(30, 30)
        btn_close.setStyleSheet("background:red; color:white; font-weight:bold; border-radius:15px;")
        btn_close.clicked.connect(self.close)
        
        # Butonu sağ üste koy (Overlay)
        btn_close.setParent(self)
        btn_close.move(self.width() - 40, 10)


class EditPhotoDialog(QDialog):
    def __init__(self, db, current_photo_id, current_athlete_id, current_comp, current_date, parent=None):
        super().__init__(parent)
        self.db = db
        self.current_photo_id = current_photo_id
        self.final_athlete_id = current_athlete_id
        self.final_comp = current_comp
        self.final_date = current_date
        self.comp_map = self.db.get_global_comp_date_mapping()
        self.updating = False

        self.setWindowTitle("Fotoğrafı Düzenle / Taşı")
        self.resize(420, 220)

        layout = QVBoxLayout(self)

        form = QFormLayout()

        # Sporcu seçimi
        self.cmb_athlete = QComboBox()
        self.athlete_ids = []
        self.cmb_athlete.setEditable(True)
        line_edit = self.cmb_athlete.lineEdit()
        if line_edit:
            line_edit.setPlaceholderText("İsim ara...")
            try:
                line_edit.setClearButtonEnabled(True)
            except Exception:
                pass
        for aid, name, division in self.db.get_all_athletes():
            label = f"{name} ({division})"
            self.cmb_athlete.addItem(label, aid)
            self.athlete_ids.append(aid)
            if aid == current_athlete_id:
                self.cmb_athlete.setCurrentIndex(self.cmb_athlete.count() - 1)

        # Arama / otomatik tamamlama
        athlete_completer = QCompleter(self.cmb_athlete.model(), self.cmb_athlete)
        athlete_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        athlete_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.cmb_athlete.setCompleter(athlete_completer)

        form.addRow("🏃 Sporcu:", self.cmb_athlete)

        # Yarışma adı
        self.cmb_comp = QComboBox()
        self.cmb_comp.setEditable(True)
        le = self.cmb_comp.lineEdit()
        if le:
            le.setPlaceholderText("Yarışma ara/seç")
            try:
                le.setClearButtonEnabled(True)
            except Exception:
                pass

        # Listeyi yükle
        comps = self.db.get_all_competitions()
        for comp in comps:
            self.cmb_comp.addItem(comp)
        if current_comp and self.cmb_comp.findText(current_comp) == -1:
            self.cmb_comp.addItem(current_comp)
        if current_comp:
            self.cmb_comp.setCurrentText(current_comp)

        comp_completer = QCompleter(self.cmb_comp.model(), self.cmb_comp)
        comp_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        comp_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.cmb_comp.setCompleter(comp_completer)
        self.cmb_comp.editTextChanged.connect(self.on_comp_text_changed)

        form.addRow("🏆 Yarışma:", self.cmb_comp)

        # Tarih
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("yyyy-MM-dd")
        self.date_edit.setDate(QDate.fromString(current_date, "yyyy-MM-dd") if current_date else QDate.currentDate())
        form.addRow("📅 Tarih:", self.date_edit)

        layout.addLayout(form)

        btns = QHBoxLayout()
        btn_cancel = QPushButton("İptal")
        btn_save = QPushButton("Kaydet")
        btn_cancel.clicked.connect(self.reject)
        btn_save.clicked.connect(self.on_save)
        btns.addStretch()
        btns.addWidget(btn_cancel)
        btns.addWidget(btn_save)
        layout.addLayout(btns)

        # Mevcut değerlerden tarih doldur
        if not current_date and current_comp:
            self.on_comp_text_changed(current_comp)

    def on_comp_text_changed(self, text):
        if self.updating:
            return
        text = (text or "").strip()
        date_str = self.comp_map.get(text)
        if date_str:
            self.updating = True
            self.date_edit.setDate(QDate.fromString(date_str, "yyyy-MM-dd"))
            self.updating = False

    def on_save(self):
        comp = self.cmb_comp.currentText().strip()
        if not comp:
            QMessageBox.warning(self, "Eksik bilgi", "Yarışma adı boş olamaz.")
            return

        self.final_athlete_id = self.cmb_athlete.currentData()
        self.final_comp = comp
        self.final_date = self.date_edit.date().toString("yyyy-MM-dd")
        self.accept()


# --- FOTO WIDGET ---
class PhotoWidget(QWidget):
    # image_path parametresini de alıyoruz
    def __init__(self, photo_id, thumb_path, date_str, competition, rank, size, main_window, athlete_name="", image_path=""):
        super().__init__()
        self.photo_id = photo_id
        self.thumb_path = thumb_path
        self.image_path = image_path 
        self.date_str = date_str
        self.competition = competition
        self.rank = rank
        self.main_window = main_window
        self.athlete_name = athlete_name

        # --- OPTİMİZASYON 1: BAŞLANGIÇTA HEP THUMBNAIL YÜKLE ---
        # HD olsa bile ilk açılışta thumbnail gösteriyoruz ki arayüz donmasın.
        self.source_pixmap = QPixmap(self.thumb_path)
        
        # Hafıza Kontrolü
        self.is_hd_target = (self.photo_id in self.main_window.hd_memory)
        self.is_high_res = False # Şu an fiziksel olarak SD yüklü

        # Varsayılan Stil (SD)
        self.border_style = "border: 1px solid #333; background-color: #252525; border-radius: 4px;"
        btn_text = "HD"
        self.btn_style = "background-color: rgba(0,0,0,150); color: #bbb; border: 1px solid #555; border-radius: 3px; font-weight: bold; font-size: 10px;"

        # Eğer HD olması gerekiyorsa, BUTONLARI hemen yeşil yap (Kullanıcı HD olduğunu anlasın)
        # Ama resmi henüz yükleme!
        if self.is_hd_target:
            btn_text = "..." # Yükleniyor manasında
            self.btn_style = "background-color: #2E7D32; color: white; border: 1px solid #1B5E20; border-radius: 3px; font-weight: bold; font-size: 10px;"
            self.border_style = "border: 2px solid #4CAF50; background-color: #252525; border-radius: 4px;"

        # Arayüz Kurulumu
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5,5,5,5)
        main_layout.setSpacing(2)

        img_container = QWidget()
        img_layout = QGridLayout(img_container)
        img_layout.setContentsMargins(0,0,0,0)

        self.lbl_img = QLabel()
        self.lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_img.setStyleSheet(self.border_style)

        self.btn_toggle = QPushButton(btn_text)
        self.btn_toggle.setFixedSize(30, 20)
        self.btn_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_toggle.setStyleSheet(self.btn_style)
        self.btn_toggle.clicked.connect(self.toggle_resolution)
        
        img_layout.addWidget(self.lbl_img, 0, 0)
        img_layout.addWidget(self.btn_toggle, 0, 0, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)

        self.chk_compare = QCheckBox("Seç")
        self.chk_compare.setStyleSheet("color: #bbb; font-size: 11px;")
        self.chk_compare.stateChanged.connect(self.on_compare_changed)

        main_layout.addWidget(img_container)
        main_layout.addWidget(self.chk_compare, 0, Qt.AlignmentFlag.AlignCenter)
        self.setLayout(main_layout)
        
        self.set_info_tooltip()
        self.update_size(size) # İlk çizim (Thumbnail ile)

        # --- OPTİMİZASYON 2: GECİKMELİ YÜKLEME ---
        # Eğer HD olması gerekiyorsa, arayüz çizildikten 10ms sonra ağır işlemi yap.
        if self.is_hd_target:
            QTimer.singleShot(10, self.load_hd_delayed)

    def mouseDoubleClickEvent(self, event):
        # Çift tıklayınca da HD moda geçsin/çıksın
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle_resolution()
        super().mouseDoubleClickEvent(event)

    def toggle_resolution(self):
        if not self.is_high_res:
            # HD'ye Geç
            if os.path.exists(self.image_path):
                # Kullanıcıya beklemesini hissettirmemek için imleci değiştir
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                
                self.source_pixmap = QPixmap(self.image_path)
                self.is_high_res = True
                self.main_window.hd_memory.add(self.photo_id)
                
                self.btn_toggle.setText("SD")
                self.btn_toggle.setStyleSheet("background-color: #4CAF50; color: white; border: 1px solid #2E7D32; border-radius: 3px; font-weight: bold; font-size: 10px;")
                self.lbl_img.setStyleSheet("border: 2px solid #4CAF50; background-color: #252525; border-radius: 4px;")
                
                QApplication.restoreOverrideCursor()
        else:
            # SD'ye Geç
            if os.path.exists(self.thumb_path):
                self.source_pixmap = QPixmap(self.thumb_path)
                self.is_high_res = False
                self.main_window.hd_memory.discard(self.photo_id)
                
                self.btn_toggle.setText("HD")
                self.btn_toggle.setStyleSheet("background-color: rgba(0,0,0,150); color: #bbb; border: 1px solid #555; border-radius: 3px; font-weight: bold; font-size: 10px;")
                self.btn_toggle.setStyleSheet("QPushButton:hover { background-color: #fff; color: #000; }")
                self.lbl_img.setStyleSheet("border: 1px solid #333; background-color: #252525; border-radius: 4px;")
        
        self.update_size(self.main_window.thumbnail_size)
        self.set_info_tooltip()  # <--- YENİ EKLENEN SATIR

    def show_full_image(self):
        # Tam ekran görüntüleyiciyi aç
        viewer = FullImageViewer(self.image_path, self.athlete_name, self.main_window)
        viewer.show()

    def set_info_tooltip(self):
        """O anki aktif resmin (SD veya HD) bilgilerini gösterir."""
        import os
        from PyQt6.QtGui import QImageReader

        # 1. Hangi yola bakacağımıza karar verelim
        if self.is_high_res:
            target_path = self.image_path
            mode_text = "<span style='color:#4CAF50;'>[HD] Orijinal</span>" # Yeşil yazı
        else:
            target_path = self.thumb_path
            mode_text = "<span style='color:#bbb;'>[SD] Önizleme</span>"   # Gri yazı

        # 2. Dosya boyutunu öğren
        try:
            size_bytes = os.path.getsize(target_path)
            if size_bytes < 1024:
                size_str = f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                size_str = f"{size_bytes / 1024:.1f} KB"
            else:
                size_str = f"{size_bytes / (1024 * 1024):.2f} MB"
        except:
            size_str = "Bilinmiyor"
            
        # 3. Çözünürlüğü öğren
        reader = QImageReader(target_path)
        sz = reader.size()
        w, h = sz.width(), sz.height()

        # 4. Tooltip HTML'ini oluştur
        tip = f"""
        <div style='background-color: #121212; color: #e0e0e0; padding: 8px; border: 1px solid #444;'>
            <b>{self.athlete_name}</b><br>
            {mode_text}<br>
            <span style='color:#00AEEF'>{w} x {h} px</span> | {size_str}
        </div>
        """
        self.setToolTip(tip)
        # Resmin üzerine gelince de çıksın istiyorsan:
        self.lbl_img.setToolTip(tip)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        # Menü stili (Dark Mode)
        menu.setStyleSheet("QMenu { background-color: #2b2b2b; color: white; border: 1px solid #444; } QMenu::item:selected { background-color: #00AEEF; }")
        
        action_copy = QAction("📋 Kopyala", self)
        action_copy.triggered.connect(self.copy_image_to_clipboard)
        menu.addAction(action_copy)

        action_save = QAction("💾 Farklı Kaydet...", self)
        action_save.triggered.connect(self.save_image_as_file)
        menu.addAction(action_save)

        menu.addSeparator()

        action_upscale = QAction("✨ AI İle Kaliteyi Artır", self)
        action_upscale.triggered.connect(self.perform_ai_upscale)
        menu.addAction(action_upscale)

        action_paste_here = QAction(f"📌 Bu Yarışmaya Yapıştır", self)
        action_paste_here.triggered.connect(lambda: self.main_window.paste_to_target(self.date_str, self.competition))
        menu.addAction(action_paste_here)
        
        menu.addSeparator()

        action_left = QAction("⬅️ Sola Taşı", self)
        action_left.triggered.connect(lambda: self.move_photo(-1))
        
        action_right = QAction("➡️ Sağa Taşı", self)
        action_right.triggered.connect(lambda: self.move_photo(1))
        
        action_edit = QAction("✏️ Düzenle / Taşı", self)
        action_edit.triggered.connect(self.edit_photo)
        
        action_del = QAction("🗑️ Sil", self)
        action_del.triggered.connect(self.delete_photo)
        
        menu.addAction(action_left)
        menu.addAction(action_right)
        menu.addSeparator()
        menu.addAction(action_edit)
        menu.addSeparator()
        menu.addAction(action_del)
        
        menu.exec(event.globalPos())

    def perform_ai_upscale(self):
        if not os.path.exists(model_path):
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Model Eksik", "Model dosyası bulunamadı.")
            return

        db = self.main_window.db 
        scale = int(db.get_setting("ai_scale", "3"))
        strength = float(db.get_setting("ai_strength", "0.70"))

        self.main_window._status.showMessage(f"⏳ AI İyileştirme...", 0)
        QApplication.processEvents()

        # Her zaman HD yoldan oku
        original_pix = QPixmap(self.image_path)
        upscaler = RealESRGANOnnxUpscaler(model_path=model_path, target_scale=scale, ai_strength=strength)
        new_pixmap = upscaler.upscale_pixmap(original_pix)

        if new_pixmap:
            # Diske yaz
            new_pixmap.save(self.image_path, "PNG")
            
            thumb = new_pixmap.scaled(600, 900, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            thumb.save(self.thumb_path, "PNG")

            self.main_window._status.showMessage("✨ Güncellendi!", 4000)
            self.main_window.refresh_gallery()

            # --- KRİTİK DÜZELTME ---
            # Eğer zaten HD moddaysak, ekrandaki kaynağı yeni HD resim yap
            if self.is_high_res:
                self.source_pixmap = new_pixmap
                # Buton ve çerçeve zaten yeşil/HD modunda, dokunmaya gerek yok.
            else:
                # SD moddaysak thumbnail'i güncelle (SD kalmaya devam et)
                self.source_pixmap = QPixmap(self.thumb_path)
                
            # Ekrana bas (Smooth modda)
            self.update_size(self.main_window.thumbnail_size, fast=False)
            
        else:
            self.main_window._status.showMessage("❌ Hata: AI Başarısız.", 4000)

    def copy_image_to_clipboard(self):
        if self.original_pixmap and not self.original_pixmap.isNull():
            QApplication.clipboard().setPixmap(self.original_pixmap)
            self.main_window._status.showMessage("✅ Panoya kopyalandı!", 2000)

    def save_image_as_file(self):
        if not self.original_pixmap or self.original_pixmap.isNull(): return
        default_name = f"{self.athlete_name}_{self.competition}_{self.date_str}.png"
        default_name = "".join(c for c in default_name if c.isalnum() or c in (' ', '.', '_', '-')).strip()
        file_path, _ = QFileDialog.getSaveFileName(self, "Resmi Kaydet", default_name, "Images (*.png *.jpg)")
        if file_path:
            self.original_pixmap.save(file_path)
            self.main_window._status.showMessage(f"✅ Kaydedildi: {file_path}", 3000)

    def force_resolution(self, target_hd):
        if target_hd and not self.is_high_res: self.toggle_resolution()
        elif not target_hd and self.is_high_res: self.toggle_resolution()

    def on_compare_changed(self, state):
        checked = (state == 2)
        # Karşılaştırma için ORİJİNAL YOLU gönderiyoruz
        data_packet = {
            "image_path": self.image_path, # ÖNEMLİ: Artık path gidiyor
            "athlete": self.athlete_name,
            "competition": self.competition,
            "date": self.date_str
        }
        self.main_window.toggle_compare_photo(self.photo_id, checked, data_packet)

    def load_hd_delayed(self):
        """Arayüzü dondurmadan HD resmi sonradan yükler."""
        import os
        if os.path.exists(self.image_path):
            hd_pix = QPixmap(self.image_path)
            if not hd_pix.isNull():
                self.source_pixmap = hd_pix
                self.is_high_res = True
                
                # Buton metnini düzelt
                self.btn_toggle.setText("SD")
                self.btn_toggle.setStyleSheet("background-color: #4CAF50; color: white; border: 1px solid #2E7D32; border-radius: 3px; font-weight: bold; font-size: 10px;")
                
                # Görüntüyü yenile
                self.update_size(self.main_window.thumbnail_size, fast=False)
                self.set_info_tooltip() # <--- YENİ EKLENEN SATIR
            else:
                # Dosya bozuksa hafızadan sil
                self.main_window.hd_memory.discard(self.photo_id)
        else:
            self.main_window.hd_memory.discard(self.photo_id)

    def update_size(self, size, fast=False):
        if self.source_pixmap.isNull(): return
        target_w = int(size)
        target_h = int(size * 1.5)
        mode = Qt.TransformationMode.FastTransformation if fast else Qt.TransformationMode.SmoothTransformation

        # Her seferinde orijinal kaynaktan ölçekle (ghosting ve kalite kaybını azaltır)
        scaled = self.source_pixmap.scaled(
            target_w,
            target_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            mode
        )

        # DPI scaling'e bırak, ekstra devicePixelRatio ayarı yapma
        self.lbl_img.setPixmap(scaled)
        self.lbl_img.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            child = self.childAt(event.position().toPoint())
            if child is not self.chk_compare:
                self.chk_compare.setChecked(not self.chk_compare.isChecked())
        super().mousePressEvent(event)

    def move_photo(self, direction):
        self.main_window.db.move_photo(self.photo_id, direction, self.main_window.current_athlete_id, self.date_str, self.competition)
        self.main_window.refresh_gallery()

    def delete_photo(self):
        if SilentConfirmDialog("Fotoğraf silinsin mi?", self).exec() == QDialog.DialogCode.Accepted:
            self.main_window.db.delete_photo(self.photo_id)
            self.main_window.refresh_gallery()

    def edit_photo(self):
        dlg = EditPhotoDialog(
            self.main_window.db,
            self.photo_id,
            self.main_window.current_athlete_id,
            self.competition,
            self.date_str,
            self
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            moved = self.main_window.db.move_photo_to_athlete(
                self.photo_id,
                dlg.final_athlete_id,
                dlg.final_date,
                dlg.final_comp
            )
            if moved:
                msg = "✅ Fotoğraf güncellendi."
                if dlg.final_athlete_id != self.main_window.current_athlete_id:
                    msg = "🚚 Fotoğraf diğer sporcuya taşındı."
                self.main_window._status.showMessage(msg, 3000)
            self.main_window.refresh_gallery()

class ZoomableGraphicsView(QGraphicsView):
    clicked = pyqtSignal(int)
    # factor, relative_x (0-1), relative_y (0-1)
    zoom_changed = pyqtSignal(float, float, float)
    pan_changed = pyqtSignal(float, float)
    reset_request = pyqtSignal()

    def __init__(self, pixmap):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.item = QGraphicsPixmapItem(pixmap)
        self.item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._scene.addItem(self.item)
        self.setScene(self._scene)
        
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.is_fit = True
        self._is_syncing = False
        self._last_mouse_pos = None
        self.fit_to_window()

    def fit_to_window(self):
        self.fitInView(self.item, Qt.AspectRatioMode.KeepAspectRatio)
        self.is_fit = True

    def reset_to_original(self):
        self.resetTransform()
        self.scale(1, 1)
        self.is_fit = False

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_mouse_pos = event.position()
            if hasattr(self, 'img_index'):
                self.clicked.emit(self.img_index)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.is_fit: self.reset_to_original()
        else: self.fit_to_window()
        if not self._is_syncing: self.reset_request.emit()
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        if event is None: return
        factor = 1.15 if event.angleDelta().y() > 0 else 1/1.15
        scene_pos = self.mapToScene(event.position().toPoint())
        self.scale(factor, factor)
        # Center around the point that was under the cursor for consistent sync
        self.centerOn(scene_pos)
        self.is_fit = False
        if not self._is_syncing:
            br = self.item.boundingRect()
            rel_x = scene_pos.x() / br.width() if br.width() else 0.5
            rel_y = scene_pos.y() / br.height() if br.height() else 0.5
            self.zoom_changed.emit(factor, rel_x, rel_y)

    def mouseMoveEvent(self, event):
        if self._last_mouse_pos and event.buttons() == Qt.MouseButton.LeftButton:
            delta = self._last_mouse_pos - event.position()
            if not self._is_syncing:
                self.pan_changed.emit(delta.x(), delta.y())
            self._last_mouse_pos = event.position()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._last_mouse_pos = None
        super().mouseReleaseEvent(event)

    def apply_zoom_sync(self, factor, rel_x, rel_y):
        self._is_syncing = True
        self.scale(factor, factor)
        self.is_fit = False
        try:
            br = self.item.boundingRect()
            target = QPointF(br.width() * rel_x, br.height() * rel_y)
            self.centerOn(target)
        except Exception:
            pass
        self._is_syncing = False

    def apply_pan_sync(self, dx, dy):
        self._is_syncing = True
        h, v = self.horizontalScrollBar(), self.verticalScrollBar()
        h.setValue(h.value() + int(dx))
        v.setValue(v.value() + int(dy))
        self._is_syncing = False

    def apply_reset_sync(self):
        self._is_syncing = True
        if self.is_fit: self.reset_to_original()
        else: self.fit_to_window()
        self._is_syncing = False


class CompareWindow(QDialog):
    def __init__(self, data_list):
        super().__init__()
        self.setWindowTitle("Karşılaştırma Modu")
        self.resize(1200, 800)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowMinMaxButtonsHint | Qt.WindowType.WindowCloseButtonHint)
        self.showMaximized()

        # --- Model ---
        # data_list içindeki öğeler: {'pixmap': QPixmap, 'athlete': str, 'competition': str, 'date': str}
        self.data_items = data_list 
        self.display_order = list(range(len(self.data_items)))
        self.visible_indices = set(self.display_order) 
        self.current_focus_index = None
        
        # --- Cache ---
        self.card_cache = {} 
        self.container = None 

        # --- UI Yapısı ---
        root = QVBoxLayout(self)

        # 1. Üst Bar
        top = QHBoxLayout()
        self.cmb_layout = QComboBox()
        self.cmb_layout.addItems(["Tek Sütun", "Tek Satır", "Grid 2", "Grid 3"])
        self.cmb_layout.setCurrentIndex(1)
        self.cmb_layout.currentIndexChanged.connect(self.rebuild_view)

        self.btn_move_prev = QPushButton("⬅️ Öne")
        self.btn_move_next = QPushButton("➡️ Sona")
        self.btn_move_prev.clicked.connect(lambda: self.move_focused(-1))
        self.btn_move_next.clicked.connect(lambda: self.move_focused(+1))

        self.chk_sync = QCheckBox("🔄 Senkronize Zoom")
        self.chk_sync.setChecked(True)
        
        self.btn_toggle_thumbs = QPushButton("🔽 Paneli Gizle")
        self.btn_toggle_thumbs.setCheckable(True)
        self.btn_toggle_thumbs.setChecked(True)
        self.btn_toggle_thumbs.toggled.connect(self.toggle_thumbs_panel)

        top.addWidget(QLabel("Dizilim:"))
        top.addWidget(self.cmb_layout)
        top.addStretch()
        top.addWidget(self.btn_move_prev)
        top.addWidget(self.btn_move_next)
        top.addWidget(self.chk_sync)
        top.addWidget(self.btn_toggle_thumbs)
        root.addLayout(top)

        # 2. Ana Alan
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("background-color: #111;")
        root.addWidget(self.scroll_area, 1)

        self.container = QWidget()
        self.container.setStyleSheet("background-color: #111;")

        self.main_layout = QGridLayout(self.container)
        self.main_layout.setSpacing(2)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        self.scroll_area.setWidget(self.container)

        # 3. Alt Thumbnail Paneli
        self.thumbs_panel = QWidget()
        self.thumbs_panel.setFixedHeight(170)
        thumbs_layout = QVBoxLayout(self.thumbs_panel)
        thumbs_layout.setContentsMargins(5, 5, 5, 5)

        thumbs_tools = QHBoxLayout()
        btn_all = QPushButton("☑️ Hepsini Göster")
        btn_none = QPushButton("⬜ Hepsini Gizle")
        btn_all.clicked.connect(self.check_all_thumbs)
        btn_none.clicked.connect(self.uncheck_all_thumbs)
        
        thumbs_tools.addWidget(QLabel("Görünürlük:"))
        thumbs_tools.addWidget(btn_all)
        thumbs_tools.addWidget(btn_none)
        thumbs_tools.addStretch()
        thumbs_layout.addLayout(thumbs_tools)
        
        self.list_thumbs = QListWidget()
        self.list_thumbs.setViewMode(QListWidget.ViewMode.IconMode)
        self.list_thumbs.setIconSize(QSize(100, 100))
        self.list_thumbs.setMovement(QListWidget.Movement.Static)
        self.list_thumbs.setSpacing(8)
        self.list_thumbs.setStyleSheet("""
            QListWidget { background-color: #222; border: 1px solid #444; } 
            QListWidget::item { color: #ddd; }
            QListWidget::item:selected { background-color: #333; border: 1px solid #00AEEF; }
        """)
        self.list_thumbs.itemChanged.connect(self.on_thumb_check_changed)
        self.list_thumbs.itemClicked.connect(self.on_thumb_clicked)

        thumbs_layout.addWidget(self.list_thumbs)
        root.addWidget(self.thumbs_panel)
        
        self.populate_thumbs()
        self.rebuild_view()

    def rebuild_view(self):
        # 1. Mevcut Layout'u Temizle
        while self.main_layout.count():
            item = self.main_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.hide() 
                widget.setParent(None)
        
        mode = self.cmb_layout.currentText()
        cols = 1 
        
        if "Tek Satır" in mode:
            cols = 9999 
            self.scroll_area.horizontalScrollBar().setValue(0)
            self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self.main_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)

        elif "Grid" in mode:
            cols = 2 if "Grid 2" in mode else 3
            self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self.main_layout.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)

        else: # Tek Sütun
            self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self.main_layout.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)

        show_indices = [i for i in self.display_order if i in self.visible_indices]
        
        row, col = 0, 0
        for idx in show_indices:
            if idx in self.card_cache:
                card = self.card_cache[idx]
            else:
                card = self.make_viewer_card(idx)
                self.card_cache[idx] = card
            
            self.main_layout.addWidget(card, row, col)
            card.show()
            
            col += 1
            if col >= cols:
                col = 0
                row += 1

        self.update_focus_style()

    def make_viewer_card(self, idx: int) -> QWidget:
            data = self.data_items[idx] 
            
            card = QFrame()
            card.setStyleSheet("QFrame { border: 1px solid #333; background: #111; margin: 0px; }")
            
            # --- DÜZELTME 1: KART SERBEST KALSIN ---
            # setFixedSize'ı kaldırdık. Yerine setMinimumSize koyuyoruz.
            # Böylece kartlar en az 400x300 olur ama pencereyi büyütürsen onlar da büyür.
            card.setMinimumSize(400, 300) 
            # ---------------------------------------

            v = QVBoxLayout(card)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(0)

            # --- BİLGİ ETİKETİ ---
            info_text = f"👤 {data['athlete']}   |   🏆 {data['competition']} ({data['date']})"
            lbl_info = QLabel(info_text)
            lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_info.setWordWrap(True) 

            # --- DÜZELTME 2: SADECE BAŞLIĞI SABİTLE ---
            # Başlık kısmı ne olursa olsun 40px yüksekliğinde kalsın.
            # Böylece resimlerin hepsi aynı hizada başlar.
            lbl_info.setFixedHeight(40)
            # ------------------------------------------

            # Kopyalama özelliği açık
            lbl_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

            lbl_info.setStyleSheet("""
                background-color: #222; 
                color: #fff; 
                font-weight: bold; 
                padding: 2px; 
                font-size: 12px; 
                border-bottom: 1px solid #444;
            """)
            v.addWidget(lbl_info) 
            full_pixmap = QPixmap(data['image_path'])
            viewer = ZoomableGraphicsView(full_pixmap)
            viewer.setFrameShape(QFrame.Shape.NoFrame)
            viewer.img_index = idx
            
            # Resim alanı esnesin (Expanding)
            viewer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            
            viewer.clicked.connect(self.set_focus)
            viewer.zoom_changed.connect(lambda f, rx, ry: self.handle_sync_zoom(viewer, f, rx, ry))
            viewer.pan_changed.connect(lambda dx, dy: self.handle_sync_pan(viewer, dx, dy))
            viewer.reset_request.connect(lambda: self.handle_sync_reset(viewer))
            
            v.addWidget(viewer)
            return card

    def populate_thumbs(self):
        self.list_thumbs.blockSignals(True)
        self.list_thumbs.clear()
        for i, data in enumerate(self.data_items):
            # Thumbnail oluştur
            pm = QPixmap(data['image_path']).scaled(120, 120, Qt.AspectRatioMode.KeepAspectRatio)
            icon = QIcon(pm.scaled(120, 120, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            
            # Alt metin olarak sadece sporcu adı yeterli olabilir
            item_text = f"#{i+1}\n{data['athlete']}"
            
            item = QListWidgetItem(icon, item_text)
            item.setData(Qt.ItemDataRole.UserRole, i)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.list_thumbs.addItem(item)
        self.list_thumbs.blockSignals(False)

    # ... (Geri kalan tüm fonksiyonlar AYNI: on_thumb_check_changed, on_thumb_clicked vb.) ...
    # Aşağıdaki fonksiyonlar değişmedi, kopyala-yapıştır yapabilirsin
    def on_thumb_check_changed(self, item):
        idx = item.data(Qt.ItemDataRole.UserRole)
        if item.checkState() == Qt.CheckState.Checked:
            self.visible_indices.add(idx)
        else:
            self.visible_indices.discard(idx)
        QTimer.singleShot(0, self.rebuild_view)

    def on_thumb_clicked(self, item):
        idx = item.data(Qt.ItemDataRole.UserRole)
        if self.current_focus_index == idx:
            self.set_focus(None)
            self.list_thumbs.clearSelection()
            return
        if item.checkState() != Qt.CheckState.Checked:
            item.setCheckState(Qt.CheckState.Checked)
        self.set_focus(idx, force_select=True)
        if idx in self.card_cache and self.card_cache[idx].isVisible():
             self.scroll_area.ensureWidgetVisible(self.card_cache[idx])

    def check_all_thumbs(self):
        self.list_thumbs.blockSignals(True)
        for i in range(self.list_thumbs.count()):
            self.list_thumbs.item(i).setCheckState(Qt.CheckState.Checked)
        self.visible_indices = set(self.display_order)
        self.list_thumbs.blockSignals(False)
        self.rebuild_view()

    def uncheck_all_thumbs(self):
        self.list_thumbs.blockSignals(True)
        for i in range(self.list_thumbs.count()):
            self.list_thumbs.item(i).setCheckState(Qt.CheckState.Unchecked)
        self.visible_indices = set()
        self.list_thumbs.blockSignals(False)
        self.rebuild_view()
    
    def toggle_thumbs_panel(self, checked):
        self.thumbs_panel.setVisible(checked)
        self.btn_toggle_thumbs.setText("🔽 Paneli Gizle" if checked else "🔼 Paneli Göster")

    def set_focus(self, idx: int, force_select=False):
        if not force_select and self.current_focus_index == idx:
            self.current_focus_index = None
            self.list_thumbs.clearSelection()
        else:
            self.current_focus_index = idx
            for i in range(self.list_thumbs.count()):
                item = self.list_thumbs.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == idx:
                    self.list_thumbs.setCurrentItem(item)
                    break
        self.update_focus_style()

    def update_focus_style(self):
        active = "2px solid #00AEEF"
        passive = "1px solid #333"
        for idx, card in self.card_cache.items():
            if idx == self.current_focus_index:
                card.setStyleSheet(f"QFrame {{ border: {active}; background: #111; margin: 0px; }}")
            else:
                card.setStyleSheet(f"QFrame {{ border: {passive}; background: #111; margin: 0px; }}")

    def move_focused(self, direction: int):
        if self.current_focus_index is None: return
        try:
            pos = self.display_order.index(self.current_focus_index)
        except: return
        new_pos = pos + direction
        if 0 <= new_pos < len(self.display_order):
            self.display_order[pos], self.display_order[new_pos] = self.display_order[new_pos], self.display_order[pos]
            self.rebuild_view()
            if self.current_focus_index in self.card_cache:
                self.scroll_area.ensureWidgetVisible(self.card_cache[self.current_focus_index])

    def get_all_viewers(self):
        viewers = []
        for idx in self.visible_indices:
            if idx in self.card_cache and self.card_cache[idx].isVisible():
                v = self.card_cache[idx].findChild(ZoomableGraphicsView)
                if v: viewers.append(v)
        return viewers

    def handle_sync_zoom(self, sender_viewer, factor, rel_x, rel_y):
        if not self.chk_sync.isChecked(): return
        for v in self.get_all_viewers():
            if v is not sender_viewer: v.apply_zoom_sync(factor, rel_x, rel_y)

    def handle_sync_pan(self, sender_viewer, dx, dy):
        if not self.chk_sync.isChecked(): return
        for v in self.get_all_viewers():
            if v is not sender_viewer: v.apply_pan_sync(dx, dy)

    def handle_sync_reset(self, sender_viewer):
        if not self.chk_sync.isChecked(): return
        for v in self.get_all_viewers():
            if v is not sender_viewer: v.apply_reset_sync()

class CoefficientSettingsDialog(QDialog):
    def __init__(self, db, division, parent=None):
        super().__init__(parent)
        self.db = db
        self.division = division
        self.setWindowTitle(f"Katsayı Ayarları - {division}")
        self.resize(520, 600)

        layout = QVBoxLayout()

        # Division seçimi (istersen sabit kalsın; ben değiştirilebilir yaptım)
        top = QHBoxLayout()
        top.addWidget(QLabel("Division:"))
        self.div_combo = QComboBox()
        self.div_combo.addItems(SCORING_SYSTEM.keys())
        self.div_combo.setCurrentText(division)
        self.div_combo.currentTextChanged.connect(self.load_division)
        top.addWidget(self.div_combo)
        layout.addLayout(top)

        self.scroll_area  = QScrollArea()
        self.scroll_area .setWidgetResizable(True)
        self.container = QWidget()
        self.form = QFormLayout()
        self.container.setLayout(self.form)
        self.scroll_area .setWidget(self.container)
        layout.addWidget(self.scroll_area)

        btns = QHBoxLayout()
        self.btn_add = QPushButton("➕ Kriter Ekle")
        self.btn_add.clicked.connect(self.add_row)
        self.btn_save = QPushButton("✅ Kaydet")
        self.btn_save.clicked.connect(self.save)
        self.btn_save.setStyleSheet("background-color:#4CAF50;color:white;padding:8px;")
        btns.addWidget(self.btn_add)
        btns.addStretch()
        btns.addWidget(self.btn_save)
        layout.addLayout(btns)

        self.setLayout(layout)

        self.rows = []  # [(QLineEdit, QDoubleSpinBox, QPushButton), ...]
        self.load_division(self.division)

    def clear_rows(self):
        while self.form.rowCount():
            self.form.removeRow(0)
        self.rows.clear()

    def load_division(self, division):
        self.division = division
        self.setWindowTitle(f"Katsayı Ayarları - {division}")
        self.clear_rows()

        data = self.db.get_criteria(division)  # [(criterion, weight)]
        if not data:
            data = []

        for crit, w in data:
            self.add_row(crit, float(w))

    def add_row(self, crit_text="", weight_val=1.0):
        # PyQt clicked(bool) sinyali buraya bool gönderirse düzelt
        if isinstance(crit_text, bool):
            crit_text = ""
            weight_val = 1.0

        row_widget = QWidget()
        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)

        crit = QLineEdit()
        crit.setText(str(crit_text) if crit_text is not None else "")

        w = QDoubleSpinBox()
        w.setRange(0, 10)
        w.setSingleStep(0.5)
        w.setValue(float(weight_val) if weight_val is not None else 1.0)

        btn_del = QPushButton("🗑️")
        btn_del.setFixedWidth(45)

        row_layout.addWidget(crit, 1)
        row_layout.addWidget(QLabel("x"))
        row_layout.addWidget(w)
        row_layout.addWidget(btn_del)
        row_widget.setLayout(row_layout)

        self.form.addRow(row_widget)
        self.rows.append((crit, w, btn_del))

        btn_del.clicked.connect(lambda: self.remove_row(crit, w, row_widget))


    def remove_row(self, crit, w, row_widget):
        # listeden çıkar
        self.rows = [r for r in self.rows if not (r[0] is crit and r[1] is w)]
        # formdan kaldır
        row_widget.setParent(None)
        row_widget.deleteLater()

    def save(self):
        items = []
        order = 1

        for crit_edit, w_spin, _ in self.rows:
            c = crit_edit.text().strip()
            if not c:
                continue
            items.append((c, float(w_spin.value()), order))
            order += 1

        self.db.upsert_criteria(self.division, items)

        # Sessiz kapat
        self.accept()

class CompetitionGroupBox(QGroupBox):
    def __init__(self, title, date_str, comp_name, main_window, is_favorite=False, year=None):
        super().__init__(title)
        self.date_str = date_str
        self.comp_name = comp_name
        self.main_window = main_window
        self.is_favorite = is_favorite
        self.year = year or (date_str.split("-")[0] if date_str else "")

    def contextMenuEvent(self, event):
        menu = QMenu(self)

        # Başlığa özel yapıştırma seçeneği
        action_paste = QAction(f"📌 Resmi Buraya Yapıştır ({self.comp_name})", self)
        action_paste.triggered.connect(self.paste_here)
        menu.addAction(action_paste)

        # Favori ayarı
        if self.is_favorite:
            action_fav = QAction("⭐ Favoriden Çıkar (Bu Yıl)", self)
            action_fav.triggered.connect(lambda: self.main_window.unfavorite_competition(self.year, self.comp_name))
        else:
            action_fav = QAction("⭐ Bu Yarışmayı Üstte Tut (Favori)", self)
            action_fav.triggered.connect(lambda: self.main_window.favorite_competition(self.year, self.comp_name))
        menu.addAction(action_fav)

        menu.exec(event.globalPos())

    def paste_here(self):
        # Doğrudan bu kutunun tarihine ve ismine gönderiyoruz
        self.main_window.paste_to_target(self.date_str, self.comp_name)

class EvolutionCanvas(QWidget):
    """İki resim arasında geçiş ve hizalama yapan özel çizim alanı."""
    def __init__(self, pixmap_before, pixmap_after, parent=None):
        super().__init__(parent)
        self.pix_base = pixmap_before # Alttaki (Sabit)
        self.pix_overlay = pixmap_after # Üstteki (Hareketli)

        self.opacity = 0.5 # Şeffaflık (0.0 = Sadece Alt, 1.0 = Sadece Üst)

        # Hizalama Değişkenleri (Üstteki resim için)
        self.overlay_scale = 1.0
        self.overlay_offset_x = 0
        self.overlay_offset_y = 0

        # Alt resim için isteğe bağlı hizalama
        self.base_scale = 1.0
        self.base_offset_x = 0
        self.base_offset_y = 0

        # Düzenlenecek katman: "overlay" veya "base"
        self.edit_target = "overlay"

        # Fare Kontrolü
        self.last_mouse_pos = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Arka planı siyah yap
        painter.fillRect(self.rect(), Qt.GlobalColor.black)

        # 1. ALT RESMİ ÇİZ (Kullanıcı ölçek/offset uygulayabilir)
        w, h = self.width(), self.height()
        if self.pix_base and not self.pix_base.isNull():
            base_fit = self.pix_base.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            base_w = int(base_fit.width() * self.base_scale)
            base_h = int(base_fit.height() * self.base_scale)
            base_scaled = self.pix_base.scaled(base_w, base_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

            center_x = w // 2
            center_y = h // 2
            x_base = int(center_x - (base_scaled.width() // 2) + self.base_offset_x)
            y_base = int(center_y - (base_scaled.height() // 2) + self.base_offset_y)

            painter.drawPixmap(x_base, y_base, base_scaled)

        # 2. ÜST RESMİ ÇİZ (HAREKETLİ, ŞEFFAF)
        if self.pix_overlay and not self.pix_overlay.isNull():
            painter.setOpacity(self.opacity)

            # Üst resmin temel boyutu (Alttakiyle uyumlu olması için önce ölçekliyoruz)
            # Mantık: Başlangıçta ikisi de ekrana sığsın
            base_scale_w = self.pix_overlay.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio).width()
            base_scale_h = self.pix_overlay.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio).height()

            # Sonra kullanıcının zoom/pan ayarlarını ekliyoruz
            final_w = int(base_scale_w * self.overlay_scale)
            final_h = int(base_scale_h * self.overlay_scale)

            # Merkezden hizalayarak çizim koordinatlarını bul
            # Varsayılan (offset=0) iken tam ortada olmalı
            center_x = w // 2
            center_y = h // 2

            draw_x = int(center_x - (final_w // 2) + self.overlay_offset_x)
            draw_y = int(center_y - (final_h // 2) + self.overlay_offset_y)

            painter.drawPixmap(draw_x, draw_y, final_w, final_h, self.pix_overlay)

    # --- FARE KONTROLLERİ ---
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_mouse_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self.last_mouse_pos:
            delta = event.position() - self.last_mouse_pos
            if self.edit_target == "base":
                self.base_offset_x += delta.x()
                self.base_offset_y += delta.y()
            else:
                self.overlay_offset_x += delta.x()
                self.overlay_offset_y += delta.y()
            self.last_mouse_pos = event.position()
            self.update() # Yeniden çiz

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_mouse_pos = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def wheelEvent(self, event):
        # Tekerlekle Zoom Yapma
        zoom_in = event.angleDelta().y() > 0
        factor = 1.05 if zoom_in else 0.95
        if self.edit_target == "base":
            self.base_scale = max(0.1, min(5.0, self.base_scale * factor))
        else:
            self.overlay_scale = max(0.1, min(5.0, self.overlay_scale * factor))
        self.update()

    def set_edit_target(self, target):
        if target in ("base", "overlay"):
            self.edit_target = target

    def reset_transforms(self):
        self.overlay_scale = 1.0
        self.overlay_offset_x = 0
        self.overlay_offset_y = 0
        self.base_scale = 1.0
        self.base_offset_x = 0
        self.base_offset_y = 0
        self.update()


class EvolutionViewerDialog(QDialog):
    def __init__(self, photo_list, parent=None):
        """photo_list: [(pixmap, date, comp), (pixmap, date, comp)] en az 2 öğe"""
        super().__init__(parent)
        self.setWindowTitle("🧬 Gelişim Analizi (Morph)")
        self.resize(1000, 800)
        self.photo_list = photo_list

        # Listeyi tarihe göre sıralayalım
        self.photo_list.sort(key=lambda x: x[1]) # Tarihe göre artan

        self.idx_a = 0
        self.idx_b = 1 if len(self.photo_list) > 1 else 0
        self.seq_index = 0  # ardışık çiftler için başlangıç
        self.play_interval_ms = 50
        self.setModal(False)
        # Pencere butonları (min/max) açık kalsın
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMinMaxButtonsHint | Qt.WindowType.WindowCloseButtonHint)

        self.init_ui()

        # Animasyon Zamanlayıcısı
        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self.animate_step)
        self.anim_direction = 1 # 1: İleri, -1: Geri
        self.is_playing = False

    def init_ui(self):
        layout = QVBoxLayout()

        # 1. Canvas
        self.canvas = EvolutionCanvas(self.photo_list[self.idx_a][0], self.photo_list[self.idx_b][0])
        layout.addWidget(self.canvas, 1) # 1 = Esnek alan

        # Çift seçimi
        pair_select = QHBoxLayout()
        pair_select.addWidget(QLabel("A:"))
        self.combo_a = QComboBox()
        pair_select.addWidget(self.combo_a, 1)

        pair_select.addWidget(QLabel("B:"))
        self.combo_b = QComboBox()
        pair_select.addWidget(self.combo_b, 1)

        self.combo_a.currentIndexChanged.connect(self.on_pair_combo_changed)
        self.combo_b.currentIndexChanged.connect(self.on_pair_combo_changed)

        self._populate_pair_combos()
        layout.addLayout(pair_select)

        # Bilgi Paneli (Hangi yıllar kıyaslanıyor)
        info_layout = QHBoxLayout()
        self.lbl_info_a = QLabel(f"A: {self.photo_list[self.idx_a][1]} ({self.photo_list[self.idx_a][2]})")
        self.lbl_info_b = QLabel(f"B: {self.photo_list[self.idx_b][1]} ({self.photo_list[self.idx_b][2]})")
        self.lbl_info_a.setStyleSheet("color: #bbb; font-weight: bold;")
        self.lbl_info_b.setStyleSheet("color: #4CAF50; font-weight: bold;")
        
        info_layout.addWidget(self.lbl_info_a)
        info_layout.addStretch()
        info_layout.addWidget(QLabel("VS"))
        info_layout.addStretch()
        info_layout.addWidget(self.lbl_info_b)
        layout.addLayout(info_layout)

        # 2. Kontroller
        ctrl_layout = QHBoxLayout()

        self.btn_play = QPushButton("▶ Oynat")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setStyleSheet("background-color: #2196F3; color: white; padding: 10px;")

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(self.on_slider_change)

        btn_reset = QPushButton("⟳ Hizalamayı Sıfırla")
        btn_reset.clicked.connect(self.reset_alignment)
        btn_reset.setStyleSheet("padding: 10px;")

        self.btn_prev_pair = QPushButton("⏮ Önceki Çift")
        self.btn_prev_pair.clicked.connect(lambda: self.step_pair(-1))
        self.btn_prev_pair.setStyleSheet("padding: 10px;")

        self.btn_next_pair = QPushButton("⏭ Sonraki Çift")
        self.btn_next_pair.clicked.connect(lambda: self.step_pair(1))
        self.btn_next_pair.setStyleSheet("padding: 10px;")

        self.chk_cycle = QCheckBox("Tüm fotolarda sırayla")
        self.chk_cycle.setChecked(True)

        # Oynatma hızı
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(10, 200)  # ms
        self.speed_slider.setValue(self.play_interval_ms)
        self.speed_slider.setSingleStep(10)
        self.speed_slider.setTickInterval(10)
        self.speed_slider.valueChanged.connect(self.on_speed_change)
        self.lbl_speed = QLabel(f"Hız: {self.play_interval_ms} ms")
        self.lbl_speed.setStyleSheet("color:#bbb;")

        ctrl_layout.addWidget(self.btn_play)
        ctrl_layout.addWidget(QLabel("Eski"))
        ctrl_layout.addWidget(self.slider, 1)
        ctrl_layout.addWidget(QLabel("Yeni"))
        ctrl_layout.addWidget(btn_reset)
        ctrl_layout.addWidget(self.btn_prev_pair)
        ctrl_layout.addWidget(self.btn_next_pair)
        ctrl_layout.addWidget(self.chk_cycle)

        # Hangi katman düzenleniyor?
        self.chk_edit_base = QCheckBox("Alt fotoğrafı düzenle")
        self.chk_edit_base.stateChanged.connect(self.on_edit_target_changed)
        self.lbl_edit_target = QLabel("Düzenlenen: Üst fotoğraf")
        self.lbl_edit_target.setStyleSheet("color:#bbb;")

        layer_layout = QHBoxLayout()
        layer_layout.addWidget(self.chk_edit_base)
        layer_layout.addWidget(self.lbl_edit_target)
        layer_layout.addStretch()

        # Speed row
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Oynatma Hızı"))
        speed_layout.addWidget(self.speed_slider, 1)
        speed_layout.addWidget(self.lbl_speed)

        layout.addLayout(ctrl_layout)
        layout.addLayout(layer_layout)
        layout.addLayout(speed_layout)

        # İpucu
        tip = QLabel("İPUCU: Fareyle üstteki fotoğrafı sürükleyerek ve tekerlekle büyüterek alttakiyle hizalayın.")
        tip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tip.setStyleSheet("color: #888; font-style: italic; margin-top: 5px;")
        layout.addWidget(tip)

        self.setLayout(layout)
        
        # Başlangıç değeri
        self.canvas.opacity = 0.5

    def _populate_pair_combos(self):
        self.combo_a.blockSignals(True)
        self.combo_b.blockSignals(True)
        self.combo_a.clear()
        self.combo_b.clear()
        for idx, (_, date, comp) in enumerate(self.photo_list):
            label = f"{date} • {comp}"
            self.combo_a.addItem(label, idx)
            self.combo_b.addItem(label, idx)
        self.combo_a.setCurrentIndex(self.idx_a)
        self.combo_b.setCurrentIndex(self.idx_b)
        self.combo_a.blockSignals(False)
        self.combo_b.blockSignals(False)

    def on_slider_change(self, val):
        self.canvas.opacity = val / 100.0
        self.canvas.update()

    def on_pair_combo_changed(self):
        a_idx = self.combo_a.currentData()
        b_idx = self.combo_b.currentData()
        if a_idx == b_idx:
            # Aynı seçilirse otomatik olarak bir sonrakine kaydır
            if b_idx < len(self.photo_list) - 1:
                b_idx += 1
            else:
                a_idx = max(0, a_idx - 1)
        self.set_pair(a_idx, b_idx, reset_slider=True, update_seq=True)

    def set_pair(self, a_idx, b_idx, reset_slider=False, update_seq=False):
        self.idx_a = max(0, min(a_idx, len(self.photo_list) - 1))
        self.idx_b = max(0, min(b_idx, len(self.photo_list) - 1))
        if self.idx_a == self.idx_b and self.idx_b < len(self.photo_list) - 1:
            self.idx_b += 1
        self.canvas.img_base = self.photo_list[self.idx_a][0]
        self.canvas.img_overlay = self.photo_list[self.idx_b][0]
        self.canvas.update()
        self.lbl_info_a.setText(f"A: {self.photo_list[self.idx_a][1]} ({self.photo_list[self.idx_a][2]})")
        self.lbl_info_b.setText(f"B: {self.photo_list[self.idx_b][1]} ({self.photo_list[self.idx_b][2]})")
        # Comboboxları eşitle
        self.combo_a.blockSignals(True)
        self.combo_b.blockSignals(True)
        self.combo_a.setCurrentIndex(self.idx_a)
        self.combo_b.setCurrentIndex(self.idx_b)
        self.combo_a.blockSignals(False)
        self.combo_b.blockSignals(False)
        if reset_slider:
            self.slider.setValue(0)
        if update_seq:
            self.seq_index = min(self.idx_a, self.idx_b)

    def toggle_play(self):
        if self.is_playing:
            self.anim_timer.stop()
            self.btn_play.setText("▶ Oynat")
            self.is_playing = False
        else:
            self.anim_timer.start(self.play_interval_ms)
            self.btn_play.setText("⏸ Durdur")
            self.is_playing = True

    def animate_step(self):
        val = self.slider.value()
        step = 2

        # Sürekli ileri modunda (cycle) sadece ileri sar
        if self.chk_cycle.isChecked():
            if val >= 100:
                self.step_pair(1, auto=True)
                self.slider.setValue(0)
            else:
                self.slider.setValue(val + step)
            return

        # Ping-pong modu
        if val >= 100:
            self.anim_direction = -1
        elif val <= 0:
            self.anim_direction = 1

        new_val = val + (step * self.anim_direction)
        self.slider.setValue(new_val)

    def step_pair(self, direction, auto=False):
        # ardışık çiftler: (i, i+1)
        n = len(self.photo_list)
        if n < 2:
            return
        self.seq_index = (self.seq_index + direction) % (n - 1)
        a_idx = self.seq_index
        b_idx = self.seq_index + 1
        self.set_pair(a_idx, b_idx, reset_slider=not auto, update_seq=False)

    def reset_alignment(self):
        self.canvas.reset_transforms()

    def on_speed_change(self, val):
        self.play_interval_ms = val
        self.lbl_speed.setText(f"Hız: {val} ms")
        if self.is_playing:
            self.anim_timer.start(self.play_interval_ms)

    def on_edit_target_changed(self, state):
        if state == Qt.CheckState.Checked.value:
            self.canvas.set_edit_target("base")
            self.lbl_edit_target.setText("Düzenlenen: Alt fotoğraf")
        else:
            self.canvas.set_edit_target("overlay")
            self.lbl_edit_target.setText("Düzenlenen: Üst fotoğraf")


# --- ANA PENCERE ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._status = QStatusBar(self)
        self.setStatusBar(self._status)
        self.setWindowTitle("Bikini Fitness Tracker - PRO V13 (Yerel - Tek Dosya)")
        self.resize(1300, 850)
        self.db = DatabaseManager()
        self.current_athlete_id = None
        self.current_division = "Bikini" 
        self.photo_widgets = []
        self.thumbnail_size = 250
        self.active_dialogs = []
        self.compare_selection = {}  # photo_id -> QPixmap
        self.last_context_date = None
        self.last_context_comp = None
        self.athlete_year_memory = {} # Örn: {12: "2021", 15: "2024"}
        self.hd_memory = set()

        self.setAcceptDrops(True)
        self.init_ui()

        self.shortcut_paste = QShortcut(QKeySequence("Ctrl+V"), self)
        self.shortcut_paste.activated.connect(self.paste_from_clipboard)

    def init_ui(self):        
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        # ==========================
        # --- SOL PANEL (MENÜ) ---
        # ==========================
        left_layout = QVBoxLayout()
        
        lbl_div = QLabel("KATEGORİ (DIVISION)")
        lbl_div.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        left_layout.addWidget(lbl_div)
        
        self.combo_division = QComboBox()
        self.combo_division.addItems(SCORING_SYSTEM.keys())
        self.combo_division.currentTextChanged.connect(self.change_division)
        left_layout.addWidget(self.combo_division)

        left_layout.addSpacing(10) # Biraz boşluk

        left_layout.addWidget(QLabel("SPORCU LİSTESİ (Puana Göre)"))

        # --- ARAMA KUTUSU (YENİ) ---
        self.txt_search = QLineEdit()
        self.txt_search.setPlaceholderText("🔍 İsim Ara...")
        self.txt_search.setClearButtonEnabled(True) # Çarpı butonu koyar
        self.txt_search.textChanged.connect(self.filter_athlete_list)
        left_layout.addWidget(self.txt_search)
        # ---------------------------

        left_layout.addSpacing(6)
        lbl_quick = QLabel("HIZLI ERİŞİM")
        lbl_quick.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        left_layout.addWidget(lbl_quick)

        self.lbl_quick_hint = QLabel("Sağ tık > Hızlı Erişim’e ekle")
        self.lbl_quick_hint.setStyleSheet("color: #888; font-size: 11px;")
        left_layout.addWidget(self.lbl_quick_hint)

        self.list_quick_access = QListWidget()
        self.list_quick_access.setFixedHeight(140)
        self.list_quick_access.itemClicked.connect(self.jump_to_quick_access_item)
        self.list_quick_access.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list_quick_access.customContextMenuRequested.connect(self.open_quick_access_menu)
        left_layout.addWidget(self.list_quick_access)

        left_layout.addSpacing(6)
        self.list_athletes = QListWidget()
        self.list_athletes.itemClicked.connect(self.load_athlete_details)
        left_layout.addWidget(self.list_athletes)

        # --- İÇE / DIŞA AKTAR BUTONLARI (YENİ) ---
        io_layout = QHBoxLayout()
        
        self.btn_import = QPushButton("📥 İçe Aktar")
        self.btn_import.setToolTip("CSV dosyasından toplu sporcu yükle")
        self.btn_import.clicked.connect(self.import_athletes_csv)
        self.btn_import.setStyleSheet("background-color: #00897B; color: white;")
        
        self.btn_export = QPushButton("📤 Dışa Aktar")
        self.btn_export.setToolTip("Tüm listeyi Excel/CSV olarak kaydet")
        self.btn_export.clicked.connect(self.export_athletes_csv)
        self.btn_export.setStyleSheet("background-color: #555; color: white;")
        
        io_layout.addWidget(self.btn_import)
        io_layout.addWidget(self.btn_export)
        left_layout.addLayout(io_layout)
        # ----------------------------------------
        
        btn_add = QPushButton("➕ Sporcu Ekle")
        btn_add.clicked.connect(self.add_new_athlete)
        
        self.btn_score = QPushButton("⭐ PUANLA")
        self.btn_score.setEnabled(False)
        self.btn_score.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        self.btn_score.clicked.connect(self.open_scoring)

        self.btn_settings = QPushButton("⚙️ Ayarlar")
        self.btn_settings.clicked.connect(self.open_settings)
        self.btn_settings.setStyleSheet("background-color: #555; color: white; margin-top: 10px;")
        left_layout.addWidget(self.btn_settings)

        
        self.btn_coeff = QPushButton("⚙️ Katsayı Ayarları")
        self.btn_coeff.clicked.connect(self.open_coeff_settings)
        left_layout.addWidget(self.btn_coeff)

        self.btn_comps = QPushButton("🏆 Yarışma Yönetimi")
        self.btn_comps.clicked.connect(self.open_competition_manager)
        self.btn_comps.setStyleSheet("background-color: #607D8B; color: white;")
        left_layout.addWidget(self.btn_comps)

        left_layout.addWidget(btn_add)
        left_layout.addWidget(self.btn_score)
        
        left_panel = QFrame()
        left_panel.setLayout(left_layout)
        left_panel.setFixedWidth(260)
        
        # Context Menüleri
        self.list_athletes.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list_athletes.customContextMenuRequested.connect(self.open_athlete_context_menu)

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.open_main_context_menu)

        # ==========================
        # --- SAĞ PANEL (İÇERİK) ---
        # ==========================
        right_layout = QVBoxLayout()
        
        # 1. ÜST BAR (BUTONLAR)
        top_bar = QHBoxLayout()
        
        self.btn_add_photo = QPushButton("📷 Fotoğraf Yükle")
        self.btn_add_photo.setEnabled(False)
        self.btn_add_photo.clicked.connect(self.upload_photo)
        self.btn_add_photo.setStyleSheet("padding: 6px 12px; font-weight: bold;") 
        
        # HD / SD Butonları
        self.btn_gallery_hd = QPushButton("Galeriyi HD Yap")
        self.btn_gallery_hd.setToolTip("Şu anki görünümü HD yapar.")
        self.btn_gallery_hd.clicked.connect(lambda: self.batch_change_resolution(True))
        self.btn_gallery_hd.setStyleSheet("""
            QPushButton { background-color: #2E7D32; color: white; font-weight: bold; border-radius: 4px; padding: 6px; }
            QPushButton:hover { background-color: #4CAF50; }
        """)

        self.btn_gallery_sd = QPushButton("Galeriyi SD Yap")
        self.btn_gallery_sd.setToolTip("Şu anki görünümü Thumbnail yapar.")
        self.btn_gallery_sd.clicked.connect(lambda: self.batch_change_resolution(False))
        self.btn_gallery_sd.setStyleSheet("""
            QPushButton { background-color: #555; color: white; font-weight: bold; border-radius: 4px; padding: 6px; }
            QPushButton:hover { background-color: #777; }
        """)

        # Acil Durum Butonu
        self.btn_reset_all_sd = QPushButton("⚠️ TÜMÜNÜ SD YAP (RAM Temizle)")
        self.btn_reset_all_sd.setToolTip("Programdaki tüm HD hafızasını siler ve her şeyi SD yapar.")
        self.btn_reset_all_sd.clicked.connect(self.emergency_reset_sd)
        self.btn_reset_all_sd.setStyleSheet("""
            QPushButton { background-color: #B71C1C; color: white; font-weight: bold; border-radius: 4px; padding: 6px; }
            QPushButton:hover { background-color: #D32F2F; }
        """)

        # Karşılaştırma Butonları
        # Başlangıçta 0 olduğu için text'i sıfırlıyoruz
        self.btn_compare = QPushButton(f"⚔️ Karşılaştır ({0})")
        self.btn_compare.clicked.connect(self.open_comparison)
        self.btn_compare.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 6px;")

        self.btn_compare_clear = QPushButton("🗑️")
        self.btn_compare_clear.setToolTip("Seçilenleri Temizle")
        self.btn_compare_clear.setFixedWidth(40)
        self.btn_compare_clear.clicked.connect(self.clear_compare_selection)
        self.btn_compare_clear.setStyleSheet("background-color: #444; color: white; font-weight: bold; padding: 6px;")

        # (Karşılaştır butonunun yanına)
        self.btn_evolution = QPushButton("🧬 Gelişim")
        self.btn_evolution.setToolTip("Seçili 2 fotoğrafı üst üste bindirerek değişimi analiz et.")
        self.btn_evolution.clicked.connect(self.open_evolution_viewer)
        self.btn_evolution.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold; padding: 6px;")
        

        # Üst Bar Yerleşimi (Slider Buradan Kaldırıldı)
        top_bar.addWidget(self.btn_add_photo)
        
        # Ayırıcı Çizgi
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        top_bar.addWidget(line)
        
        top_bar.addWidget(self.btn_gallery_hd)
        top_bar.addWidget(self.btn_gallery_sd)
        top_bar.addWidget(self.btn_reset_all_sd)
        
        top_bar.addStretch() # Araya boşluk at (Butonlar sola, Compare sağa)
        
        # top_bar'a ekle
        top_bar.addWidget(self.btn_evolution)
        top_bar.addWidget(self.btn_compare)
        top_bar.addWidget(self.btn_compare_clear)

        # 2. SEKMELER (ORTA ALAN)
        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self.on_tab_changed)

        # 3. ALT BAR (SLIDER BURAYA GELİYOR)
        bottom_bar = QHBoxLayout()
        bottom_bar.setContentsMargins(10, 5, 10, 5) # Biraz kenar boşluğu
        #bottom_bar.setStyleSheet("background-color: #1e1e1e; border-top: 1px solid #333;") # Alt tarafı görsel olarak ayır

        lbl_zoom_icon = QLabel("🔍")
        lbl_zoom_min = QLabel("Küçük")
        lbl_zoom_max = QLabel("Büyük")
        lbl_zoom_min.setStyleSheet("color: #888; font-size: 11px;")
        lbl_zoom_max.setStyleSheet("color: #888; font-size: 11px;")

        self.slider_size = QSlider(Qt.Orientation.Horizontal)
        self.slider_size.setRange(100, 900)
        self.slider_size.setValue(self.thumbnail_size)
        self.slider_size.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_size.setTickInterval(50)
        self.slider_size.setSingleStep(25)
        self.slider_size.setFixedWidth(300)

        # Flicker önleyici bağlantılar (Sürüklerken Hızlı, Bırakınca Kaliteli)
        self.slider_size.valueChanged.connect(lambda: self.resize_thumbnails(fast=True))
        self.slider_size.sliderReleased.connect(lambda: self.resize_thumbnails(fast=False))

        # Alt Bar Yerleşimi
        bottom_bar.addStretch() # Sola boşluk (Her şeyi ortalamak için)
        bottom_bar.addWidget(lbl_zoom_icon)
        bottom_bar.addSpacing(10) # Simge ile yazı arasına az boşluk
        bottom_bar.addWidget(lbl_zoom_min)
        # '1' parametresini kaldırdık, artık esnemeyecek
        bottom_bar.addWidget(self.slider_size) 
        bottom_bar.addWidget(lbl_zoom_max)
        bottom_bar.addStretch() # Sağa boşluk

        # SAĞ PANEL YERLEŞİMİ BİRLEŞTİRME
        right_layout.addLayout(top_bar)    # En üstte butonlar
        right_layout.addWidget(self.tabs)  # Ortada galeri
        right_layout.addLayout(bottom_bar) # En altta slider
        
        right_panel = QWidget()
        right_panel.setLayout(right_layout)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        self.refresh_athlete_list(auto_select_first=True)

    def perform_backup(self):
        """Program kapanırken veritabanını yedekler ve eski yedekleri temizler."""
        import shutil
        import glob
        import os
        from datetime import datetime

        # Klasör Tanımları
        base_dir = os.path.dirname(os.path.abspath(__file__))
        backup_dir = os.path.join(base_dir, "backups")
        db_file = os.path.join(base_dir, "bodybuilding.db") # Senin DB dosyanın adı neyse
        
        # Eğer veritabanı dosyası yoksa (ilk açılışsa) çık
        if not os.path.exists(db_file):
            return

        # Yedek klasörünü oluştur
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)

        # 1. YENİ YEDEK OLUŞTUR
        # İsim formatı: backup_2023-10-27_15-30-00.db
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        backup_name = f"backup_{timestamp}.db"
        target_path = os.path.join(backup_dir, backup_name)
        
        try:
            shutil.copy2(db_file, target_path)
            print(f"✅ Yedek alındı: {backup_name}")
        except Exception as e:
            print(f"❌ Yedekleme hatası: {e}")

        # 2. ESKİ YEDEKLERİ TEMİZLE (Son 10 tanesi kalsın)
        try:
            # Tüm .db yedeklerini listele
            backups = sorted(glob.glob(os.path.join(backup_dir, "backup_*.db")))
            
            # Eğer 10'dan fazla varsa, en eskileri sil
            max_backups = 10
            if len(backups) > max_backups:
                to_delete = backups[: -max_backups] # Sondan 10 tanesi HARİÇ hepsini al
                for f in to_delete:
                    os.remove(f)
                    print(f"🗑️ Eski yedek silindi: {os.path.basename(f)}")
        except Exception as e:
            print(f"Temizlik hatası: {e}")

    def open_evolution_viewer(self):
        """Seçili fotoğrafları Morph analizine gönderir."""
        # 1. Seçili (Karşılaştır tikli) fotoğrafları bul
        selected_data = []
        
        # compare_selection listesi {photo_id: data} formatında
        if len(self.compare_selection) < 2:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Yetersiz Seçim", "Gelişim analizi için en az 2 fotoğrafı 'Seç' kutucuğuyla işaretleyin.")
            return

        # Verileri hazırla
        for pid, data in self.compare_selection.items():
            path = data['image_path']
            date_str = data['date']
            comp = data['competition']
            
            pix = QPixmap(path)
            if not pix.isNull():
                selected_data.append((pix, date_str, comp))
        
        # Diyaloğu aç
        if len(selected_data) >= 2:
            dlg = EvolutionViewerDialog(selected_data, self)
            # Modeless aç ki ana pencereyi engellemesin
            self.active_dialogs.append(dlg)
            dlg.finished.connect(lambda: self.active_dialogs.remove(dlg) if dlg in self.active_dialogs else None)
            dlg.show()

    def closeEvent(self, event):
        """Program kapatılırken tetiklenir."""
        
        # Kullanıcıya sormak istersen (Opsiyonel, bence gerek yok direkt kapansın)
        # reply = QMessageBox.question(self, 'Çıkış', "Programdan çıkmak istiyor musunuz?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        # if reply == QMessageBox.StandardButton.No:
        #     event.ignore()
        #     return

        # Durum çubuğuna bilgi ver (Kullanıcı görsün)
        self._status.showMessage("💾 Veritabanı yedekleniyor...", 2000)
        
        # Arayüzün donmaması için processEvents diyebiliriz ama işlem çok kısa sürer
        QApplication.processEvents()
        
        # Yedeği Al
        self.perform_backup()
        
        # Veritabanı bağlantısını güvenle kapat
        self.db.close()
        
        event.accept()

    def open_competition_manager(self):
        dlg = CompetitionManagerDialog(self.db, self)
        dlg.exec()

    def filter_athlete_list(self, text):
        """Arama kutusuna yazılan isme göre listeyi filtreler."""
        search_text = text.lower().strip()

        for i in range(self.list_athletes.count()):
            item = self.list_athletes.item(i)
            # İsmin içinde aranan kelime geçiyor mu?
            if search_text in item.text().lower():
                item.setHidden(False) # Göster
            else:
                item.setHidden(True)  # Gizle

    def refresh_quick_access_list(self):
        """Favori sporcuları yıldızlı kısa listede gösterir."""
        if not hasattr(self, "list_quick_access"):
            return

        self.list_quick_access.clear()
        favorites = self.db.get_favorite_athletes()

        for aid, name, division, _ in favorites:
            label = f"⭐ {name}"
            if division != self.current_division:
                label = f"⭐ {name} ({division})"

            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, (aid, division))
            self.list_quick_access.addItem(item)

        has_items = self.list_quick_access.count() > 0
        self.lbl_quick_hint.setVisible(not has_items)

    def jump_to_quick_access_item(self, item):
        """Hızlı erişimdeki bir sporcuya tıklandığında listeyi ve galeriyi aç."""
        data = item.data(Qt.ItemDataRole.UserRole)
        if not data:
            return

        target_id, target_division = data

        if self.current_division != target_division:
            # Kategori değiştirmeyi sessiz yap (loop olmasın)
            self.combo_division.blockSignals(True)
            self.combo_division.setCurrentText(target_division)
            self.combo_division.blockSignals(False)
            self.current_division = target_division
            self.db.set_setting("last_division", target_division)
            self.refresh_athlete_list(auto_select_first=False)

        if not self.select_athlete_by_id(target_id):
            self._status.showMessage("⚠️ Sporcu listede bulunamadı, liste güncellendi.", 3000)
        else:
            self._status.showMessage("⭐ Hızlı erişim: sporcuya geçildi.", 2000)

    def open_quick_access_menu(self, pos):
        item = self.list_quick_access.itemAt(pos)
        if not item:
            return

        data = item.data(Qt.ItemDataRole.UserRole)
        if not data:
            return

        athlete_id, _ = data

        row = self.db.get_athlete(athlete_id)
        athlete_name = row[1] if row else item.text().replace("⭐", "").strip()

        menu = QMenu(self)
        act_copy = QAction("📋 İsmi Kopyala", self)
        act_copy.triggered.connect(lambda: QApplication.clipboard().setText(athlete_name))
        act_remove = QAction("🗑️ Hızlı Erişimden Çıkar", self)
        act_remove.triggered.connect(lambda: self.remove_from_quick_access(athlete_id))
        menu.addAction(act_copy)
        menu.addAction(act_remove)
        menu.exec(self.list_quick_access.mapToGlobal(pos))

    def add_to_quick_access(self, athlete_id):
        if not athlete_id:
            return
        self.db.add_favorite_athlete(athlete_id)
        self.refresh_quick_access_list()

        row = self.db.get_athlete(athlete_id)
        name = row[1] if row else ""
        self._status.showMessage(f"⭐ {name} hızlı erişime eklendi.", 2000)

    def remove_from_quick_access(self, athlete_id):
        if not athlete_id:
            return
        self.db.remove_favorite_athlete(athlete_id)
        self.refresh_quick_access_list()
        self._status.showMessage("🗑️ Hızlı erişimden çıkarıldı.", 2000)

    def select_athlete_by_id(self, athlete_id):
        """Listedeki ID'ye göre sporcuyu seç ve detaylarını yükle."""
        for i in range(self.list_athletes.count()):
            item = self.list_athletes.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == athlete_id:
                self.list_athletes.setCurrentRow(i)
                self.load_athlete_details(item)
                return True
        return False

    def favorite_competition(self, year, comp_name):
        """Belirli yıl için yarışmayı favori yapar ve üstte tutar."""
        if not self.current_athlete_id or not year or not comp_name:
            return
        self.db.set_favorite_competition(self.current_athlete_id, year, comp_name)
        self.refresh_gallery()
        self._status.showMessage(f"⭐ {year} favori: {comp_name}", 2000)

    def unfavorite_competition(self, year, comp_name):
        """Favori yarışmayı temizler (eğer mevcut favori buysa)."""
        if not self.current_athlete_id or not year:
            return
        current = self.db.get_favorite_competitions(self.current_athlete_id).get(str(year))
        if current == comp_name:
            self.db.remove_favorite_competition(self.current_athlete_id, year)
            self.refresh_gallery()
            self._status.showMessage("⭐ Favori kaldırıldı.", 2000)

    def export_athletes_csv(self):
        """Mevcut kategorideki veya tüm sporcuları CSV olarak kaydeder."""
        import csv
        
        # Kayıt yerini sor
        path, _ = QFileDialog.getSaveFileName(self, "Listeyi Dışa Aktar", "Sporcu_Listesi.csv", "CSV Dosyası (*.csv)")
        if not path:
            return

        try:
            # Veritabanından verileri çek
            cur = self.db.conn.cursor()
            # İstersen sadece şu anki kategori: WHERE division=?
            # Ama genelde hepsini almak istenir:
            cur.execute("SELECT name, division, total_score FROM athletes ORDER BY division, name")
            rows = cur.fetchall()

            # Dosyayı yaz (utf-8-sig: Excel'in Türkçe karakterleri tanıması için şart)
            with open(path, mode='w', newline='', encoding='utf-8-sig') as file:
                writer = csv.writer(file, delimiter=';') # Excel için noktalı virgül daha güvenli
                # Başlıklar
                writer.writerow(["Isim", "Kategori", "Puan"])
                
                for name, div, score in rows:
                    writer.writerow([name, div, score])

            self._status.showMessage(f"✅ Liste başarıyla kaydedildi: {path}", 4000)
            
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Hata", f"Dışa aktarma başarısız:\n{e}")

    def import_athletes_csv(self):
        """CSV dosyasından sporcuları okur ve veritabanına ekler."""
        import csv
        
        path, _ = QFileDialog.getOpenFileName(self, "Listeyi İçe Aktar", "", "CSV Dosyası (*.csv);;Metin Dosyası (*.txt)")
        if not path:
            return

        added_count = 0
        skipped_count = 0
        
        try:
            with open(path, mode='r', encoding='utf-8-sig') as file:
                # Ayırıcıyı otomatik tahmin etmeye çalış (virgül mü noktalı virgül mü?)
                sample = file.read(1024)
                file.seek(0)
                dialect = csv.Sniffer().sniff(sample)
                
                reader = csv.reader(file, dialect)
                
                # İlk satırı başlık varsayıp atla (Opsiyonel, kontrol edelim)
                header = next(reader, None)
                
                # Veritabanı işlemleri
                cur = self.db.conn.cursor()
                
                for row in reader:
                    # En az 1 sütun (İsim) olmalı
                    if not row or not row[0].strip():
                        continue
                        
                    name = row[0].strip()
                    
                    # 2. sütun varsa Kategori, yoksa şu anki seçili kategori
                    division = row[1].strip() if len(row) > 1 and row[1].strip() else self.current_division
                    
                    # 3. sütun varsa Puan (Opsiyonel)
                    score = 0
                    if len(row) > 2:
                        try: score = float(row[2])
                        except: pass

                    # Aynı isimde biri var mı kontrol et (Çift kayıt olmasın)
                    cur.execute("SELECT id FROM athletes WHERE name=? AND division=?", (name, division))
                    if cur.fetchone():
                        skipped_count += 1
                        continue

                    # Ekle
                    cur.execute("INSERT INTO athletes (name, division, total_score) VALUES (?, ?, ?)", (name, division, score))
                    added_count += 1

                self.db.conn.commit()
                
            # Listeyi yenile
            self.refresh_athlete_list()
            
            msg = f"✅ {added_count} yeni sporcu eklendi."
            if skipped_count > 0:
                msg += f" ({skipped_count} kayıt zaten vardı, atlandı.)"
            
            self._status.showMessage(msg, 5000)
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, "İçe Aktarma Tamamlandı", msg)

        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Hata", f"Dosya okunamadı:\n{e}\n\nLütfen formatın 'İsim;Kategori' şeklinde olduğundan emin olun.")


    def batch_change_resolution(self, target_hd):
        count = len(self.photo_widgets)
        if count == 0: return

        mode_text = "HD (Orijinal)" if target_hd else "SD (Thumbnail)"
        self._status.showMessage(f"⏳ Tüm fotoğraflar {mode_text} moduna geçiriliyor...", 0)
        
        # İşlem sırasında arayüz donmasın diye processEvents kullanıyoruz
        for i, widget in enumerate(self.photo_widgets):
            widget.force_resolution(target_hd)
            
            # Her 5 fotoda bir arayüzü güncelle (Çok sık yaparsak yavaşlar)
            if i % 5 == 0:
                QApplication.processEvents()
        
        self._status.showMessage(f"✅ Tüm fotoğraflar {mode_text} yapıldı.", 3000)

    def emergency_reset_sd(self):
        """Hafızadaki tüm HD kayıtlarını siler ve mevcut ekranı SD'ye çeker."""
        # 1. Hafızayı temizle
        self.hd_memory.clear()
        
        # 2. Şu an ekranda ne varsa SD'ye zorla
        self.batch_change_resolution(False)
        
        self._status.showMessage("🧹 RAM Temizlendi: Tüm fotoğraflar SD moduna alındı.", 3000)

    def open_settings(self):
            dlg = SettingsDialog(self.db, self)
            dlg.exec()

    def on_tab_changed(self, index):
        # Eğer sekmeler temizleniyorsa (index -1 olur) veya sporcu seçili değilse işlem yapma
        if index == -1 or not self.current_athlete_id:
            return
        
        # O an açık olan sekmenin başlığını (Yılı) al
        year_text = self.tabs.tabText(index)
        
        # Hafızaya sporcu ID'si ile kaydet
        if year_text:
            self.athlete_year_memory[self.current_athlete_id] = year_text

    def toggle_compare_photo(self, photo_id, checked, data_packet):
        if checked:
            # ESKİ KOD BUYDU (Hata veren kısım):
            # if data_packet['pixmap'] and not data_packet['pixmap'].isNull():
            
            # YENİ KOD (Dosya yolu kontrolü):
            if 'image_path' in data_packet and data_packet['image_path']:
                self.compare_selection[photo_id] = data_packet
        else:
            self.compare_selection.pop(photo_id, None)
        
        self.update_compare_button_text()

    def update_compare_button_text(self):
        n = len(self.compare_selection)
        if n == 0:
            self.btn_compare.setText(f"⚔️ Karşılaştır ({n})")
            self.btn_compare.setEnabled(True)  # istersen False da yapabilirsin
        else:
            self.btn_compare.setText(f"⚔️ Karşılaştır ({n})")

    def clear_compare_selection(self):
        self.compare_selection.clear()
        self.update_compare_button_text()
        # mevcut galerideki checkboxları da kapat
        for w in self.photo_widgets:
            w.chk_compare.blockSignals(True)
            w.chk_compare.setChecked(False)
            w.chk_compare.blockSignals(False)

    def open_main_context_menu(self, pos):
            if not self.current_athlete_id:
                return

            menu = QMenu(self)

            # Mevcut butona dokunmuyoruz
            act_paste = QAction("📌 Resim Yapıştır (Detaylı)", self)
            act_paste.triggered.connect(self.paste_single_image_from_menu)
            menu.addAction(act_paste)
            
            # --- YENİ EKLENEN BUTON ---
            menu.addSeparator()
            # İpucu: Eğer hafızada bir yer varsa parantez içinde gösterelim
            context_hint = ""
            if self.last_context_comp:
                context_hint = f" -> {self.last_context_comp}"
            
            act_paste_direct = QAction(f"🚀 Resmi Buraya Yapıştır (Hızlı){context_hint}", self)
            act_paste_direct.triggered.connect(self.paste_image_direct)
            menu.addAction(act_paste_direct)
            # ---------------------------

            menu.exec(self.mapToGlobal(pos))

# MainWindow sınıfının içine ekleyin:
    def paste_to_target(self, target_date, target_comp):
        """Doğrudan belirtilen tarih ve yarışmaya yapıştırır."""
        if not self.current_athlete_id:
            return

        clipboard = QApplication.clipboard()
        if not clipboard: return
        pixmap = clipboard.pixmap()
        
        if pixmap.isNull():
            self._status.showMessage("⚠️ Panoda resim yok.", 2000)
            return

        # Hafızayı da güncelle (Böylece sonraki Ctrl+V'ler de buraya gider)
        self.last_context_date = target_date
        self.last_context_comp = target_comp

        # Temp kaydet
        if not os.path.exists("temp_cache"): os.makedirs("temp_cache")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        temp_path = os.path.join("temp_cache", f"target_{timestamp}.png")
        pixmap.save(temp_path, "PNG")

        # Veritabanına yaz
        ok = self.db.add_photo(self.current_athlete_id, target_date, target_comp, temp_path) 
 
        if ok:
            try: os.remove(temp_path) 
            except: pass
            self._status.showMessage(f"✅ Eklendi: {target_comp}", 3000)
            self.refresh_gallery()
        else:
            self._status.showMessage("❌ Hata oluştu.", 3000)


    def paste_image_direct(self):
            """Panodaki resmi soru sormadan en son kullanılan veya en son DB kaydındaki konuma yapıştırır."""
            if not self.current_athlete_id:
                self._status.showMessage("⚠️ Önce sporcu seçin.", 3000)
                return

            # 1. Panodan resmi al
            clipboard = QApplication.clipboard()
            if not clipboard: return
            pixmap = clipboard.pixmap()
            if pixmap.isNull():
                self._status.showMessage("⚠️ Panoda resim yok.", 2000)
                return

            # 2. Hedef Konumu Belirle (Mantık: Hafıza > DB Son Kayıt > Varsayılan)
            target_date = None
            target_comp = None

            # A) Hafızada bu oturumda işlem yapılmış mı?
            if self.last_context_date and self.last_context_comp:
                target_date = self.last_context_date
                target_comp = self.last_context_comp
            
            # B) Hafıza boşsa, veritabanından bu sporcunun son fotoğrafına bak
            else:
                cur = self.db.conn.cursor()
                cur.execute("SELECT date, competition FROM photos WHERE athlete_id=? ORDER BY id DESC LIMIT 1", (self.current_athlete_id,))
                row = cur.fetchone()
                if row:
                    target_date, target_comp = row
                else:
                    # C) Hiçbir kayıt yoksa bugüne at
                    target_date = datetime.datetime.now().strftime("%Y-%m-%d")
                    target_comp = "Yeni Eklenenler"

            # 3. Resmi Temp olarak kaydet (Senin db.add_photo fonksiyonun dosya yolu istiyor)
            if not os.path.exists("temp_cache"): os.makedirs("temp_cache")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            temp_path = os.path.join("temp_cache", f"direct_{timestamp}.png")
            pixmap.save(temp_path, "PNG")

            # 4. Veritabanına kaydet
            ok = self.db.add_photo(self.current_athlete_id, target_date, target_comp, temp_path)
            
            if ok:
                # Temp dosyasını temizle
                try: os.remove(temp_path) 
                except: pass
                
                self._status.showMessage(f"✅ Hızlı Eklendi: {target_comp}", 3000)
                self.refresh_gallery()
            else:
                self._status.showMessage("❌ Kayıt hatası oluştu.", 3000)

    def paste_single_image_from_menu(self):
        if not self.current_athlete_id:
            return

        clipboard = QApplication.clipboard()
        if clipboard is None:
            return
        
        mime = clipboard.mimeData()
        if mime is None:
            return
        
        if not mime.hasImage():
            self._status.showMessage("⚠️ Panoda resim yok.", 2000)
            return

        pixmap = clipboard.pixmap()
        if pixmap.isNull():
            return

        import datetime, os
        os.makedirs("temp_cache", exist_ok=True)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        temp_path = os.path.join("temp_cache", f"paste_{ts}.png")
        pixmap.save(temp_path, "PNG")

        self.process_batch_images([temp_path])

    def open_athlete_context_menu(self, pos):
        item = self.list_athletes.itemAt(pos)
        if not item:
            return

        self.list_athletes.setCurrentItem(item)
        self.current_athlete_id = item.data(Qt.ItemDataRole.UserRole)
        #athlete_link = self.db.get_athlete_link(current_athlete_id)
        athlete_name = item.text()
        clean_name = athlete_name
        if "]" in athlete_name:
            # "]" karakterinden sonrasını al ve boşlukları temizle
            clean_name = athlete_name.split("]", 1)[1].strip()

        menu = QMenu(self)
        act_edit = QAction("✏️ İsmi Düzenle", self)
        act_delete = QAction("🗑️ Sporcuyu Sil", self)
        act_copy = QAction("📋 İsmi Kopyala", self)

        act_edit.triggered.connect(self.edit_current_athlete_name)
        act_delete.triggered.connect(self.delete_current_athlete)
        act_copy.triggered.connect(lambda: QApplication.clipboard().setText(clean_name))

        if self.db.is_favorite_athlete(self.current_athlete_id):
            act_quick = QAction("⭐ Hızlı Erişimden Çıkar", self)
            act_quick.triggered.connect(lambda: self.remove_from_quick_access(self.current_athlete_id))
        else:
            act_quick = QAction("⭐ Hızlı Erişime Ekle", self)
            act_quick.triggered.connect(lambda: self.add_to_quick_access(self.current_athlete_id))

        menu.addAction(act_copy)
        menu.addAction(act_edit)
        menu.addAction(act_quick)
        menu.addAction(act_delete)

        menu.addSeparator()
        open_link_action = menu.addAction("🌐 Bağlantıyı Aç")
        #open_link_action.setEnabled(bool(athlete_link)) # Link yoksa tıklanamaz
        #open_link_action.triggered.connect(lambda: QDesktopServices.openUrl(QUrl(athlete_link)))

        copy_link_action = menu.addAction("🔗 Bağlantıyı Kopyala")
        #copy_link_action.setEnabled(bool(athlete_link))
        #copy_link_action.triggered.connect(lambda: QApplication.clipboard().setText(athlete_link))

        edit_link_action = menu.addAction("🔗 Bağlantıyı Düzenle/Ekle")
        #edit_link_action.triggered.connect(lambda: self.edit_athlete_link(athlete_id, athlete_name, athlete_link))

        menu.exec(self.list_athletes.mapToGlobal(pos))

    def select_first_athlete_and_load(self):
        """Listede atlet varsa ilkini seçer ve galeriye yükler."""
        if self.list_athletes.count() == 0:
            self.current_athlete_id = None
            self.tabs.clear()
            self.btn_add_photo.setEnabled(False)
            self.btn_score.setEnabled(False)
            return

        self.list_athletes.setCurrentRow(0)
        first_item = self.list_athletes.item(0)
        self.load_athlete_details(first_item)

    def open_coeff_settings(self):
        dlg = CoefficientSettingsDialog(self.db, self.current_division, self)
        dlg.exec()

    def change_division(self, division_name):
        self.current_division = division_name
        
        # Hata aldığın satır: Şimdi DatabaseManager'a eklediğimiz için çalışacak
        self.db.set_setting("last_division", division_name)
        
        # Arama kutusu henüz oluşturulmamışsa (başlangıç aşamasında) hata vermemesi için kontrol:
        if hasattr(self, 'txt_search'):
            self.txt_search.clear()
        
        self.refresh_athlete_list(auto_select_first=True)


    def refresh_athlete_list(self, auto_select_first=True):
        self.list_athletes.clear()
        athletes = self.db.get_athletes_by_division(self.current_division)

        for aid, name, score in athletes:
            score_text = f"[{score:.0f}]" if score > 0 else "[-]"
            item_text = f"{score_text}  {name}"
            self.list_athletes.addItem(item_text)
            it = self.list_athletes.item(self.list_athletes.count() - 1)
            if it is not None:
                it.setData(Qt.ItemDataRole.UserRole, aid)

        self.refresh_quick_access_list()

        if auto_select_first:
            self.select_first_athlete_and_load()


    def load_athlete_details(self, item):
        try:
            self.current_athlete_id = item.data(Qt.ItemDataRole.UserRole)
            self.btn_add_photo.setEnabled(True)
            self.btn_score.setEnabled(True)
            self.refresh_gallery()
        except: pass

    def open_scoring(self):
        if not self.current_athlete_id:
            self._status.showMessage("⚠️ Önce sporcu seçin.", 3000)
            return

        row = self.db.get_athlete(self.current_athlete_id)
        if not row:
            self._status.showMessage("⚠️ Sporcu bulunamadı.", 3000)
            return

        _, name, division, _ = row

        dlg = ScoringDialog(self.db, self.current_athlete_id, name, division, self)

        # Kaydedince listeyi güncelle, seçim düşerse geri seç
        def after_save():
            self.refresh_athlete_list()
            self.reselect_current_athlete()

        dlg.score_saved.connect(after_save)

        # Dialog kapanınca listeden çıkar (GC için)
        self.active_dialogs.append(dlg)
        dlg.finished.connect(lambda: self.active_dialogs.remove(dlg) if dlg in self.active_dialogs else None)

        dlg.setModal(False)   # non-modal
        dlg.show()            # exec() YOK
        dlg.raise_()
        dlg.activateWindow()

    def reselect_current_athlete(self):
        if not self.current_athlete_id:
            return
        for i in range(self.list_athletes.count()):
            it = self.list_athletes.item(i)
            if it is None:
                continue
            if it.data(Qt.ItemDataRole.UserRole) == self.current_athlete_id:
                self.list_athletes.setCurrentRow(i)
                return

    def add_new_athlete(self):
        from PyQt6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Ekle", f"{self.current_division} Kategorisine Eklenecek İsim:")
        if ok and name:
            self.db.add_athlete(name, self.current_division)
            self.refresh_athlete_list()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event is None:
            return
        md = event.mimeData()
        if md is None:
            event.ignore()
            return
        if md.hasUrls() or md.hasImage():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        if event is None:
            return

        md = event.mimeData()
        if md is None:
            return

        if not self.current_athlete_id:
            self._status.showMessage("⚠️ Önce sporcu seçin.", 3000)
            return

        files_to_process = []

        if md.hasUrls():
            for qurl in md.urls():
                url_str = qurl.toString()
                if qurl.isLocalFile():
                    local_path = qurl.toLocalFile()
                    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
                    if local_path.lower().endswith(valid_ext):
                        files_to_process.append(local_path)
                elif url_str.startswith('http'):
                    try:
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        ext = ".jpg"
                        if ".png" in url_str.lower():
                            ext = ".png"

                        temp_dir = "temp_web_downloads"
                        os.makedirs(temp_dir, exist_ok=True)

                        temp_path = os.path.join(temp_dir, f"web_img_{timestamp}{ext}")
                        opener = urllib.request.build_opener()
                        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
                        urllib.request.install_opener(opener)
                        urllib.request.urlretrieve(url_str, temp_path)
                        files_to_process.append(temp_path)
                    except Exception as e:
                        self._status.showMessage(f"⚠️ Web görsel indirilemedi: {e}", 5000)

        if files_to_process:
            self.process_batch_images(files_to_process)

    def edit_current_athlete_name(self):
        if not self.current_athlete_id:
            return

        row = self.db.get_athlete(self.current_athlete_id)
        if not row:
            return

        _, current_name, _, _ = row

        new_name, ok = QInputDialog.getText(
            self,
            "Atlet İsmini Düzenle",
            "Yeni isim:",
            text=current_name
        )

        if not ok:
            return

        new_name = new_name.strip()
        if not new_name or new_name == current_name:
            return

        self.db.update_athlete_name(self.current_athlete_id, new_name)
        self.refresh_athlete_list()
        self.reselect_current_athlete()

        self._status.showMessage("✏️ Atlet ismi güncellendi.", 2000)
    

    def delete_current_athlete(self):
        if not self.current_athlete_id:
            return

        dlg = SilentConfirmDialog("Seçili sporcu silinsin mi?", self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        self.db.delete_athlete(self.current_athlete_id)
        self.current_athlete_id = None
        self.tabs.clear()
        self.btn_add_photo.setEnabled(False)
        self.btn_score.setEnabled(False)
        self.refresh_athlete_list()

    
    def upload_photo(self):
        if not self.current_athlete_id: 
            self._status.showMessage("⚠️ Önce sporcu seçin.", 3000)
            return
        
        # 1. Sporcu İsmini Al
        athlete_name = "Bilinmiyor"
        row = self.db.get_athlete(self.current_athlete_id)
        if row:
            athlete_name = row[1]

        # 2. Diyaloğu Oluştur (parent=None yaparak bağımsız pencere yapıyoruz)
        # file_paths=None diyerek Manuel Mod'u açıyoruz
        dlg = PhotoDetailsDialog(self.db, self.current_athlete_id, athlete_name=athlete_name, file_paths=None, parent=None)
        
        # Eğer pencerenin görev çubuğunda da ayrı bir simge olarak görünmesini istersen:
        dlg.setWindowFlags(Qt.WindowType.Window) 

        def on_success():
            if dlg.final_date and dlg.final_comp:
                self.last_context_date = dlg.final_date
                self.last_context_comp = dlg.final_comp
            self.refresh_gallery()
            
        dlg.saved_successfully.connect(on_success)
        
        # GC (Çöp toplayıcı) silmesin diye listeye ekle
        self.active_dialogs.append(dlg)
        dlg.finished.connect(lambda: self.active_dialogs.remove(dlg) if dlg in self.active_dialogs else None)
        
        dlg.show()

    def paste_from_clipboard(self):
        if not self.current_athlete_id:
            self._status.showMessage("⚠️ Önce sporcu seçin.", 3000)
            return
        clipboard = QApplication.clipboard()
        if clipboard is None:
            return
        mime_data = clipboard.mimeData()
        if mime_data is None:
            return
        if mime_data.hasImage():
            pixmap = clipboard.pixmap()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if not os.path.exists("temp_cache"): os.makedirs("temp_cache")
            temp_path = os.path.join("temp_cache", f"temp_{timestamp}.png")
            pixmap.save(temp_path, "PNG")
            self.process_batch_images([temp_path])

    def process_batch_images(self, file_paths):
            dlg = PhotoDetailsDialog(self.db, self.current_athlete_id, file_paths=file_paths)
            dlg.saved_successfully.connect(self.refresh_gallery)
            
            # --- EKLENECEK KISIM ---
            # Dialog başarıyla kapanırsa hafızayı güncelle
            def update_memory():
                if dlg.final_date and dlg.final_comp:
                    self.last_context_date = dlg.final_date
                    self.last_context_comp = dlg.final_comp
            
            dlg.saved_successfully.connect(update_memory)
            # -----------------------

            self.active_dialogs.append(dlg)
            dlg.finished.connect(lambda: self.active_dialogs.remove(dlg) if dlg in self.active_dialogs else None)
            dlg.show()

    def resize_thumbnails(self, fast=False):
        self.thumbnail_size = self.slider_size.value()
        # Fast parametresini widget'a iletiyoruz
        for widget in self.photo_widgets:
            widget.update_size(self.thumbnail_size, fast=fast)

    def _handle_ctrl_zoom(self, event):
        """Ctrl + Mouse wheel ile zoom'u tek noktadan yönetir. True dönerse event tüketilir."""
        if event is None:
            return False
        if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            return False

        delta = event.angleDelta().y()
        if delta == 0 and event.pixelDelta():
            delta = event.pixelDelta().y()

        if delta == 0:
            return False

        step = self.slider_size.singleStep() or 25
        direction = 1 if delta > 0 else -1
        new_val = self.slider_size.value() + direction * step
        new_val = max(self.slider_size.minimum(), min(self.slider_size.maximum(), new_val))

        if new_val != self.slider_size.value():
            # Sinyalleri bloklayıp direkt smooth resize yapıyoruz (flicker olmasın)
            self.slider_size.blockSignals(True)
            self.slider_size.setValue(new_val)
            self.slider_size.blockSignals(False)
            self.resize_thumbnails(fast=False)

        event.accept()
        return True

    def wheelEvent(self, event):
        """Ctrl + Mouse wheel ile zoom (flicker ve kayma olmadan)."""
        if self._handle_ctrl_zoom(event):
            return
        super().wheelEvent(event)

    def eventFilter(self, obj, event):
        # ScrollArea veya viewport üzerinde Ctrl+Wheel yapılırsa ana zoom fonksiyonuna yönlendir
        if event.type() == QEvent.Type.Wheel:
            mods = event.modifiers()
            if self._handle_ctrl_zoom(event):
                return True

            if (mods & Qt.KeyboardModifier.ShiftModifier) and not (mods & Qt.KeyboardModifier.ControlModifier):
                # Shift + wheel ile yatay kaydır
                delta = event.angleDelta().y()
                if delta == 0 and event.pixelDelta():
                    delta = event.pixelDelta().y()

                target_scroll = obj if isinstance(obj, QScrollArea) else obj.parent()
                if isinstance(target_scroll, QScrollArea):
                    bar = target_scroll.horizontalScrollBar()
                    if bar:
                        step = max(bar.singleStep(), 40) * 2  # hızlandırılmış yatay kaydırma
                        direction = -1 if delta > 0 else 1  # Wheel up -> sola kay
                        bar.setValue(bar.value() + direction * step)
                        event.accept()
                        return True
        return super().eventFilter(obj, event)

    def refresh_gallery(self):
        # 1. Sinyalleri durdur
        self.tabs.blockSignals(True)
        self.tabs.clear()
        self.photo_widgets = []

        # 2. Sporcu ismi
        athlete_name = "Bilinmiyor"
        if self.current_athlete_id:
            row = self.db.get_athlete(self.current_athlete_id)
            if row:
                athlete_name = row[1]

        # 3. DB'den fotoları çek (ARTIK 6 SÜTUN GELİYOR)
        photos = self.db.get_photos(self.current_athlete_id)
        favorites_by_year = self.db.get_favorite_competitions(self.current_athlete_id) if self.current_athlete_id else {}

        # 4. Fotoları YILLARA göre grupla
        photos_by_year = {}

        # --- HATA BURADAYDI: Değişken sayısını 6'ya çıkardık ---
        for date_str, comp, thumb_path, pid, rank, image_path in photos:
            year = date_str.split("-")[0]
            # Listeye hepsini ekliyoruz
            photos_by_year.setdefault(year, []).append((date_str, comp, thumb_path, pid, rank, image_path))

        # 5. Yılları Yeniden Eskiye Sırala
        for year in sorted(photos_by_year.keys(), reverse=True):
            year_tab = QWidget()
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.installEventFilter(self)
            scroll.viewport().installEventFilter(self)
            content_widget = QWidget()

            main_layout = QVBoxLayout()
            main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

            year_photos = photos_by_year[year]
            photos_by_comp = {}

            # Yarışma ismine göre grupla
            for p in year_photos:
                comp_name = p[1]
                photos_by_comp.setdefault(comp_name, []).append(p)

            # Yarışmaları tarihe göre sırala
            sorted_comps = sorted(photos_by_comp.items(), key=lambda x: x[1][0][0], reverse=True)

            # Favori yarışma varsa aynı yıl için en üste çek
            fav_comp = favorites_by_year.get(year)
            if fav_comp:
                for idx, (cname, cphotos) in enumerate(sorted_comps):
                    if cname == fav_comp:
                        sorted_comps.insert(0, sorted_comps.pop(idx))
                        break

            for comp_name, comp_photos in sorted_comps:
                ref_date = comp_photos[0][0]

                is_favorite = (favorites_by_year.get(year) == comp_name)
                date_label = f" ({ref_date})" if ref_date else ""
                display_title = f"🏆 {comp_name}{date_label}"
                if is_favorite:
                    display_title = f"⭐ {display_title}"

                group_box = CompetitionGroupBox(display_title, ref_date, comp_name, self, is_favorite=is_favorite, year=year)
                group_box.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #aaa; margin-top: 10px; }")

                grid_layout = QGridLayout()
                grid_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

                row, col = 0, 0
                # --- BURADA DA UNPACK İŞLEMİNİ GÜNCELLİYORUZ ---
                for date_str, _, thumb_path, pid, rank, image_path in comp_photos:
                    
                    # PhotoWidget'ı YENİ parametrelerle oluşturuyoruz:
                    # (id, thumb_path, date, comp, rank, size, main_window, athlete_name, image_path)
                    p_widget = PhotoWidget(pid, thumb_path, date_str, comp_name, rank, self.thumbnail_size, self, athlete_name, image_path)
                    
                    self.photo_widgets.append(p_widget)

                    # Karşılaştırma seçimi varsa koru
                    if pid in self.compare_selection:
                        p_widget.chk_compare.blockSignals(True)
                        p_widget.chk_compare.setChecked(True)
                        p_widget.chk_compare.blockSignals(False)

                    grid_layout.addWidget(p_widget, row, col)
                    col += 1
                    if col >= 4:
                        col = 0
                        row += 1

                group_box.setLayout(grid_layout)
                main_layout.addWidget(group_box)

            content_widget.setLayout(main_layout)
            scroll.setWidget(content_widget)

            tab_layout = QVBoxLayout()
            tab_layout.addWidget(scroll)
            year_tab.setLayout(tab_layout)

            self.tabs.addTab(year_tab, year)

        # 6. Sinyalleri geri aç
        self.tabs.blockSignals(False)

        # 7. Hafızadan sekme geri yükleme
        target_year = self.athlete_year_memory.get(self.current_athlete_id)
        found_in_memory = False
        if target_year:
            for i in range(self.tabs.count()):
                if self.tabs.tabText(i) == target_year:
                    self.tabs.setCurrentIndex(i)
                    found_in_memory = True
                    break

        if not found_in_memory and self.tabs.count() > 0:
            self.tabs.setCurrentIndex(0)
            self.on_tab_changed(0)

    def open_comparison(self):
            # Sözlüklerin listesini gönderiyoruz (.values() dictionary listesi döndürür)
            data_list = list(self.compare_selection.values())
            if not data_list:
                self._status.showMessage("⚠️ Karşılaştırmak için foto seçin.", 3000)
                return
            CompareWindow(data_list).exec()


if __name__ == "__main__":
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
