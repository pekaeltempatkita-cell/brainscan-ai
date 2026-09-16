"""
database.py — Simpan & ambil riwayat hasil prediksi user (SQLite).
"""
import sqlite3
import json
from datetime import datetime
from contextlib import contextmanager

from config import DATABASE_PATH


@contextmanager
def get_connection():
    """Context manager biar koneksi DB selalu ditutup rapi, walau ada error."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row   # biar hasil query bisa diakses kayak dict
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _safe_add_column(conn, table, column, coltype):
    """ALTER TABLE yang aman dipanggil berkali-kali (skip kalau kolom sudah ada)."""
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
    except sqlite3.OperationalError as e:
        if "duplicate column" not in str(e).lower():
            raise


def init_db():
    """Bikin tabel kalau belum ada. Panggil ini SEKALI pas aplikasi start."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                upload_time TEXT NOT NULL,

                -- Hasil precheck (tahap 1: apakah ini gambar otak?)
                precheck_status TEXT NOT NULL,          -- 'valid' atau 'invalid'
                precheck_confidence REAL NOT NULL,

                -- Hasil klasifikasi (tahap 2: penyakit apa? NULL kalau precheck invalid)
                prediction_label TEXT,
                prediction_confidence REAL,
                all_probabilities TEXT,                 -- JSON string semua probabilitas per kelas

                -- Tambahan
                gradcam_path TEXT,                       -- path gambar heatmap (opsional)
                gemini_explanation TEXT                  -- penjelasan dari Gemini (opsional)
            )
        """)

        # --- Tabel data pasien ---
        conn.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nik TEXT UNIQUE NOT NULL,
                nama TEXT NOT NULL,
                tanggal_lahir TEXT,
                jenis_kelamin TEXT,
                alamat TEXT,
                no_telepon TEXT,
                created_at TEXT NOT NULL,
                created_by_email TEXT
            )
        """)

        # --- Tabel rekam medis (hubungin pasien <-> hasil prediksi) ---
        conn.execute("""
            CREATE TABLE IF NOT EXISTS medical_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                prediction_id INTEGER,
                catatan_dokter TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES patients(id),
                FOREIGN KEY (prediction_id) REFERENCES predictions(id)
            )
        """)

        # --- Migrasi tambahan: modalitas pemindaian & informed consent ---
        _safe_add_column(conn, "predictions", "modalitas", "TEXT")
        _safe_add_column(conn, "predictions", "sequence_type", "TEXT")
        _safe_add_column(conn, "predictions", "informed_consent", "INTEGER DEFAULT 0")
        _safe_add_column(conn, "predictions", "consent_note", "TEXT")
        _safe_add_column(conn, "predictions", "jenis_scan", "TEXT")
        _safe_add_column(conn, "predictions", "umur_saat_scan", "TEXT")
        _safe_add_column(conn, "predictions", "gejala", "TEXT")

        # --- Migrasi tambahan: verifikasi & pengesahan dokter ---
        _safe_add_column(conn, "medical_records", "status_verifikasi", "TEXT DEFAULT 'Pending Review'")
        _safe_add_column(conn, "medical_records", "dokter_nama", "TEXT")
        _safe_add_column(conn, "medical_records", "dokter_sip", "TEXT")
        _safe_add_column(conn, "medical_records", "tanggal_verifikasi", "TEXT")

    print("Database siap (tabel predictions, patients, medical_records sudah ada/dibuat).")


def save_prediction(filename, precheck_status, precheck_confidence,
                     prediction_label=None, prediction_confidence=None,
                     all_probabilities=None, gradcam_path=None, gemini_explanation=None,
                     modalitas=None, sequence_type=None, informed_consent=0, consent_note=None,
                     jenis_scan=None, umur_saat_scan=None, gejala=None):
    """Simpan 1 hasil prediksi ke database. Return id record yang baru dibuat."""
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO predictions (
                filename, upload_time, precheck_status, precheck_confidence,
                prediction_label, prediction_confidence, all_probabilities,
                gradcam_path, gemini_explanation, modalitas, sequence_type,
                informed_consent, consent_note, jenis_scan, umur_saat_scan, gejala
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            filename, datetime.now().isoformat(), precheck_status, precheck_confidence,
            prediction_label, prediction_confidence,
            json.dumps(all_probabilities) if all_probabilities else None,
            gradcam_path, gemini_explanation, modalitas, sequence_type,
            informed_consent, consent_note, jenis_scan, umur_saat_scan, gejala
        ))
        return cursor.lastrowid


def get_history(limit=50):
    """Ambil riwayat prediksi terbaru, urut dari yang paling baru."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM predictions ORDER BY upload_time DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(row) for row in rows]


def get_prediction_by_id(prediction_id):
    """Ambil 1 record spesifik (misal buat generate PDF laporan)."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM predictions WHERE id = ?", (prediction_id,)
        ).fetchone()
        return dict(row) if row else None


def create_patient(nik, nama, tanggal_lahir, jenis_kelamin, alamat, no_telepon, created_by_email):
    """Daftarkan pasien baru. Return id pasien, atau None kalau NIK sudah ada."""
    try:
        with get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO patients (nik, nama, tanggal_lahir, jenis_kelamin, alamat, no_telepon, created_at, created_by_email)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (nik, nama, tanggal_lahir, jenis_kelamin, alamat, no_telepon, datetime.now().isoformat(), created_by_email))
            return cursor.lastrowid
    except sqlite3.IntegrityError:
        return None  # NIK sudah terdaftar


def update_patient(nik, nama, tanggal_lahir, jenis_kelamin, alamat, no_telepon):
    """Update data pasien berdasarkan NIK."""
    with get_connection() as conn:
        conn.execute("""
            UPDATE patients
            SET nama = ?, tanggal_lahir = ?, jenis_kelamin = ?, alamat = ?, no_telepon = ?
            WHERE nik = ?
        """, (nama, tanggal_lahir, jenis_kelamin, alamat, no_telepon, nik))



def get_patient_by_nik(nik):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM patients WHERE nik = ?", (nik,)).fetchone()
        return dict(row) if row else None


def get_all_patients():
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM patients ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]


def search_patients(query: str):
    """Cari pasien berdasarkan NIK atau nama (LIKE, case-insensitive)."""
    like = f"%{query}%"
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM patients
            WHERE nik LIKE ? OR nama LIKE ?
            ORDER BY created_at DESC
        """, (like, like)).fetchall()
        return [dict(row) for row in rows]


def add_medical_record(patient_id, prediction_id, catatan_dokter=None):
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO medical_records (patient_id, prediction_id, catatan_dokter, created_at)
            VALUES (?, ?, ?, ?)
        """, (patient_id, prediction_id, catatan_dokter, datetime.now().isoformat()))
        return cursor.lastrowid


def get_medical_records_by_patient(patient_id):
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT mr.id, mr.catatan_dokter, mr.created_at, mr.status_verifikasi,
                   p.id AS prediction_id, p.filename, p.prediction_label, p.prediction_confidence,
                   p.gemini_explanation, p.gradcam_path
            FROM medical_records mr
            JOIN predictions p ON mr.prediction_id = p.id
            WHERE mr.patient_id = ?
            ORDER BY mr.created_at DESC
        """, (patient_id,)).fetchall()
        return [dict(row) for row in rows]


def get_all_medical_records():
    """Ambil SEMUA rekam medis dari SEMUA pasien, buat panel admin."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT mr.id, mr.created_at, mr.catatan_dokter,
                   p.nik, p.nama AS nama_pasien,
                   pr.id AS prediction_id, pr.prediction_label, pr.prediction_confidence, pr.gemini_explanation
            FROM medical_records mr
            JOIN patients p ON mr.patient_id = p.id
            JOIN predictions pr ON mr.prediction_id = pr.id
            ORDER BY mr.created_at DESC
        """).fetchall()
        return [dict(row) for row in rows]


def get_medical_record_by_id(record_id):
    """Ambil 1 rekam medis lengkap (join pasien + prediksi) buat halaman laporan."""
    with get_connection() as conn:
        row = conn.execute("""
            SELECT mr.id, mr.created_at, mr.catatan_dokter,
                   mr.status_verifikasi, mr.dokter_nama, mr.dokter_sip, mr.tanggal_verifikasi,
                   p.nik, p.nama AS nama_pasien, p.tanggal_lahir, p.jenis_kelamin, p.alamat, p.no_telepon,
                   pr.id AS prediction_id, pr.filename, pr.upload_time, pr.precheck_status, pr.precheck_confidence,
                   pr.prediction_label, pr.prediction_confidence, pr.all_probabilities,
                   pr.gradcam_path, pr.gemini_explanation, pr.modalitas, pr.sequence_type,
                   pr.informed_consent, pr.consent_note
            FROM medical_records mr
            JOIN patients p ON mr.patient_id = p.id
            JOIN predictions pr ON mr.prediction_id = pr.id
            WHERE mr.id = ?
        """, (record_id,)).fetchone()
        return dict(row) if row else None


def verify_medical_record(record_id, dokter_nama, dokter_sip, status_verifikasi):
    """Dokter/admin mengisi verifikasi manual atas 1 hasil pemeriksaan."""
    with get_connection() as conn:
        conn.execute("""
            UPDATE medical_records
            SET dokter_nama = ?, dokter_sip = ?, status_verifikasi = ?, tanggal_verifikasi = ?
            WHERE id = ?
        """, (dokter_nama, dokter_sip, status_verifikasi, datetime.now().isoformat(), record_id))